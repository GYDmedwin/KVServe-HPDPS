"""
TileLang KV Cache 混合精度量化算子 (Fused Operator)
========================================================
【整体架构说明】
本文件用 TileLang（一种比 Triton 更底层的 GPU 编程 DSL）
将以下三个步骤融合成 1 个 GPU Kernel，彻底消灭中间显存分配：

  原版 2-Step 流程（慢 ~21ms / 32层）:
    Step 1: KVServeTransformer.transform()  -- Hadamard 变换（含随机符号）
    Step 2: KVServeQuantizer.quantize()     -- 混合精度截断至 uint8

  融合后 1-Step 流程（快 ~2.8ms / 32层）:
    fused_compress()                        -- 变换+求极值+量化 一气呵成

【TileLang 的使用方式】
1. 用 @T.prim_func 装饰器定义一个「伪代码函数」
2. 在其中使用 T.alloc_shared / T.alloc_fragment 规划物理显存层级
3. 调用 tilelang.compile(fn, target="cuda") 触发 JIT 编译
4. 编译后返回的对象可直接像普通函数一样调用（传入 PyTorch Tensor）
"""

import math
import os
import torch
import tilelang
import tilelang.language as T
from typing import Tuple


# ===========================================================================
# 辅助函数：Rademacher 随机符号生成（CPU 端，每层只调用一次，结果永久缓存）
# ===========================================================================
# 【作用】：生成随机的 +1/-1 符号矩阵，用于对 Hadamard 矩阵做「随机旋转」。
# 这样可以破坏 KV Cache 里的异常値（Outliers），大幅提升量化精度。
# 【关键优化】：这个函数使用固定 seed，每次对相同 (layer_id, head_id)
# 永远生成同一个符号矩阵，保证压缩/解压缩的一致性，同时只需生成一次。

def make_rademacher_signs(
    layer_id: int,     # 第几层 Transformer Layer
    num_heads: int,    # 注意力头数量
    head_dim: int,     # 每个头的特征维度（通常是 128）
    base_seed: int = 0xC0FEBABE,  # 基础随机种子（固定就保证可复现）
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    signs_list = []
    for h in range(num_heads):
        # 对每个 (层, 头) 对，用 XOR 混合生成唯一种子，保证不同头有不同符号
        seed = base_seed ^ (layer_id << 16) ^ h
        gen.manual_seed(seed)
        s = torch.randint(0, 2, (head_dim,), generator=gen, dtype=torch.int8)
        s = s.float().mul_(2).sub_(1)   # 把 {0, 1} 转换成 {-1, +1}
        signs_list.append(s)
    signs = torch.stack(signs_list)     # 最终形状: [num_heads, head_dim]
    return signs.to(device=device, dtype=dtype)


# ===========================================================================
# 核心算子类：Fused TileLang Operator
# ===========================================================================
# 这个类是整个文件的核心。它在初始化时完成 TileLang JIT 编译，
# 并在 compress() 和 decompress() 方法中暴露干净的 PyTorch 接口。

from scipy.linalg import hadamard

class KVHadamardQuantOp:
    def __init__(
        self,
        num_heads:  int = 8,
        head_dim:   int = 128,
        base_seed:  int = 0xC0FEBABE,
        dtype:      torch.dtype = torch.bfloat16,
        quant_type: str = "absmax",
        **kwargs
    ):
        self.num_heads = num_heads
        self.head_dim  = head_dim
        self.base_seed = base_seed
        self.dtype     = dtype
        self.quant_type = quant_type
        # Phase-D2 knob: rows-per-CTA (tile M) of the fused kernel. Default 32
        # reproduces the original kernel bit-for-bit; larger values cut the CTA
        # count (grid_x = ceil(rows/block_M)) and thus the wave count in a small
        # SM partition. Per-row Hadamard+quant is row-independent → output is
        # identical regardless of block_M.
        self.block_M   = int(kwargs.get("block_M", 32))
        # None → 原核(block_M==32)或 block_M 参数化核; "lowfoot" → 低 footprint
        # 单融合核(去掉 out_shared 往返,fragment 上规约+量化)。
        self.fused_variant = kwargs.get("fused_variant", None)
        # throttle: >0 launches the vec_shfl reduce kernels as a persistent grid
        # of this many x-blocks (grid-stride) → caps active SMs (throttle). 0=off.
        self.throttle_gx = int(kwargs.get("throttle_gx", 0))
        self._signs    = {}  # 随机符号缓存字典，key=layer_id，避免重复生成
        self._uniform_max_vals_cache = {}
        self._token_group_map_cache = {}
        self._token_group_map_cpu_cache = {}
        
        # --- 混合精度量化配置（对应 KVServe 的 Hybrid Config）---
        # hybrid_ratio：有多少比例的头被分配「低配」量化精度
        # high_*_max_value：高配头的量化区间上限（比特数越高精度越好）
        # low_*_max_value：低配头的量化区间上限（压缩更激进，但精度略低）
        self.model_name = kwargs.get("model_name", "Llama-3.1-8B-Instruct")
        self.hybrid_ratio = kwargs.get("hybrid_ratio", 0.3)
        self.high_key_max_value = kwargs.get("high_key_max_value", 16.0)
        self.high_value_max_value = kwargs.get("high_value_max_value", 16.0)
        self.low_key_max_value = kwargs.get("low_key_max_value", 12.0)
        self.low_value_max_value = kwargs.get("low_value_max_value", 12.0)
        self.split_type = kwargs.get("split_type", "head")  # "head"按头切分 or "layer"按层切分
        self.tensor_parallel_size = kwargs.get("tensor_parallel_size", 1)
        self.tp_rank = kwargs.get("tp_rank", 0)
        
        # --- 加载混合精度掌码（和 KVServeQuantizer 使用相同的 CSV 配置）---
        # DuoConfigGenerator 会从 CSV 文件里加载每个头的「重要性分数」，
        # 分数低的头会被分配「低配」量化，分数高的头保留「高配」量化。
        try:
            from kvserve_v1.compression.config.duo_config.get_config import (
                DuoConfigGenerator)
            self.head_scores = DuoConfigGenerator.get_scores_from_csv(self.model_name)
            low_precision_num_heads = round(self.head_scores.numel() * self.hybrid_ratio)
            self.head_scores_mask = torch.zeros_like(self.head_scores, dtype=torch.bool)
            flat_indices = torch.argsort(self.head_scores.flatten())[:low_precision_num_heads]
            multi_indices = torch.unravel_index(flat_indices, self.head_scores.shape)
            self.head_scores_mask[multi_indices] = True  # True = 这个头是低配头
            self.head_scores_mask = self._get_local_head_scores_mask(self.num_heads)
            
            self.layer_scores_mask = torch.zeros(self.head_scores.shape[0], dtype=torch.bool)
            low_precision_num_layers = round(self.layer_scores_mask.shape[0] * 0.33)
            self.layer_scores_mask[-low_precision_num_layers:] = True  # 最后 33% 的层为低配
        except Exception as e:
            print(f"  [TileLang] Note: Using uniform max_vals (could not load hybrid config: {e})")
            self.head_scores_mask = None
            self.layer_scores_mask = None
            
        # --- 关键优化：预计算所有层+头的 Max_Vals，避免在热循环里动态分配 ---
        # 原则：在 __init__ 阶段一次性把所有 (层, 头) 的量化上限算好，
        # 放在 self.precomputed_k / self.precomputed_v 里。
        # 推理时直接切片（O(1) 查表），绝对不在热路径里调用 torch.full 等分配函数！
        self._max_vals_cache = {}  # key=device，把 precomputed 张量搶到对应 GPU 的缓存
        if self.head_scores_mask is not None:
            num_layers = self.head_scores_mask.shape[0]
            self.precomputed_k = torch.zeros((num_layers, self.num_heads), dtype=torch.float32)
            self.precomputed_v = torch.zeros((num_layers, self.num_heads), dtype=torch.float32)
            
            for l in range(num_layers):
                if self.split_type == "head":
                    m = self.head_scores_mask[l]  # m[i]=True 表示第 i 个头是低配头
                    self.precomputed_k[l][~m] = self.high_key_max_value   # 高配 Key 头
                    self.precomputed_k[l][m]  = self.low_key_max_value    # 低配 Key 头
                    self.precomputed_v[l][~m] = self.high_value_max_value # 高配 Value 头
                    self.precomputed_v[l][m]  = self.low_value_max_value  # 低配 Value 头
                elif self.split_type == "layer":
                    is_low = self.layer_scores_mask[l]  # 整层统一用高配或低配
                    self.precomputed_k[l][:] = self.low_key_max_value if is_low else self.high_key_max_value
                    self.precomputed_v[l][:] = self.low_value_max_value if is_low else self.high_value_max_value
        
        # --- 构建基础 Hadamard 矩阵（CPU 端，所有层共享同一个矩阵骨架）---
        # hadamard(D) 生成一个 D×D 的矩阵，元素只有 +1 和 -1。
        # 除以 sqrt(D) 做归一化，保证变换不改变向量的模长（正交变换）。
        h_base = hadamard(head_dim)
        h_tensor = torch.tensor(h_base, dtype=torch.float32) / math.sqrt(head_dim)
        self.h_base = h_tensor.cpu()  # 存在 CPU 上，后续按需搶 GPU 并和随机符号合并
        
        # --- 核心步骤：触发 TileLang JIT 编译，生成 GPU 机器码 ---
        # 这一步会花几秒钟编译，但只做一次。编译产物缓存在 self.fused_kernel 里。
        # block_M==32 走原始内核（逐位不变）；否则走 block_M 参数化变体；
        # fused_variant=="lowfoot" 走低 footprint 单融合核。
        if self.fused_variant == "fwht":
            self.fused_kernel = self._compile_fused_fwht_minmax(self.block_M)
        elif self.fused_variant == "fwht_f16":
            self.fused_kernel = self._compile_fused_fwht_minmax_f16(self.block_M)
        elif self.fused_variant == "fwht_vec":
            self.fused_kernel = self._compile_fused_fwht_minmax_vec(self.block_M)
        elif self.fused_variant == "fwht_shfl":
            self.fused_kernel = self._compile_fused_fwht_minmax_shfl(self.block_M)
        elif self.fused_variant == "fwht_r4":
            self.fused_kernel = self._compile_fused_fwht_minmax_radix4(self.block_M)
        elif self.fused_variant == "lowfoot":
            self.fused_kernel = self._compile_fused_minmax_lowfoot(self.block_M)
        elif self.block_M == 32:
            self.fused_kernel = self._compile_fused_kernel()
        else:
            self.fused_kernel = self._compile_fused_minmax_bm(self.block_M)

    def _get_local_head_scores_mask(self, num_heads: int) -> torch.Tensor:
        """Return the hybrid head mask aligned with this worker's local KV heads."""
        head_scores_mask = self.head_scores_mask
        if head_scores_mask.shape[1] == num_heads:
            return head_scores_mask

        total_heads = head_scores_mask.shape[1]
        if total_heads % num_heads != 0:
            raise ValueError(
                f"Head mask size mismatch: mask_heads={total_heads}, "
                f"tensor_heads={num_heads}."
            )

        inferred_tp_size = total_heads // num_heads
        tp_size = self.tensor_parallel_size or inferred_tp_size
        if tp_size * num_heads != total_heads:
            raise ValueError(
                f"TP mismatch: tp_size={tp_size}, tensor_heads={num_heads}, "
                f"total_heads={total_heads}."
            )
        if self.tp_rank < 0 or self.tp_rank >= tp_size:
            raise ValueError(
                f"Invalid tp_rank={self.tp_rank} for tp_size={tp_size}."
            )

        start = self.tp_rank * num_heads
        end = start + num_heads
        return head_scores_mask[:, start:end]

    def _compile_fused_kernel(self):
        """
        【TileLang 编译的入口】
        这个函数是整个文件里与 TileLang 直接打交道的唯一入口。
        它的工作方式：
          1. 用 @T.prim_func 将一个 Python 伪代码函数包装为 TileLang 的 IR
          2. 调用 tilelang.compile(...) 触发 JIT 编译 -> 返回可调用对象
          3. 调用者以后直接传入 PyTorch Tensor 即可执行 GPU 内核

        使用 rows_var 改变维度，允许这个内核处理任意 Token 数量的输入。
        """
        head_dim = self.head_dim
        num_heads = self.num_heads
        quant_type = self.quant_type
        
        import tvm
        # rows_var 是一个动态变量，表示当前处理的 Token 行数。
        # 它允许我们的内核处理任意长度的序列，而不需要针对每个长度重新编译。
        rows_var = tvm.tir.SizeVar("rows_var", "int32")
        
        if quant_type == "absmax":
            @T.prim_func
            def fused_compress(
                KV: T.Buffer([rows_var, num_heads, head_dim], "float16"),      
                H_Batched: T.Buffer([num_heads, head_dim, head_dim], "float16"), 
                Q_Out: T.Buffer([rows_var, num_heads, head_dim], "uint8"),     
                Scale_Out: T.Buffer([rows_var, num_heads], "float32"),
                Max_Vals: T.Buffer([num_heads], "float32")
            ):
                with T.Kernel(T.ceildiv(rows_var, 32), num_heads, threads=128) as (bx, by):
                    # =========================================================================
                    # 【物理显存仓库规划】: 这是 TileLang 比 Triton 更底层、更强大的精髓！
                    # 我们在这里直接规划物理硬件，明确告诉 GPU 每一个变量应该存在哪级缓存里。
                    # =========================================================================
                    
                    # 1. 放在【共享内存 (Shared Memory / L1 Cache)】里：
                    # 特点：所有线程共享，读写极快，用于块级别的数据交互。
                    kv_shared = T.alloc_shared([32, head_dim], "float16")      # 把 32 行的 KV 数据搬到超高速的 L1 缓存
                    h_shared = T.alloc_shared([head_dim, head_dim], "float16") # 把 128x128 的哈达玛矩阵搬到 L1 缓存

                    # 2. 放在【寄存器碎片 (Fragment / Registers)】里：
                    # 特点：GPU 里速度最快、最核心的计算存储器，直接喂给 Tensor Core（矩阵乘法核心）。
                    # 我们规定 Tensor Core 计算出来的矩阵乘法结果，直接落在寄存器里！
                    out_frag = T.alloc_fragment([32, head_dim], "float32")

                    # 3. 再次回到【共享内存 (Shared Memory)】：
                    # 为什么？因为寄存器是每个线程私有的。为了算这一整块的最大值(Max)，
                    # 我们必须把寄存器里的乘法结果，放回到大家都看得见的共享内存里，才能做线程间的规约(Reduction)。
                    out_shared = T.alloc_shared([32, head_dim], "float32")
                    scale_shared = T.alloc_shared([32], "float32")             # 存放算出来的 Scale 因子
                    
                    T.clear(out_frag)

                    for i, j in T.Parallel(32, head_dim):
                        row_idx = bx * 32 + i
                        if row_idx < rows_var:
                            kv_shared[i, j] = KV[row_idx, by, j]
                        else:
                            kv_shared[i, j] = 0.0

                    for i, j in T.Parallel(head_dim, head_dim):
                        h_shared[i, j] = H_Batched[by, i, j]
                        
                    T.gemm(kv_shared, h_shared, out_frag)
                    T.copy(out_frag, out_shared)

                    T.reduce_absmax(out_shared, scale_shared, dim=-1, clear=True)

                    # 【核心魔法在这里】:
                    # 此时，所有线程仍然在极速的寄存器（SRAM）中
                    for i, j in T.Parallel(32, head_dim):
                        row_idx = bx * 32 + i
                        if row_idx < rows_var:
                            if j == 0:
                                Scale_Out[row_idx, by] = scale_shared[i]

                            # 1. 瞬间判断自己是高配头还是低配头
                            # 'by' 是当前的 head_id。直接从 Max_Vals 查表拿到 15.0 还是 7.0
                            m_val = Max_Vals[by]
                            
                            # 2. 算缩放比例 (基于当前 block 的最大值)
                            inv_s = m_val / (scale_shared[i] + 1e-8)
                            
                            # 3. 乘法、Clamp 截断、在原处瞬间完成
                            clamped = T.min(T.max(out_shared[i, j] * inv_s, -m_val), m_val)
                            
                            # 4. 类型转换并【直接写回最终的物理显存】 Q_Out！中间没有任何临时 Tensor！
                            Q_Out[row_idx, by, j] = T.cast(clamped + m_val, "uint8")
                            
            return tilelang.compile(fused_compress, target="cuda")

        elif quant_type == "minmax":
            @T.prim_func
            def fused_compress(
                KV: T.Buffer([rows_var, num_heads, head_dim], "float16"),      
                H_Batched: T.Buffer([num_heads, head_dim, head_dim], "float16"), 
                Q_Out: T.Buffer([rows_var, num_heads, head_dim], "uint8"),     
                Scale_Out: T.Buffer([rows_var, num_heads], "float32"),
                Min_Out: T.Buffer([rows_var, num_heads], "float32"),
                Max_Vals: T.Buffer([num_heads], "float32")
            ):
                with T.Kernel(T.ceildiv(rows_var, 32), num_heads, threads=128) as (bx, by):
                    # =========================================================================
                    # 【物理显存仓库规划】: 掌控硬件命运的 6 行代码
                    # =========================================================================
                    
                    # 1. L1 共享缓存 (Shared Memory)：存放输入数据，方便多线程读取和喂给 Tensor Core
                    kv_shared = T.alloc_shared([32, head_dim], "float16")
                    h_shared = T.alloc_shared([head_dim, head_dim], "float16")
                    
                    # 2. 寄存器碎片 (Registers)：直接承接 Tensor Core 的核爆级计算输出，零延迟！
                    out_frag = T.alloc_fragment([32, head_dim], "float32")
                    
                    # 3. L1 共享缓存 (Shared Memory)：为了算最大值和最小值，把寄存器数据放出来给大家看
                    out_shared = T.alloc_shared([32, head_dim], "float32")
                    max_shared = T.alloc_shared([32], "float32")  # 存放 32 行数据的最大值
                    min_shared = T.alloc_shared([32], "float32")  # 存放 32 行数据的最小值
                    
                    # ---------------------------------------------------------
                    # 步骤 1：清空寄存器，准备迎接 Tensor Core 的轰炸
                    # ---------------------------------------------------------
                    T.clear(out_frag)

                    # ---------------------------------------------------------
                    # 步骤 2：多线程齐心协力，把全局显存 (KV) 搬到共享内存
                    # ---------------------------------------------------------
                    # 128 个线程被 T.Parallel 自动分配任务，每个人搬运几个元素
                    for i, j in T.Parallel(32, head_dim):
                        row_idx = bx * 32 + i
                        # 边界检查：如果超出了真实的 Token 数量，就用 0.0 填充（Padding）
                        if row_idx < rows_var:
                            kv_shared[i, j] = KV[row_idx, by, j]
                        else:
                            kv_shared[i, j] = 0.0

                    # 同样地，把哈达玛常量矩阵也搬进 L1 缓存
                    for i, j in T.Parallel(head_dim, head_dim):
                        h_shared[i, j] = H_Batched[by, i, j]
                        
                    # ---------------------------------------------------------
                    # 步骤 3：核爆时刻！硬件级张量乘法 (Tensor Core GEMM)
                    # ---------------------------------------------------------
                    # 将 kv_shared 和 h_shared 送入张量核心。
                    # 计算出的矩阵乘法结果会被直接打碎，存在各个线程的寄存器 (out_frag) 里。
                    T.gemm(kv_shared, h_shared, out_frag)
                    
                    # ---------------------------------------------------------
                    # 步骤 4：数据大串联！拼凑寄存器碎片，求得极值
                    # ---------------------------------------------------------
                    # 因为找极值需要看整行数据，而数据散落在各人口袋（寄存器）里。
                    # 所以先用 T.copy 把寄存器碎片拼回到大家都能看到的共享内存 (out_shared) 上。
                    T.copy(out_frag, out_shared)

                    # 启动硬件级的规约指令：找出这 32 行数据中，每一行的最大值和最小值
                    T.reduce_max(out_shared, max_shared, dim=-1, clear=True)
                    T.reduce_min(out_shared, min_shared, dim=-1, clear=True)

                    # ---------------------------------------------------------
                    # 步骤 5：【终极魔法】混合精度量化 + 强行入库
                    # ---------------------------------------------------------
                    for i, j in T.Parallel(32, head_dim):
                        row_idx = bx * 32 + i
                        if row_idx < rows_var:
                            c_max = max_shared[i]
                            c_min = min_shared[i]
                            
                            # 魔法 1：瞬时查表 O(1)。
                            # 'by' 是当前的头编号。去大本营 Max_Vals 里拿到这一层的这个头分配的量化精度 (63.0 or 15.0)
                            m_val = Max_Vals[by]
                            
                            # 魔法 2：计算缩放因子 Scale
                            diff = T.max(c_max - c_min, 1e-5)
                            scale = diff / m_val
                            
                            # 每行的第一个线程 (j==0) 负责把 Scale 和 Min 写入全局显存 (Metadata)
                            if j == 0:
                                Scale_Out[row_idx, by] = scale
                                Min_Out[row_idx, by] = c_min

                            inv_s = 1.0 / scale
                            # 魔法 3：取出刚才算好的哈达玛结果 (out_shared)，减去极小值，乘以缩放因子
                            val = (out_shared[i, j] - c_min) * inv_s
                            
                            # 魔法 4：四舍五入 (floor(x+0.5)) 并截断 (Clamp)。
                            # 注意：截断的上限就是我们刚刚瞬间查到的 m_val (63.0 或 15.0)！
                            clamped = T.min(T.max(T.floor(val + 0.5), 0.0), m_val)
                            
                            # 魔法 5：直接转换成 uint8 并暴力写入最终显存 (Q_Out)！没有任何中间拷贝！
                            Q_Out[row_idx, by, j] = T.cast(clamped, "uint8")
                            
            return tilelang.compile(fused_compress, target="cuda")
        else:
            raise ValueError(f"Unsupported quant_type: {quant_type}")

    def _compile_fused_minmax_bm(self, bm: int):
        """Phase-D2: block_M-parameterized copy of the minmax fused kernel.

        Math is identical to ``_compile_fused_kernel`` (minmax) — same per-row
        Hadamard GEMM, per-row min/max reduction, and quantization — but tiles
        ``bm`` rows per CTA instead of 32.  Because every row is processed
        independently, the quantized output and metadata are BIT-IDENTICAL to
        the block_M=32 path; the only change is grid_x = ceil(rows/bm), i.e. the
        CTA count and (in a small SM partition) the wave count.  This isolates
        the single causal variable 'CTA granularity → waves → cmp_span'.
        """
        if self.quant_type != "minmax":
            raise ValueError("block_M variant is implemented for minmax only")
        head_dim = self.head_dim
        num_heads = self.num_heads

        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")

        @T.prim_func
        def fused_compress(
            KV: T.Buffer([rows_var, num_heads, head_dim], "float16"),
            H_Batched: T.Buffer([num_heads, head_dim, head_dim], "float16"),
            Q_Out: T.Buffer([rows_var, num_heads, head_dim], "uint8"),
            Scale_Out: T.Buffer([rows_var, num_heads], "float32"),
            Min_Out: T.Buffer([rows_var, num_heads], "float32"),
            Max_Vals: T.Buffer([num_heads], "float32")
        ):
            with T.Kernel(T.ceildiv(rows_var, bm), num_heads, threads=128) as (bx, by):
                kv_shared = T.alloc_shared([bm, head_dim], "float16")
                h_shared = T.alloc_shared([head_dim, head_dim], "float16")
                out_frag = T.alloc_fragment([bm, head_dim], "float32")
                out_shared = T.alloc_shared([bm, head_dim], "float32")
                max_shared = T.alloc_shared([bm], "float32")
                min_shared = T.alloc_shared([bm], "float32")

                T.clear(out_frag)

                for i, j in T.Parallel(bm, head_dim):
                    row_idx = bx * bm + i
                    if row_idx < rows_var:
                        kv_shared[i, j] = KV[row_idx, by, j]
                    else:
                        kv_shared[i, j] = 0.0

                for i, j in T.Parallel(head_dim, head_dim):
                    h_shared[i, j] = H_Batched[by, i, j]

                T.gemm(kv_shared, h_shared, out_frag)
                T.copy(out_frag, out_shared)

                T.reduce_max(out_shared, max_shared, dim=-1, clear=True)
                T.reduce_min(out_shared, min_shared, dim=-1, clear=True)

                for i, j in T.Parallel(bm, head_dim):
                    row_idx = bx * bm + i
                    if row_idx < rows_var:
                        c_max = max_shared[i]
                        c_min = min_shared[i]
                        m_val = Max_Vals[by]
                        diff = T.max(c_max - c_min, 1e-5)
                        scale = diff / m_val
                        if j == 0:
                            Scale_Out[row_idx, by] = scale
                            Min_Out[row_idx, by] = c_min
                        inv_s = 1.0 / scale
                        val = (out_shared[i, j] - c_min) * inv_s
                        clamped = T.min(T.max(T.floor(val + 0.5), 0.0), m_val)
                        Q_Out[row_idx, by, j] = T.cast(clamped, "uint8")

        return tilelang.compile(fused_compress, target="cuda")

    def _compile_fused_minmax_lowfoot(self, bm: int = 32):
        """Phase-D capstone: low-footprint, high-occupancy single fused kernel.

        Same minmax math as ``_compile_fused_kernel`` (per-row Hadamard GEMM →
        per-row min/max → quantize) and BIT-IDENTICAL output, but **drops the
        16KB ``out_shared`` round-trip**: the GEMM accumulator stays in the
        fragment, per-row min/max are reduced on the fragment, and quantization
        writes out directly from the fragment (flash-attention style — rowmax/
        rowmin then in-place rescale).  Fewer shared bytes per CTA → higher
        occupancy, while staying a SINGLE kernel (= the "少核 + 高占用" target).
        ``bm`` = rows/CTA (default 32); smaller bm further cuts the accumulator
        register pressure.
        """
        if self.quant_type != "minmax":
            raise ValueError("low-footprint variant is implemented for minmax only")
        head_dim = self.head_dim
        num_heads = self.num_heads

        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")

        @T.prim_func
        def fused_compress(
            KV: T.Buffer([rows_var, num_heads, head_dim], "float16"),
            H_Batched: T.Buffer([num_heads, head_dim, head_dim], "float16"),
            Q_Out: T.Buffer([rows_var, num_heads, head_dim], "uint8"),
            Scale_Out: T.Buffer([rows_var, num_heads], "float32"),
            Min_Out: T.Buffer([rows_var, num_heads], "float32"),
            Max_Vals: T.Buffer([num_heads], "float32")
        ):
            with T.Kernel(T.ceildiv(rows_var, bm), num_heads, threads=128) as (bx, by):
                kv_shared = T.alloc_shared([bm, head_dim], "float16")
                h_shared = T.alloc_shared([head_dim, head_dim], "float16")
                out_frag = T.alloc_fragment([bm, head_dim], "float32")
                # min/max kept as fragments — no full-tile out_shared (saves bm*head_dim*4 B)
                max_frag = T.alloc_fragment([bm], "float32")
                min_frag = T.alloc_fragment([bm], "float32")

                T.clear(out_frag)

                for i, j in T.Parallel(bm, head_dim):
                    row_idx = bx * bm + i
                    if row_idx < rows_var:
                        kv_shared[i, j] = KV[row_idx, by, j]
                    else:
                        kv_shared[i, j] = 0.0

                for i, j in T.Parallel(head_dim, head_dim):
                    h_shared[i, j] = H_Batched[by, i, j]

                T.gemm(kv_shared, h_shared, out_frag)

                # rowmax / rowmin directly on the accumulator fragment
                T.reduce_max(out_frag, max_frag, dim=-1, clear=True)
                T.reduce_min(out_frag, min_frag, dim=-1, clear=True)

                for i, j in T.Parallel(bm, head_dim):
                    row_idx = bx * bm + i
                    if row_idx < rows_var:
                        c_max = max_frag[i]
                        c_min = min_frag[i]
                        m_val = Max_Vals[by]
                        diff = T.max(c_max - c_min, 1e-5)
                        scale = diff / m_val
                        if j == 0:
                            Scale_Out[row_idx, by] = scale
                            Min_Out[row_idx, by] = c_min
                        inv_s = 1.0 / scale
                        val = (out_frag[i, j] - c_min) * inv_s
                        clamped = T.min(T.max(T.floor(val + 0.5), 0.0), m_val)
                        Q_Out[row_idx, by, j] = T.cast(clamped, "uint8")

        return tilelang.compile(fused_compress, target="cuda")

    def _compile_fused_fwht_minmax(self, bm: int = 32):
        """Phase-D capstone++ : the true "少核 + 高占用" single fused kernel.

        Combines BOTH levers from the partition study into ONE kernel:
          * **FWHT butterfly** (7-stage, O(n log n)) instead of the Hadamard
            GEMM → needs only the [num_heads, head_dim] sign vector, **not the
            128×128 H matrix in smem** → smem drops 40KB→~16KB → occupancy up
            (this is v3's occupancy lever, but v3 spends it across 11 kernels).
          * **fragment/shared-only reduce + in-place quantize** (lowfoot's
            traffic lever) → no Had_Out write-out, no separate quant kernel.
        So: 1 kernel, high occupancy, low traffic — exactly the target that
        neither `compress` (1 kernel, low occ) nor `compress_v3` (high occ, 11
        kernels) reaches.

        Transform math equals `compress` ((kv⊙signs)·H/√d then per-token minmax)
        but computed in **f32** (vs GEMM's f16 H), so output is *numerically
        equivalent, not bit-identical* — validated by roundtrip error, not q==.
        Inverse (`decompress`) is unchanged (H is symmetric+orthogonal, signs
        self-inverse), so the existing decode reconstructs it correctly.
        """
        if self.quant_type != "minmax":
            raise ValueError("FWHT fused variant is implemented for minmax only")
        head_dim = self.head_dim
        num_heads = self.num_heads
        log_dim = int(math.log2(head_dim))
        assert (1 << log_dim) == head_dim, f"head_dim must be pow2, got {head_dim}"
        scale_val = 1.0 / math.sqrt(head_dim)

        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")

        @T.prim_func
        def fused_fwht(
            KV: T.Buffer([rows_var, num_heads, head_dim], "float16"),
            Signs: T.Buffer([num_heads, head_dim], "float16"),
            Q_Out: T.Buffer([rows_var, num_heads, head_dim], "uint8"),
            Scale_Out: T.Buffer([rows_var, num_heads], "float32"),
            Min_Out: T.Buffer([rows_var, num_heads], "float32"),
            Max_Vals: T.Buffer([num_heads], "float32")
        ):
            with T.Kernel(T.ceildiv(rows_var, bm), num_heads, threads=128) as (bx, by):
                data = T.alloc_shared([bm, head_dim], "float32")
                max_shared = T.alloc_shared([bm], "float32")
                min_shared = T.alloc_shared([bm], "float32")

                # load + Rademacher signs (in f32, like the reference FWHT kernel)
                for i, j in T.Parallel(bm, head_dim):
                    row_idx = bx * bm + i
                    if row_idx < rows_var:
                        data[i, j] = T.cast(KV[row_idx, by, j], "float32") * \
                            T.cast(Signs[by, j], "float32")
                    else:
                        data[i, j] = 0.0

                # 7-stage in-place FWHT butterfly (no H matrix staged)
                for s in range(log_dim):
                    half = 1 << s
                    for i, j in T.Parallel(bm, head_dim // 2):
                        group = j // half
                        offset = j % half
                        idx0 = group * (2 * half) + offset
                        idx1 = idx0 + half
                        a = data[i, idx0]
                        b = data[i, idx1]
                        data[i, idx0] = a + b
                        data[i, idx1] = a - b

                for i, j in T.Parallel(bm, head_dim):
                    data[i, j] = data[i, j] * T.float32(scale_val)

                # per-token (per-row) min/max, then fused in-place quantize
                T.reduce_max(data, max_shared, dim=-1, clear=True)
                T.reduce_min(data, min_shared, dim=-1, clear=True)

                for i, j in T.Parallel(bm, head_dim):
                    row_idx = bx * bm + i
                    if row_idx < rows_var:
                        c_max = max_shared[i]
                        c_min = min_shared[i]
                        m_val = Max_Vals[by]
                        diff = T.max(c_max - c_min, 1e-5)
                        scale = diff / m_val
                        if j == 0:
                            Scale_Out[row_idx, by] = scale
                            Min_Out[row_idx, by] = c_min
                        inv_s = 1.0 / scale
                        val = (data[i, j] - c_min) * inv_s
                        clamped = T.min(T.max(T.floor(val + 0.5), 0.0), m_val)
                        Q_Out[row_idx, by, j] = T.cast(clamped, "uint8")

        return tilelang.compile(fused_fwht, target="cuda")

    def _compile_fused_fwht_minmax_radix4(self, bm: int = 32):
        """[co-design Route A] radix-4 FWHT butterfly — cut SMEM/LSU passes.

        Diagnosis (ncu): the radix-2 `_compile_fused_fwht_minmax` butterfly is
        LSU-bound (pipe_lsu 81%) — 7 stages × (read+write) per element = 14 SMEM
        ops/element. A radix-4 butterfly fuses two radix-2 stages into one 4-point
        Walsh-Hadamard combine → for head_dim=128 (log2=7): 3 radix-4 stages
        (h=1,4,16) + 1 radix-2 remainder (h=64) = 4 passes → 8 SMEM ops/element
        (−43%). Goal: drop pipe_lsu 81%→~46% so the compress stops saturating the
        SM's LSU pipe and overlaps more behind a TC-bound GEMM. Math identical to
        radix-2 (same minmax, 8-bit out) → numerically equivalent, decode unchanged.
        """
        if self.quant_type != "minmax":
            raise ValueError("FWHT radix-4 fused variant is implemented for minmax only")
        head_dim = self.head_dim
        num_heads = self.num_heads
        log_dim = int(math.log2(head_dim))
        assert (1 << log_dim) == head_dim, f"head_dim must be pow2, got {head_dim}"
        scale_val = 1.0 / math.sqrt(head_dim)

        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")

        @T.prim_func
        def fused_fwht_r4(
            KV: T.Buffer([rows_var, num_heads, head_dim], "float16"),
            Signs: T.Buffer([num_heads, head_dim], "float16"),
            Q_Out: T.Buffer([rows_var, num_heads, head_dim], "uint8"),
            Scale_Out: T.Buffer([rows_var, num_heads], "float32"),
            Min_Out: T.Buffer([rows_var, num_heads], "float32"),
            Max_Vals: T.Buffer([num_heads], "float32")
        ):
            with T.Kernel(T.ceildiv(rows_var, bm), num_heads, threads=128) as (bx, by):
                # ping-pong double buffer: radix-4 reads src / writes dst (no
                # in-place 4-way scatter — TileLang's thread-sync analysis chokes
                # on the in-place radix-4 pattern; out-of-place is clean).
                bufs = [T.alloc_shared([bm, head_dim], "float32"),
                        T.alloc_shared([bm, head_dim], "float32")]
                max_shared = T.alloc_shared([bm], "float32")
                min_shared = T.alloc_shared([bm], "float32")

                for i, j in T.Parallel(bm, head_dim):
                    row_idx = bx * bm + i
                    if row_idx < rows_var:
                        bufs[0][i, j] = T.cast(KV[row_idx, by, j], "float32") * \
                            T.cast(Signs[by, j], "float32")
                    else:
                        bufs[0][i, j] = 0.0
                T.sync_threads()

                # ping-pong via Python int buffer index (no alias var reassignment,
                # which the TileLang eager builder rejects). cur/nxt are Python ints.
                # Explicit barriers between stages: the cross-buffer RAW dependency
                # is NOT auto-synced by TileLang (unlike the in-place radix-2 path).
                cur = 0
                # radix-4 stages (fuse two radix-2 levels each): stride h4 = 1, 4, 16
                for s4 in range(log_dim // 2):
                    h4 = 1 << (2 * s4)
                    nxt = 1 - cur
                    for i, j in T.Parallel(bm, head_dim // 4):
                        g4 = j // h4
                        o4 = j % h4
                        p0 = g4 * (4 * h4) + o4
                        e0 = bufs[cur][i, p0]
                        e1 = bufs[cur][i, p0 + h4]
                        e2 = bufs[cur][i, p0 + 2 * h4]
                        e3 = bufs[cur][i, p0 + 3 * h4]
                        bufs[nxt][i, p0] = e0 + e1 + e2 + e3
                        bufs[nxt][i, p0 + h4] = e0 - e1 + e2 - e3
                        bufs[nxt][i, p0 + 2 * h4] = e0 + e1 - e2 - e3
                        bufs[nxt][i, p0 + 3 * h4] = e0 - e1 - e2 + e3
                    T.sync_threads()
                    cur = nxt
                # remainder radix-2 stage if log_dim is odd (stride h2 = 64)
                if log_dim % 2 == 1:
                    h2 = 1 << (log_dim - 1)
                    nxt = 1 - cur
                    for i, j in T.Parallel(bm, head_dim // 2):
                        g2 = j // h2
                        o2 = j % h2
                        r0 = g2 * (2 * h2) + o2
                        r1 = r0 + h2
                        av = bufs[cur][i, r0]
                        bv = bufs[cur][i, r1]
                        bufs[nxt][i, r0] = av + bv
                        bufs[nxt][i, r1] = av - bv
                    T.sync_threads()
                    cur = nxt
                fin = cur  # final buffer index holding the transformed data

                for i, j in T.Parallel(bm, head_dim):
                    bufs[fin][i, j] = bufs[fin][i, j] * T.float32(scale_val)

                T.reduce_max(bufs[fin], max_shared, dim=-1, clear=True)
                T.reduce_min(bufs[fin], min_shared, dim=-1, clear=True)

                for i, j in T.Parallel(bm, head_dim):
                    row_idx = bx * bm + i
                    if row_idx < rows_var:
                        c_max = max_shared[i]
                        c_min = min_shared[i]
                        m_val = Max_Vals[by]
                        diff = T.max(c_max - c_min, 1e-5)
                        scale = diff / m_val
                        if j == 0:
                            Scale_Out[row_idx, by] = scale
                            Min_Out[row_idx, by] = c_min
                        inv_s = 1.0 / scale
                        val = (bufs[fin][i, j] - c_min) * inv_s
                        clamped = T.min(T.max(T.floor(val + 0.5), 0.0), m_val)
                        Q_Out[row_idx, by, j] = T.cast(clamped, "uint8")

        return tilelang.compile(fused_fwht_r4, target="cuda")

    def _compile_fused_fwht_minmax_f16(self, bm: int = 32):
        """[co-design Edit C] FWHT fused kernel with the in-SMEM butterfly buffer
        kept in **float16** instead of float32.

        Motivation (ncu, kernel_factor_probe): the float32 `_compile_fused_fwht_minmax`
        kernel allocates bm*head_dim*4 = 16.6 KB dynamic SMEM and is occupancy-
        bound by SMEM (≈5 blocks) ≈ registers (≈6). Halving the SMEM buffer to
        bm*head_dim*2 = 8.3 KB removes SMEM as the binding limiter → occupancy
        rises to the register limit. Output stays 8-bit uint8 (same OUTPUT
        precision); only the INTERNAL transform-compute precision drops to fp16
        — a kernel-level footprint↔accuracy knob, validated by roundtrip error.
        """
        if self.quant_type != "minmax":
            raise ValueError("FWHT-f16 fused variant is implemented for minmax only")
        head_dim = self.head_dim
        num_heads = self.num_heads
        log_dim = int(math.log2(head_dim))
        assert (1 << log_dim) == head_dim, f"head_dim must be pow2, got {head_dim}"
        scale_val = 1.0 / math.sqrt(head_dim)

        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")

        @T.prim_func
        def fused_fwht_f16(
            KV: T.Buffer([rows_var, num_heads, head_dim], "float16"),
            Signs: T.Buffer([num_heads, head_dim], "float16"),
            Q_Out: T.Buffer([rows_var, num_heads, head_dim], "uint8"),
            Scale_Out: T.Buffer([rows_var, num_heads], "float32"),
            Min_Out: T.Buffer([rows_var, num_heads], "float32"),
            Max_Vals: T.Buffer([num_heads], "float32")
        ):
            with T.Kernel(T.ceildiv(rows_var, bm), num_heads, threads=128) as (bx, by):
                data = T.alloc_shared([bm, head_dim], "float16")     # ½ SMEM vs f32
                max_shared = T.alloc_shared([bm], "float16")
                min_shared = T.alloc_shared([bm], "float16")

                for i, j in T.Parallel(bm, head_dim):
                    row_idx = bx * bm + i
                    if row_idx < rows_var:
                        data[i, j] = KV[row_idx, by, j] * Signs[by, j]
                    else:
                        data[i, j] = T.float16(0.0)

                # 7-stage in-place FWHT butterfly, all in fp16
                for s in range(log_dim):
                    half = 1 << s
                    for i, j in T.Parallel(bm, head_dim // 2):
                        group = j // half
                        offset = j % half
                        idx0 = group * (2 * half) + offset
                        idx1 = idx0 + half
                        a = data[i, idx0]
                        b = data[i, idx1]
                        data[i, idx0] = a + b
                        data[i, idx1] = a - b

                for i, j in T.Parallel(bm, head_dim):
                    data[i, j] = data[i, j] * T.float16(scale_val)

                T.reduce_max(data, max_shared, dim=-1, clear=True)
                T.reduce_min(data, min_shared, dim=-1, clear=True)

                for i, j in T.Parallel(bm, head_dim):
                    row_idx = bx * bm + i
                    if row_idx < rows_var:
                        c_max = T.cast(max_shared[i], "float32")
                        c_min = T.cast(min_shared[i], "float32")
                        m_val = Max_Vals[by]
                        diff = T.max(c_max - c_min, 1e-5)
                        scale = diff / m_val
                        if j == 0:
                            Scale_Out[row_idx, by] = scale
                            Min_Out[row_idx, by] = c_min
                        inv_s = 1.0 / scale
                        val = (T.cast(data[i, j], "float32") - c_min) * inv_s
                        clamped = T.min(T.max(T.floor(val + 0.5), 0.0), m_val)
                        Q_Out[row_idx, by, j] = T.cast(clamped, "uint8")

        return tilelang.compile(fused_fwht_f16, target="cuda")

    def _compile_fused_fwht_minmax_vec(self, bm: int = 32):
        """[co-design Route A] Register-resident FWHT butterfly.

        Diagnosis (ncu, prefill_cap_designsweep): `_compile_fused_fwht_minmax`
        keeps `data` in SHARED memory and runs 7 in-place butterfly stages, each
        doing 2 SMEM loads + 2 SMEM stores per pair → the kernel is LSU-bound
        (sm__inst_executed_pipe_lsu = 81%), which is what makes it a BAD co-runner
        with a TC-bound prefill GEMM (overlap caps ~12-15%). A minimal HBM-bound
        copy hides ~40% behind the same GEMM → the cap is the butterfly's LSU/SMEM
        instruction stream, not a regime ceiling.

        This variant allocates `data` as a register FRAGMENT instead of shared
        memory. The 7 butterfly stages then read/write registers (no LSU/SMEM
        round-trips); TileLang's layout inference handles the few cross-lane
        exchanges. Goal: push sm__throughput from 81% (LSU) toward the ~40%
        HBM-bound copy floor → lift prefill∥compress overlap. Output bit-format
        unchanged (8-bit uint8, same minmax math) → decode path untouched.
        """
        if self.quant_type != "minmax":
            raise ValueError("FWHT-vec fused variant is implemented for minmax only")
        head_dim = self.head_dim
        num_heads = self.num_heads
        log_dim = int(math.log2(head_dim))
        assert (1 << log_dim) == head_dim, f"head_dim must be pow2, got {head_dim}"
        scale_val = 1.0 / math.sqrt(head_dim)

        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")

        @T.prim_func
        def fused_fwht_vec(
            KV: T.Buffer([rows_var, num_heads, head_dim], "float16"),
            Signs: T.Buffer([num_heads, head_dim], "float16"),
            Q_Out: T.Buffer([rows_var, num_heads, head_dim], "uint8"),
            Scale_Out: T.Buffer([rows_var, num_heads], "float32"),
            Min_Out: T.Buffer([rows_var, num_heads], "float32"),
            Max_Vals: T.Buffer([num_heads], "float32")
        ):
            with T.Kernel(T.ceildiv(rows_var, bm), num_heads, threads=128) as (bx, by):
                data = T.alloc_fragment([bm, head_dim], "float32")   # registers, not SMEM
                max_f = T.alloc_fragment([bm], "float32")
                min_f = T.alloc_fragment([bm], "float32")

                for i, j in T.Parallel(bm, head_dim):
                    row_idx = bx * bm + i
                    if row_idx < rows_var:
                        data[i, j] = T.cast(KV[row_idx, by, j], "float32") * \
                            T.cast(Signs[by, j], "float32")
                    else:
                        data[i, j] = 0.0

                # 7-stage in-place FWHT butterfly, all in registers
                for s in range(log_dim):
                    half = 1 << s
                    for i, j in T.Parallel(bm, head_dim // 2):
                        group = j // half
                        offset = j % half
                        idx0 = group * (2 * half) + offset
                        idx1 = idx0 + half
                        a = data[i, idx0]
                        b = data[i, idx1]
                        data[i, idx0] = a + b
                        data[i, idx1] = a - b

                for i, j in T.Parallel(bm, head_dim):
                    data[i, j] = data[i, j] * T.float32(scale_val)

                T.reduce_max(data, max_f, dim=-1, clear=True)
                T.reduce_min(data, min_f, dim=-1, clear=True)

                for i, j in T.Parallel(bm, head_dim):
                    row_idx = bx * bm + i
                    if row_idx < rows_var:
                        c_max = max_f[i]
                        c_min = min_f[i]
                        m_val = Max_Vals[by]
                        diff = T.max(c_max - c_min, 1e-5)
                        scale = diff / m_val
                        if j == 0:
                            Scale_Out[row_idx, by] = scale
                            Min_Out[row_idx, by] = c_min
                        inv_s = 1.0 / scale
                        val = (data[i, j] - c_min) * inv_s
                        clamped = T.min(T.max(T.floor(val + 0.5), 0.0), m_val)
                        Q_Out[row_idx, by, j] = T.cast(clamped, "uint8")

        return tilelang.compile(fused_fwht_vec, target="cuda")

    def _compile_fused_fwht_minmax_shfl(self, bm: int = 32):
        """[co-design Route B] Warp-shuffle FWHT — one row per warp, no SMEM.

        head_dim=128 over a 32-lane warp = 4 elements/lane. Butterfly stages with
        half∈{1,2} are WITHIN a lane's 4 registers (pure register ops); stages
        half∈{4,8,...,64} cross lanes → exchanged via warp shuffle. No shared
        memory and no SMEM round-trip at all → moves the work off the saturated
        LSU pipe entirely. Most aggressive route; expressed here via a fragment
        with a warp-mapped layout so TileLang lowers cross-lane stages to shuffles.

        NOTE: bm rows are processed by bm warps (threads = bm*32). Output format
        unchanged (8-bit), decode path untouched.
        """
        if self.quant_type != "minmax":
            raise ValueError("FWHT-shfl fused variant is implemented for minmax only")
        head_dim = self.head_dim
        num_heads = self.num_heads
        log_dim = int(math.log2(head_dim))
        assert (1 << log_dim) == head_dim, f"head_dim must be pow2, got {head_dim}"
        scale_val = 1.0 / math.sqrt(head_dim)
        WARP = 32
        per_lane = head_dim // WARP            # 4 elements/lane for head_dim=128
        n_warps = bm                            # one warp per row
        threads = n_warps * WARP

        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")

        @T.prim_func
        def fused_fwht_shfl(
            KV: T.Buffer([rows_var, num_heads, head_dim], "float16"),
            Signs: T.Buffer([num_heads, head_dim], "float16"),
            Q_Out: T.Buffer([rows_var, num_heads, head_dim], "uint8"),
            Scale_Out: T.Buffer([rows_var, num_heads], "float32"),
            Min_Out: T.Buffer([rows_var, num_heads], "float32"),
            Max_Vals: T.Buffer([num_heads], "float32")
        ):
            with T.Kernel(T.ceildiv(rows_var, bm), num_heads, threads=threads) as (bx, by):
                tid = T.get_thread_binding()
                lane = tid % WARP
                warp = tid // WARP
                reg = T.alloc_local([per_lane], "float32")
                row_idx = bx * bm + warp

                # each lane loads its `per_lane` contiguous channels (coalesced)
                for c in T.serial(per_lane):
                    ch = lane * per_lane + c
                    if row_idx < rows_var:
                        reg[c] = T.cast(KV[row_idx, by, ch], "float32") * \
                            T.cast(Signs[by, ch], "float32")
                    else:
                        reg[c] = 0.0

                # FWHT stages (Python-unrolled; per_lane is a compile constant)
                for s in range(log_dim):
                    half = 1 << s
                    if half < per_lane:
                        # within a lane's own registers — static indices
                        for c in range(per_lane):
                            if (c % (2 * half)) < half:
                                a = reg[c]
                                b = reg[c + half]
                                reg[c] = a + b
                                reg[c + half] = a - b
                    else:
                        # cross-lane: partner lane = lane ^ lane_mask, same c offset
                        lane_mask = half // per_lane
                        for c in range(per_lane):
                            partner = T.shfl_xor(reg[c], lane_mask)
                            reg[c] = T.if_then_else((lane & lane_mask) == 0,
                                                    reg[c] + partner, partner - reg[c])

                # scale
                for c in range(per_lane):
                    reg[c] = reg[c] * T.float32(scale_val)

                # per-row min/max: local over regs, then warp-reduce over 32 lanes
                lmax = T.alloc_local([1], "float32")
                lmin = T.alloc_local([1], "float32")
                lmax[0] = reg[0]; lmin[0] = reg[0]
                for c in range(per_lane):
                    lmax[0] = T.max(lmax[0], reg[c])
                    lmin[0] = T.min(lmin[0], reg[c])
                log_warp = int(math.log2(WARP))
                for off in range(log_warp - 1, -1, -1):   # 16,8,4,2,1 for WARP=32
                    step = 1 << off
                    lmax[0] = T.max(lmax[0], T.shfl_xor(lmax[0], step))
                    lmin[0] = T.min(lmin[0], T.shfl_xor(lmin[0], step))

                m_val = Max_Vals[by]
                diff = T.max(lmax[0] - lmin[0], 1e-5)
                scale = diff / m_val
                inv_s = 1.0 / scale
                if row_idx < rows_var and lane == 0:
                    Scale_Out[row_idx, by] = scale
                    Min_Out[row_idx, by] = lmin[0]
                for c in T.serial(per_lane):
                    ch = lane * per_lane + c
                    if row_idx < rows_var:
                        val = (reg[c] - lmin[0]) * inv_s
                        clamped = T.min(T.max(T.floor(val + 0.5), 0.0), m_val)
                        Q_Out[row_idx, by, ch] = T.cast(clamped, "uint8")

        return tilelang.compile(fused_fwht_shfl, target="cuda")

    def _get_signs(self, layer_id: int, device) -> torch.Tensor:
        """
        带缓存的随机符号获取函数。
        第一次调用时会在 CPU 上生成并将结果转到 GPU，之后直接返回缓存。
        """
        if layer_id not in self._signs:
            self._signs[layer_id] = make_rademacher_signs(
                layer_id, self.num_heads, self.head_dim,
                self.base_seed, str(device), self.dtype,
            )
        return self._signs[layer_id]

    def _get_batched_hadamard(self, layer_id: int, device) -> torch.Tensor:
        """
        返回形状为 [num_heads, head_dim, head_dim] 的。2层缓存字典。
        内容 = Hadamard 基础矩阵 * 随机符号 （即 Randomized Hadamard Transform）。

        【性能关键】这里是消灭 CPU 卡顿的重第工作。过去每次调用都要：
            CPU 生成随机数 -> 拷贝到 GPU -> 转换类型
        现在增加了大本营，所有和 layer_id 对应的矩阵只渲染一次。
        """
        if not hasattr(self, '_h_batched_cache'):
            self._h_batched_cache = {}
            
        if device not in self._h_batched_cache:
            self._h_batched_cache[device] = {}
            
        if layer_id not in self._h_batched_cache[device]:
            signs = self._get_signs(layer_id, device)  # [num_heads, head_dim]
            # h_base 是 [head_dim, head_dim]
            # signs_exp 是 [num_heads, head_dim, 1]
            # 广播相乘：每个头用自己的符号行向对 h_base 的对应行进行逐元素相乘
            signs_exp = signs.unsqueeze(-1)
            h_batched = (signs_exp * self.h_base.to(device)).to(torch.float16)
            self._h_batched_cache[device][layer_id] = h_batched
            
        return self._h_batched_cache[device][layer_id]

    def compress(
        self,
        kv: torch.Tensor,    # 输入形状: [num_blocks, block_size, num_heads, head_dim], bfloat16
        layer_id: int,       # 当前是第几层（用于查找预计算的 Max_Vals）
        is_key: bool = True, # True = 处理 K，False = 处理 V（K/V 可能用不同的量化上限）
        max_val: float = None, # 强行覆盖参数（方便测试）
    ) -> Tuple[torch.Tensor, dict]:
        """
        对外公开 API：将 KV Cache 应用融合量化并返回压缩结果。
        返回内容：
            q_kv     : uint8 类型，形状与输入 kv 相同
            metadata : 包含解码所需信息的字典（scales, mins, max_vals）
        """
        assert kv.is_cuda and kv.dtype == torch.bfloat16
        num_blocks, block_size, num_heads, head_dim = kv.shape
        rows = num_blocks * block_size  # 把块和时间步展开成总行数

        # 整形并转换为 float16（TileLang 内核的要求）
        kv_flat = kv.reshape(rows, num_heads, head_dim).to(torch.float16)
        # FWHT 变体只需符号向量[heads,hd];其余变体用 [heads,hd,hd] 哈达玛矩阵
        if self.fused_variant in ("fwht", "fwht_f16", "fwht_vec", "fwht_shfl", "fwht_r4"):
            h_batched = self._get_signs(layer_id, kv.device).to(torch.float16)
        else:
            h_batched = self._get_batched_hadamard(layer_id, kv.device)

        # 预分配输出空间。注意：这里的 torch.empty 是必要开销，
        # 因为 GPU Kernel 需要一个地方写回结果。
        q_kv_flat = torch.empty((rows, num_heads, head_dim), dtype=torch.uint8, device=kv.device)
        scales = torch.empty((rows, num_heads), dtype=torch.float32, device=kv.device)

        # 获取每个头的量化上限（Max_Vals）—— O(1) 查表，是符合防碌低延迟的设计
        if max_val is not None:
            # 测试模式：强行用指定常数场
            max_vals = torch.full((self.num_heads,), max_val, dtype=torch.float32, device=kv.device)
        else:
            if self.head_scores_mask is not None and layer_id < self.precomputed_k.shape[0]:
                # 生产模式：从预计算缓存中拿到当前层当前 K 或 V 的混合精度配置
                if kv.device not in self._max_vals_cache:
                    # 第一次在新设备上使用时，把预计算值搬到该设备
                    self._max_vals_cache[kv.device] = (
                        self.precomputed_k.to(kv.device),
                        self.precomputed_v.to(kv.device)
                    )
                cache_k, cache_v = self._max_vals_cache[kv.device]
                # 第 layer_id 行就是这一层所有头的量化上限向量
                max_vals = cache_k[layer_id] if is_key else cache_v[layer_id]
            else:
                # 如果没有 CSV 配置，所有头全部用高配值
                val = self.high_key_max_value if is_key else self.high_value_max_value
                max_vals = torch.full((self.num_heads,), val, dtype=torch.float32, device=kv.device)

        # 调用已编译好的 TileLang 融合内核！
        # kv_flat + h_batched 是输入，q_kv_flat + scales 是输出，max_vals 是量化参数表
        if self.quant_type == "absmax":
            self.fused_kernel(kv_flat, h_batched, q_kv_flat, scales, max_vals)
            # 【修复阻塞】metadata 留在 GPU，避免 64 次同步阻塞摧毁异步流水线
            metadata = {
                "scales": scales.flatten(),
                "max_vals": max_vals,
            }
        elif self.quant_type == "minmax":
            mins = torch.empty((rows, num_heads), dtype=torch.float32, device=kv.device)
            self.fused_kernel(kv_flat, h_batched, q_kv_flat, scales, mins, max_vals)
            # 【修复阻塞】metadata 留在 GPU，避免同步阻塞
            metadata = {
                "scales": scales.flatten(),
                "mins": mins.flatten(),
                "max_vals": max_vals,
            }
        else:
            raise ValueError(f"Unsupported quant_type: {self.quant_type}")

        # 重新还原形状（将展开的 rows 恢复成 [blocks, block_size, ...]）
        q_kv = q_kv_flat.reshape(num_blocks, block_size, num_heads, head_dim)
        return q_kv, metadata

    # ===========================================================================
    # ██████████████████████████████████████████████████████████████████████████
    #  compress_v3：终极融合方案
    #  ─────────────────────────────────────────────────────────────────────────
    #  Channel 轴 (Key)：2 个 TileLang 内核 + 1 次微型 PyTorch 归约
    #    Kernel A: Hadamard + 局部 Channel Min/Max（融合，数据留在显存）
    #    Python:   torch.amin/amax 汇总局部极值 → 全局极值 (~10μs)
    #    Kernel B: 读取数据 + 全局极值 → 量化输出 uint8
    #    对比原版 PyTorch 的 7+ 次内核启动，减少到 2+1 次！
    #
    #  Token 轴 (Value)：1 个 TileLang 内核（完全融合）
    #    和现有的 fused_kernel 逻辑相同，Hadamard + Per-row 量化一次完成
    #
    
    # ===========================================================================
    # FWHT 蝶形融合内核（精度对齐版，替代 gemm Hadamard）
    # ===========================================================================

    def _compile_fwht_channel_reduce_kernel(self):
        """
        TileLang FWHT 融合内核：蝶形 Hadamard + Channel Reduce
        
        与 gemm 版的区别：
          - 用 7 级蝶形加减（O(n log n)）替代矩阵乘法（O(n²)）
          - 全程 float32 精度计算，精度 ≥ 原版 FWHT 库
          - 数据始终在 SRAM，不回显存 → 真正的融合算子
        """
        head_dim = self.head_dim
        log_dim = int(math.log2(head_dim))
        assert (1 << log_dim) == head_dim, f"head_dim must be power of 2, got {head_dim}"
        scale_val = 1.0 / math.sqrt(head_dim)

        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")
        num_tiles_var = tvm.tir.SizeVar("num_tiles_var", "int32")
        num_heads_var = tvm.tir.SizeVar("num_heads_var", "int32")

        @T.prim_func
        def fwht_channel_reduce(
            KV:       T.Buffer([rows_var, num_heads_var, head_dim], "bfloat16"),
            Signs:    T.Buffer([num_heads_var, head_dim], "bfloat16"),
            Had_Out:  T.Buffer([rows_var, num_heads_var, head_dim], "bfloat16"),
            Tile_Min: T.Buffer([num_tiles_var, num_heads_var, head_dim], "float32"),
            Tile_Max: T.Buffer([num_tiles_var, num_heads_var, head_dim], "float32"),
        ):
            with T.Kernel(num_tiles_var, num_heads_var, threads=128) as (bx, by):
                data = T.alloc_shared([32, head_dim], "float32")
                ch_min = T.alloc_fragment([head_dim], "float32")
                ch_max = T.alloc_fragment([head_dim], "float32")

                for j in T.Parallel(head_dim):
                    ch_min[j] = 1e10
                    ch_max[j] = -1e10

                # Load + 乘 Rademacher signs → float32
                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        data[i, j] = T.cast(KV[row_idx, by, j], "float32") * T.cast(Signs[by, j], "float32")
                    else:
                        data[i, j] = 0.0

                # FWHT 蝶形：7 级，全部在 SRAM 中完成
                for s in range(log_dim):
                    half = 1 << s
                    for i, j in T.Parallel(32, head_dim // 2):
                        group = j // half
                        offset = j % half
                        idx0 = group * (2 * half) + offset
                        idx1 = idx0 + half
                        a = data[i, idx0]
                        b = data[i, idx1]
                        data[i, idx0] = a + b
                        data[i, idx1] = a - b

                # Scale + Channel Reduce
                for i, j in T.Parallel(32, head_dim):
                    data[i, j] = data[i, j] * T.float32(scale_val)
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        ch_min[j] = T.min(ch_min[j], data[i, j])
                        ch_max[j] = T.max(ch_max[j], data[i, j])

                # 写出 Hadamard 结果 + 局部极值
                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        Had_Out[row_idx, by, j] = T.cast(data[i, j], "bfloat16")

                for j in T.Parallel(head_dim):
                    Tile_Min[bx, by, j] = ch_min[j]
                    Tile_Max[bx, by, j] = ch_max[j]

        return tilelang.compile(fwht_channel_reduce, target="cuda")

    def _compile_fwht_token_reduce_kernel(self):
        """
        TileLang FWHT 融合内核：蝶形 Hadamard + Token Reduce
        """
        head_dim = self.head_dim
        log_dim = int(math.log2(head_dim))
        scale_val = 1.0 / math.sqrt(head_dim)

        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")
        num_heads_var = tvm.tir.SizeVar("num_heads_var", "int32")

        @T.prim_func
        def fwht_token_reduce(
            KV:      T.Buffer([rows_var, num_heads_var, head_dim], "bfloat16"),
            Signs:   T.Buffer([num_heads_var, head_dim], "bfloat16"),
            Had_Out: T.Buffer([rows_var, num_heads_var, head_dim], "bfloat16"),
            Row_Min: T.Buffer([rows_var, num_heads_var], "float32"),
            Row_Max: T.Buffer([rows_var, num_heads_var], "float32"),
        ):
            with T.Kernel(T.ceildiv(rows_var, 32), num_heads_var, threads=128) as (bx, by):
                data = T.alloc_shared([32, head_dim], "float32")
                max_shared = T.alloc_shared([32], "float32")
                min_shared = T.alloc_shared([32], "float32")

                # Load + signs
                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        data[i, j] = T.cast(KV[row_idx, by, j], "float32") * T.cast(Signs[by, j], "float32")
                    else:
                        data[i, j] = 0.0

                # FWHT 蝶形
                for s in range(log_dim):
                    half = 1 << s
                    for i, j in T.Parallel(32, head_dim // 2):
                        group = j // half
                        offset = j % half
                        idx0 = group * (2 * half) + offset
                        idx1 = idx0 + half
                        a = data[i, idx0]
                        b = data[i, idx1]
                        data[i, idx0] = a + b
                        data[i, idx1] = a - b

                # Scale
                for i, j in T.Parallel(32, head_dim):
                    data[i, j] = data[i, j] * T.float32(scale_val)

                # 写出 + Token Reduce（沿 head_dim 方向求每行 min/max）
                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        Had_Out[row_idx, by, j] = T.cast(data[i, j], "bfloat16")

                T.reduce_max(data, max_shared, dim=-1, clear=True)
                T.reduce_min(data, min_shared, dim=-1, clear=True)

                for i in T.Parallel(32):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        Row_Min[row_idx, by] = min_shared[i]
                        Row_Max[row_idx, by] = max_shared[i]

        return tilelang.compile(fwht_token_reduce, target="cuda")

    # ===========================================================================
    # [vec_shfl] 向量化 FWHT reduce:register/warp-shuffle 蝶形(零 SMEM 蝶形)+
    # 向量化 global I/O。FWHT 数值与上面 f32 版 bit-exact(已验证);channel 版用
    # 小 SMEM 做跨行(per-column)归约。bf16 输入/输出,签名与 f32 版一致,可直接替换。
    # ===========================================================================
    def _compile_vec_token_reduce_kernel(self, bm: int = 4):
        head_dim = self.head_dim; WARP = 32; per_lane = head_dim // WARP
        log_dim = int(math.log2(head_dim)); scale_val = 1.0 / math.sqrt(head_dim)
        log_warp = int(math.log2(WARP)); threads = bm * WARP
        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")
        nh_var = tvm.tir.SizeVar("nh_var", "int32")

        @T.prim_func
        def vec_tok_reduce(
            KV: T.Buffer([rows_var, nh_var, head_dim], "bfloat16"),
            Signs: T.Buffer([nh_var, head_dim], "bfloat16"),
            Had: T.Buffer([rows_var, nh_var, head_dim], "bfloat16"),
            Row_Min: T.Buffer([rows_var, nh_var], "float32"),
            Row_Max: T.Buffer([rows_var, nh_var], "float32"),
        ):
            with T.Kernel(T.ceildiv(rows_var, bm), nh_var, threads=threads) as (bx, by):
                tid = T.get_thread_binding(); lane = tid % WARP; warp = tid // WARP
                reg = T.alloc_local([per_lane], "float32")
                lmax = T.alloc_local([1], "float32"); lmin = T.alloc_local([1], "float32")
                row = bx * bm + warp
                for c in T.vectorized(per_lane):
                    ch = lane * per_lane + c
                    reg[c] = T.if_then_else(row < rows_var,
                        T.cast(KV[row, by, ch], "float32") * T.cast(Signs[by, ch], "float32"), T.float32(0.0))
                for s in range(log_dim):
                    half = 1 << s
                    if half < per_lane:
                        for c in range(per_lane):
                            if (c % (2 * half)) < half:
                                a = reg[c]; bb = reg[c + half]; reg[c] = a + bb; reg[c + half] = a - bb
                    else:
                        lm = half // per_lane
                        for c in range(per_lane):
                            p = T.shfl_xor(reg[c], lm)
                            reg[c] = T.if_then_else((lane & lm) == 0, reg[c] + p, p - reg[c])
                for c in range(per_lane):
                    reg[c] = reg[c] * T.float32(scale_val)
                lmax[0] = reg[0]; lmin[0] = reg[0]
                for c in range(per_lane):
                    lmax[0] = T.max(lmax[0], reg[c]); lmin[0] = T.min(lmin[0], reg[c])
                for off in range(log_warp - 1, -1, -1):
                    st = 1 << off
                    lmax[0] = T.max(lmax[0], T.shfl_xor(lmax[0], st)); lmin[0] = T.min(lmin[0], T.shfl_xor(lmin[0], st))
                for c in T.vectorized(per_lane):
                    if row < rows_var:
                        Had[row, by, lane * per_lane + c] = T.cast(reg[c], "bfloat16")
                if row < rows_var and lane == 0:
                    Row_Min[row, by] = lmin[0]; Row_Max[row, by] = lmax[0]
        return tilelang.compile(vec_tok_reduce, target="cuda")

    def _compile_vec_channel_reduce_kernel(self, bm: int = 16):  # bm=16: overlap 甜点(warp 槽友好);solo 上 bm=32 略快
        head_dim = self.head_dim; WARP = 32; per_lane = head_dim // WARP
        log_dim = int(math.log2(head_dim)); scale_val = 1.0 / math.sqrt(head_dim)
        threads = bm * WARP
        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")
        nt_var = tvm.tir.SizeVar("nt_var", "int32")
        nh_var = tvm.tir.SizeVar("nh_var", "int32")

        @T.prim_func
        def vec_ch_reduce(
            KV: T.Buffer([rows_var, nh_var, head_dim], "bfloat16"),
            Signs: T.Buffer([nh_var, head_dim], "bfloat16"),
            Had: T.Buffer([rows_var, nh_var, head_dim], "bfloat16"),
            Tile_Min: T.Buffer([nt_var, nh_var, head_dim], "float32"),
            Tile_Max: T.Buffer([nt_var, nh_var, head_dim], "float32"),
        ):
            with T.Kernel(T.ceildiv(rows_var, bm), nh_var, threads=threads) as (bx, by):
                tid = T.get_thread_binding(); lane = tid % WARP; warp = tid // WARP
                reg = T.alloc_local([per_lane], "float32")
                mn1 = T.alloc_local([1], "float32"); mx1 = T.alloc_local([1], "float32")
                smem = T.alloc_shared([bm, head_dim], "float32")
                row = bx * bm + warp
                for c in T.vectorized(per_lane):
                    ch = lane * per_lane + c
                    reg[c] = T.if_then_else(row < rows_var,
                        T.cast(KV[row, by, ch], "float32") * T.cast(Signs[by, ch], "float32"), T.float32(0.0))
                for s in range(log_dim):
                    half = 1 << s
                    if half < per_lane:
                        for c in range(per_lane):
                            if (c % (2 * half)) < half:
                                a = reg[c]; bb = reg[c + half]; reg[c] = a + bb; reg[c + half] = a - bb
                    else:
                        lm = half // per_lane
                        for c in range(per_lane):
                            p = T.shfl_xor(reg[c], lm)
                            reg[c] = T.if_then_else((lane & lm) == 0, reg[c] + p, p - reg[c])
                for c in range(per_lane):
                    reg[c] = reg[c] * T.float32(scale_val)
                for c in T.vectorized(per_lane):
                    if row < rows_var:
                        Had[row, by, lane * per_lane + c] = T.cast(reg[c], "bfloat16")
                for c in range(per_lane):
                    smem[warp, lane * per_lane + c] = reg[c]
                T.sync_threads()
                if tid < head_dim:
                    mn1[0] = smem[0, tid]; mx1[0] = smem[0, tid]
                    for w in range(bm):
                        valid = (bx * bm + w) < rows_var
                        v = smem[w, tid]
                        mn1[0] = T.if_then_else(valid, T.min(mn1[0], v), mn1[0])
                        mx1[0] = T.if_then_else(valid, T.max(mx1[0], v), mx1[0])
                    Tile_Min[bx, by, tid] = mn1[0]; Tile_Max[bx, by, tid] = mx1[0]
                T.sync_threads()
        return tilelang.compile(vec_ch_reduce, target="cuda")

    # ===========================================================================
    # [throttle] Persistent grid-stride variants of the vec_shfl reduce kernels.
    # Launch only `gx` blocks (x-dim) that grid-stride over row-tiles → caps the
    # number of concurrently-active SMs → lowers HBM/SM demand so the transform
    # fits in the prefill GEMM's leftover (co-residency without contention).
    # Same I/O as the non-persistent vec kernels, so callers just swap which
    # compiled kernel they use.
    # ===========================================================================
    def _compile_vec_token_reduce_persistent(self, gx: int, bm: int = 4):
        head_dim = self.head_dim; WARP = 32; per_lane = head_dim // WARP
        log_dim = int(math.log2(head_dim)); scale_val = 1.0 / math.sqrt(head_dim)
        log_warp = int(math.log2(WARP)); threads = bm * WARP
        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")
        nh_var = tvm.tir.SizeVar("nh_var", "int32")

        @T.prim_func
        def vec_tok_reduce_p(
            KV: T.Buffer([rows_var, nh_var, head_dim], "bfloat16"),
            Signs: T.Buffer([nh_var, head_dim], "bfloat16"),
            Had: T.Buffer([rows_var, nh_var, head_dim], "bfloat16"),
            Row_Min: T.Buffer([rows_var, nh_var], "float32"),
            Row_Max: T.Buffer([rows_var, nh_var], "float32"),
        ):
            with T.Kernel(gx, nh_var, threads=threads) as (bx, by):
                tid = T.get_thread_binding(); lane = tid % WARP; warp = tid // WARP
                reg = T.alloc_local([per_lane], "float32")
                lmax = T.alloc_local([1], "float32"); lmin = T.alloc_local([1], "float32")
                ntiles = T.ceildiv(rows_var, bm)
                for t in T.serial(T.ceildiv(ntiles, gx)):
                    tile = bx + t * gx
                    row = tile * bm + warp
                    for c in T.vectorized(per_lane):
                        ch = lane * per_lane + c
                        reg[c] = T.if_then_else(row < rows_var,
                            T.cast(KV[row, by, ch], "float32") * T.cast(Signs[by, ch], "float32"), T.float32(0.0))
                    for s in range(log_dim):
                        half = 1 << s
                        if half < per_lane:
                            for c in range(per_lane):
                                if (c % (2 * half)) < half:
                                    a = reg[c]; bb = reg[c + half]; reg[c] = a + bb; reg[c + half] = a - bb
                        else:
                            lm = half // per_lane
                            for c in range(per_lane):
                                p = T.shfl_xor(reg[c], lm)
                                reg[c] = T.if_then_else((lane & lm) == 0, reg[c] + p, p - reg[c])
                    for c in range(per_lane):
                        reg[c] = reg[c] * T.float32(scale_val)
                    lmax[0] = reg[0]; lmin[0] = reg[0]
                    for c in range(per_lane):
                        lmax[0] = T.max(lmax[0], reg[c]); lmin[0] = T.min(lmin[0], reg[c])
                    for off in range(log_warp - 1, -1, -1):
                        st = 1 << off
                        lmax[0] = T.max(lmax[0], T.shfl_xor(lmax[0], st)); lmin[0] = T.min(lmin[0], T.shfl_xor(lmin[0], st))
                    for c in T.vectorized(per_lane):
                        if row < rows_var:
                            Had[row, by, lane * per_lane + c] = T.cast(reg[c], "bfloat16")
                    if row < rows_var and lane == 0:
                        Row_Min[row, by] = lmin[0]; Row_Max[row, by] = lmax[0]
        return tilelang.compile(vec_tok_reduce_p, target="cuda")

    def _compile_vec_channel_reduce_persistent(self, gx: int, bm: int = 16):
        head_dim = self.head_dim; WARP = 32; per_lane = head_dim // WARP
        log_dim = int(math.log2(head_dim)); scale_val = 1.0 / math.sqrt(head_dim)
        threads = bm * WARP
        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")
        nt_var = tvm.tir.SizeVar("nt_var", "int32")
        nh_var = tvm.tir.SizeVar("nh_var", "int32")

        @T.prim_func
        def vec_ch_reduce_p(
            KV: T.Buffer([rows_var, nh_var, head_dim], "bfloat16"),
            Signs: T.Buffer([nh_var, head_dim], "bfloat16"),
            Had: T.Buffer([rows_var, nh_var, head_dim], "bfloat16"),
            Tile_Min: T.Buffer([nt_var, nh_var, head_dim], "float32"),
            Tile_Max: T.Buffer([nt_var, nh_var, head_dim], "float32"),
        ):
            with T.Kernel(gx, nh_var, threads=threads) as (bx, by):
                tid = T.get_thread_binding(); lane = tid % WARP; warp = tid // WARP
                reg = T.alloc_local([per_lane], "float32")
                mn1 = T.alloc_local([1], "float32"); mx1 = T.alloc_local([1], "float32")
                smem = T.alloc_shared([bm, head_dim], "float32")
                ntiles = T.ceildiv(rows_var, bm)
                for t in T.serial(T.ceildiv(ntiles, gx)):
                    tile = bx + t * gx
                    row = tile * bm + warp
                    for c in T.vectorized(per_lane):
                        ch = lane * per_lane + c
                        reg[c] = T.if_then_else(row < rows_var,
                            T.cast(KV[row, by, ch], "float32") * T.cast(Signs[by, ch], "float32"), T.float32(0.0))
                    for s in range(log_dim):
                        half = 1 << s
                        if half < per_lane:
                            for c in range(per_lane):
                                if (c % (2 * half)) < half:
                                    a = reg[c]; bb = reg[c + half]; reg[c] = a + bb; reg[c + half] = a - bb
                        else:
                            lm = half // per_lane
                            for c in range(per_lane):
                                p = T.shfl_xor(reg[c], lm)
                                reg[c] = T.if_then_else((lane & lm) == 0, reg[c] + p, p - reg[c])
                    for c in range(per_lane):
                        reg[c] = reg[c] * T.float32(scale_val)
                    for c in T.vectorized(per_lane):
                        if row < rows_var:
                            Had[row, by, lane * per_lane + c] = T.cast(reg[c], "bfloat16")
                    for c in range(per_lane):
                        smem[warp, lane * per_lane + c] = reg[c]
                    T.sync_threads()
                    if tid < head_dim:
                        mn1[0] = smem[0, tid]; mx1[0] = smem[0, tid]
                        for w in range(bm):
                            valid = (tile * bm + w) < rows_var
                            v = smem[w, tid]
                            mn1[0] = T.if_then_else(valid, T.min(mn1[0], v), mn1[0])
                            mx1[0] = T.if_then_else(valid, T.max(mx1[0], v), mx1[0])
                        Tile_Min[tile, by, tid] = mn1[0]; Tile_Max[tile, by, tid] = mx1[0]
                    T.sync_threads()
        return tilelang.compile(vec_ch_reduce_p, target="cuda")

    # ===========================================================================
    # [co-design Edit C, v3 path] FWHT channel/token reduce kernels with the
    # in-SMEM butterfly buffer kept in float16 (½ SMEM vs float32 → lower
    # footprint → higher occupancy). Output Had_Out stays bfloat16, reductions
    # cast to float32, so the downstream quantize kernels + decompress_v3 are
    # UNCHANGED. Selected when fused_variant=="fwht_f16". The float32 versions
    # above are left byte-identical (configs 1/2 baseline untouched).
    # ===========================================================================
    def _compile_fwht_channel_reduce_kernel_f16(self):
        head_dim = self.head_dim
        log_dim = int(math.log2(head_dim))
        assert (1 << log_dim) == head_dim, f"head_dim must be power of 2, got {head_dim}"
        scale_val = 1.0 / math.sqrt(head_dim)

        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")
        num_tiles_var = tvm.tir.SizeVar("num_tiles_var", "int32")
        num_heads_var = tvm.tir.SizeVar("num_heads_var", "int32")

        @T.prim_func
        def fwht_channel_reduce_f16(
            KV:       T.Buffer([rows_var, num_heads_var, head_dim], "bfloat16"),
            Signs:    T.Buffer([num_heads_var, head_dim], "bfloat16"),
            Had_Out:  T.Buffer([rows_var, num_heads_var, head_dim], "bfloat16"),
            Tile_Min: T.Buffer([num_tiles_var, num_heads_var, head_dim], "float32"),
            Tile_Max: T.Buffer([num_tiles_var, num_heads_var, head_dim], "float32"),
        ):
            with T.Kernel(num_tiles_var, num_heads_var, threads=128) as (bx, by):
                data = T.alloc_shared([32, head_dim], "float16")     # ½ SMEM vs f32
                ch_min = T.alloc_fragment([head_dim], "float32")
                ch_max = T.alloc_fragment([head_dim], "float32")

                for j in T.Parallel(head_dim):
                    ch_min[j] = 1e10
                    ch_max[j] = -1e10

                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        data[i, j] = T.cast(T.cast(KV[row_idx, by, j], "float32") * T.cast(Signs[by, j], "float32"), "float16")
                    else:
                        data[i, j] = T.float16(0.0)

                for s in range(log_dim):
                    half = 1 << s
                    for i, j in T.Parallel(32, head_dim // 2):
                        group = j // half
                        offset = j % half
                        idx0 = group * (2 * half) + offset
                        idx1 = idx0 + half
                        a = data[i, idx0]
                        b = data[i, idx1]
                        data[i, idx0] = a + b
                        data[i, idx1] = a - b

                for i, j in T.Parallel(32, head_dim):
                    data[i, j] = data[i, j] * T.float16(scale_val)
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        v = T.cast(data[i, j], "float32")
                        ch_min[j] = T.min(ch_min[j], v)
                        ch_max[j] = T.max(ch_max[j], v)

                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        Had_Out[row_idx, by, j] = T.cast(T.cast(data[i, j], "float32"), "bfloat16")

                for j in T.Parallel(head_dim):
                    Tile_Min[bx, by, j] = ch_min[j]
                    Tile_Max[bx, by, j] = ch_max[j]

        return tilelang.compile(fwht_channel_reduce_f16, target="cuda")

    def _compile_fwht_token_reduce_kernel_f16(self):
        head_dim = self.head_dim
        log_dim = int(math.log2(head_dim))
        scale_val = 1.0 / math.sqrt(head_dim)

        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")
        num_heads_var = tvm.tir.SizeVar("num_heads_var", "int32")

        @T.prim_func
        def fwht_token_reduce_f16(
            KV:      T.Buffer([rows_var, num_heads_var, head_dim], "bfloat16"),
            Signs:   T.Buffer([num_heads_var, head_dim], "bfloat16"),
            Had_Out: T.Buffer([rows_var, num_heads_var, head_dim], "bfloat16"),
            Row_Min: T.Buffer([rows_var, num_heads_var], "float32"),
            Row_Max: T.Buffer([rows_var, num_heads_var], "float32"),
        ):
            with T.Kernel(T.ceildiv(rows_var, 32), num_heads_var, threads=128) as (bx, by):
                data = T.alloc_shared([32, head_dim], "float16")     # ½ SMEM vs f32
                max_shared = T.alloc_shared([32], "float16")
                min_shared = T.alloc_shared([32], "float16")

                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        data[i, j] = T.cast(T.cast(KV[row_idx, by, j], "float32") * T.cast(Signs[by, j], "float32"), "float16")
                    else:
                        data[i, j] = T.float16(0.0)

                for s in range(log_dim):
                    half = 1 << s
                    for i, j in T.Parallel(32, head_dim // 2):
                        group = j // half
                        offset = j % half
                        idx0 = group * (2 * half) + offset
                        idx1 = idx0 + half
                        a = data[i, idx0]
                        b = data[i, idx1]
                        data[i, idx0] = a + b
                        data[i, idx1] = a - b

                for i, j in T.Parallel(32, head_dim):
                    data[i, j] = data[i, j] * T.float16(scale_val)

                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        Had_Out[row_idx, by, j] = T.cast(T.cast(data[i, j], "float32"), "bfloat16")

                T.reduce_max(data, max_shared, dim=-1, clear=True)
                T.reduce_min(data, min_shared, dim=-1, clear=True)

                for i in T.Parallel(32):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        Row_Min[row_idx, by] = T.cast(min_shared[i], "float32")
                        Row_Max[row_idx, by] = T.cast(max_shared[i], "float32")

        return tilelang.compile(fwht_token_reduce_f16, target="cuda")

    # ===========================================================================
    # gemm-based Hadamard 内核（旧版，精度较低）
    # ===========================================================================

    def _compile_hadamard_channel_reduce_kernel(self):
        """
        TileLang 融合内核 A：Hadamard 变换 + 每 Tile 局部 Channel Min/Max
        
        每个 Thread Block 处理 32 行 × 1 个 Head：
          1. 从显存读取 32 行数据到 SRAM
          2. 在 SRAM 中做 Hadamard 矩阵乘法（Tensor Core 加速）
          3. 在 SRAM 中对 128 个 Channel 各自求 32 行的局部 min/max
          4. 将 Hadamard 结果写回显存（fp16，供 Kernel B 读取）
          5. 将局部 min/max 写回显存（极小数据，几乎零开销）
        
        输出：
          Had_Out:   [rows, heads, dim] float16  — Hadamard 变换结果
          Tile_Min:  [num_tiles, heads, dim] float32 — 每 tile 每 channel 的局部最小值
          Tile_Max:  [num_tiles, heads, dim] float32 — 每 tile 每 channel 的局部最大值
        """
        head_dim = self.head_dim

        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")
        num_tiles_var = tvm.tir.SizeVar("num_tiles_var", "int32")
        num_heads_var = tvm.tir.SizeVar("num_heads_var", "int32")

        @T.prim_func
        def hadamard_channel_reduce(
            KV:       T.Buffer([rows_var, num_heads_var, head_dim], "float16"),
            H_Batch:  T.Buffer([num_heads_var, head_dim, head_dim], "float16"),
            Had_Out:  T.Buffer([rows_var, num_heads_var, head_dim], "float16"),
            Tile_Min: T.Buffer([num_tiles_var, num_heads_var, head_dim], "float32"),
            Tile_Max: T.Buffer([num_tiles_var, num_heads_var, head_dim], "float32"),
        ):
            with T.Kernel(num_tiles_var, num_heads_var, threads=128) as (bx, by):
                # ─── SRAM 分配 ───
                kv_shared  = T.alloc_shared([32, head_dim], "float16")
                h_shared   = T.alloc_shared([head_dim, head_dim], "float16")
                out_frag   = T.alloc_fragment([32, head_dim], "float32")
                out_shared = T.alloc_shared([32, head_dim], "float32")
                
                # 局部 channel 极值（寄存器级）
                ch_min = T.alloc_fragment([head_dim], "float32")
                ch_max = T.alloc_fragment([head_dim], "float32")

                T.clear(out_frag)
                
                # 初始化极值
                for j in T.Parallel(head_dim):
                    ch_min[j] = 1e10
                    ch_max[j] = -1e10

                # ─── Step 1: 搬入 SRAM ───
                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        kv_shared[i, j] = KV[row_idx, by, j]
                    else:
                        kv_shared[i, j] = 0.0

                for i, j in T.Parallel(head_dim, head_dim):
                    h_shared[i, j] = H_Batch[by, i, j]

                # ─── Step 2: Hadamard (Tensor Core GEMM) ───
                T.gemm(kv_shared, h_shared, out_frag)
                T.copy(out_frag, out_shared)

                # ─── Step 3: 局部 Channel Reduce ───
                # 对 32 行中的每个 channel (j) 求 min/max
                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        val = out_shared[i, j]
                        ch_min[j] = T.min(ch_min[j], val)
                        ch_max[j] = T.max(ch_max[j], val)

                # ─── Step 4: 写出 Hadamard 结果 + 局部极值 ───
                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        Had_Out[row_idx, by, j] = T.cast(out_shared[i, j], "float16")

                for j in T.Parallel(head_dim):
                    Tile_Min[bx, by, j] = ch_min[j]
                    Tile_Max[bx, by, j] = ch_max[j]

        return tilelang.compile(hadamard_channel_reduce, target="cuda")

    def _compile_channel_quantize_kernel(self):
        """
        TileLang 融合内核 B：用全局 Channel Min/Max 做量化
        
        每个 Thread Block 处理 32 行 × 1 个 Head：
          1. 从显存读取 Hadamard 结果（fp16）
          2. 从显存读取全局 min/max（极小数据，广播到所有线程）
          3. 计算 scale = (max - min) / (max_value - 1)
          4. 量化: Q = clamp(round((x - min) / scale), 0, max_value)
          5. 写出 uint8
        
        与原版 quantizer_func.quantize(axis="channel") 字节级一致！
        """
        head_dim = self.head_dim

        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")
        num_heads_var = tvm.tir.SizeVar("num_heads_var", "int32")

        @T.prim_func
        def channel_quantize_perhead(
            Had_In:     T.Buffer([rows_var, num_heads_var, head_dim], "bfloat16"),
            Global_Min: T.Buffer([num_heads_var, head_dim], "float32"),
            Global_Max: T.Buffer([num_heads_var, head_dim], "float32"),
            Max_Value:  T.Buffer([num_heads_var], "float32"),   # 每头量化上限
            Q_Out:      T.Buffer([rows_var, num_heads_var, head_dim], "uint8"),
        ):
            with T.Kernel(T.ceildiv(rows_var, 32), num_heads_var, threads=128) as (bx, by):
                # ─── SRAM 分配 ───
                data_shared = T.alloc_shared([32, head_dim], "bfloat16")
                g_min_frag  = T.alloc_fragment([head_dim], "float32")
                g_scale_frag = T.alloc_fragment([head_dim], "float32")

                # ─── Step 1: 加载全局极值到寄存器（每个 block 只读一次，128 个 float）───
                for j in T.Parallel(head_dim):
                    g_min = Global_Min[by, j]
                    g_max = Global_Max[by, j]
                    diff = T.max(g_max - g_min, 1e-5)
                    g_min_frag[j] = g_min
                    g_scale_frag[j] = diff / (Max_Value[by] - 1.0)

                # ─── Step 2: 加载 Hadamard 数据 ───
                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        data_shared[i, j] = Had_In[row_idx, by, j]
                    else:
                        data_shared[i, j] = 0.0

                # ─── Step 3: 量化（与原版公式完全一致）───
                # Q = clamp(round((x - min) / scale), 0, max_value)
                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        val = T.cast(data_shared[i, j], "float32")
                        normalized = (val - g_min_frag[j]) / g_scale_frag[j]
                        clamped = T.min(T.max(T.floor(normalized + 0.5), 0.0), Max_Value[by])
                        Q_Out[row_idx, by, j] = T.cast(clamped, "uint8")

        return tilelang.compile(channel_quantize_perhead, target="cuda")

    def _compile_channel_quantize_kernel_fast(self, bm: int = 64):
        """[overlap co-design] Leaner channel-quantize: a pure memory-streaming
        kernel that hides better behind a compute-bound GEMM (Exp A: the staged
        version hides only +20.5% vs copy's +44.6% because its SM/issue is 48%).

        Two cuts vs `_compile_channel_quantize_kernel`:
          1. NO shared-memory staging — quantize reads Had_In straight from global
             into registers (the staging write+read was pure LSU/issue overhead;
             quantize is elementwise, no cross-thread sharing needed).
          2. precompute inv_scale = 1/scale per channel → per-element MULTIPLY
             instead of DIVIDE.
        Goal: drop sm__throughput toward copy's ~38% → lift overlap toward ~40%.
        Output bit-identical to the staged version. bm rows/block (larger → better
        coalescing of the global stream).
        """
        head_dim = self.head_dim
        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")
        num_heads_var = tvm.tir.SizeVar("num_heads_var", "int32")

        @T.prim_func
        def channel_quantize_fast(
            Had_In:     T.Buffer([rows_var, num_heads_var, head_dim], "bfloat16"),
            Global_Min: T.Buffer([num_heads_var, head_dim], "float32"),
            Global_Max: T.Buffer([num_heads_var, head_dim], "float32"),
            Max_Value:  T.Buffer([num_heads_var], "float32"),
            Q_Out:      T.Buffer([rows_var, num_heads_var, head_dim], "uint8"),
        ):
            with T.Kernel(T.ceildiv(rows_var, bm), num_heads_var, threads=128) as (bx, by):
                g_min_frag = T.alloc_fragment([head_dim], "float32")
                g_inv_frag = T.alloc_fragment([head_dim], "float32")
                mv = T.alloc_fragment([1], "float32")

                mv[0] = Max_Value[by]
                for j in T.Parallel(head_dim):
                    gm = Global_Min[by, j]
                    diff = T.max(Global_Max[by, j] - gm, 1e-5)
                    scale = diff / (mv[0] - 1.0)
                    g_min_frag[j] = gm
                    g_inv_frag[j] = 1.0 / scale          # divide once per channel

                for i, j in T.Parallel(bm, head_dim):
                    row_idx = bx * bm + i
                    if row_idx < rows_var:
                        val = T.cast(Had_In[row_idx, by, j], "float32")
                        normalized = (val - g_min_frag[j]) * g_inv_frag[j]   # multiply
                        clamped = T.min(T.max(T.floor(normalized + 0.5), 0.0), mv[0])
                        Q_Out[row_idx, by, j] = T.cast(clamped, "uint8")

        return tilelang.compile(channel_quantize_fast, target="cuda")

    def _compile_channel_quantize_kernel_mulonly(self):
        """[overlap co-design, Exp 1] KEEP the SMEM staging (it aids coalescing),
        change ONLY the per-element divide→multiply (precompute inv_scale). Isolates
        whether divide→mul alone helps, vs the fast variant that also dropped SMEM."""
        head_dim = self.head_dim
        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")
        num_heads_var = tvm.tir.SizeVar("num_heads_var", "int32")

        @T.prim_func
        def channel_quantize_mulonly(
            Had_In:     T.Buffer([rows_var, num_heads_var, head_dim], "bfloat16"),
            Global_Min: T.Buffer([num_heads_var, head_dim], "float32"),
            Global_Max: T.Buffer([num_heads_var, head_dim], "float32"),
            Max_Value:  T.Buffer([num_heads_var], "float32"),
            Q_Out:      T.Buffer([rows_var, num_heads_var, head_dim], "uint8"),
        ):
            with T.Kernel(T.ceildiv(rows_var, 32), num_heads_var, threads=128) as (bx, by):
                data_shared = T.alloc_shared([32, head_dim], "bfloat16")
                g_min_frag = T.alloc_fragment([head_dim], "float32")
                g_inv_frag = T.alloc_fragment([head_dim], "float32")
                for j in T.Parallel(head_dim):
                    gm = Global_Min[by, j]
                    diff = T.max(Global_Max[by, j] - gm, 1e-5)
                    g_min_frag[j] = gm
                    g_inv_frag[j] = (Max_Value[by] - 1.0) / diff      # = 1/scale
                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        data_shared[i, j] = Had_In[row_idx, by, j]
                    else:
                        data_shared[i, j] = 0.0
                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        val = T.cast(data_shared[i, j], "float32")
                        normalized = (val - g_min_frag[j]) * g_inv_frag[j]   # multiply
                        clamped = T.min(T.max(T.floor(normalized + 0.5), 0.0), Max_Value[by])
                        Q_Out[row_idx, by, j] = T.cast(clamped, "uint8")

        return tilelang.compile(channel_quantize_mulonly, target="cuda")

    def _compile_channel_quantize_kernel_packed(self):
        """[overlap co-design, Exp 2] staged + divide→mul + PACK 4×uint8 → one
        uint32 store (cut sub-word store instructions 4×). Output is a uint32 buffer
        [rows, heads, head_dim//4]; little-endian byte order matches the uint8 layout."""
        head_dim = self.head_dim
        assert head_dim % 4 == 0
        hd4 = head_dim // 4
        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")
        num_heads_var = tvm.tir.SizeVar("num_heads_var", "int32")

        @T.prim_func
        def channel_quantize_packed(
            Had_In:     T.Buffer([rows_var, num_heads_var, head_dim], "bfloat16"),
            Global_Min: T.Buffer([num_heads_var, head_dim], "float32"),
            Global_Max: T.Buffer([num_heads_var, head_dim], "float32"),
            Max_Value:  T.Buffer([num_heads_var], "float32"),
            Q_Out32:    T.Buffer([rows_var, num_heads_var, hd4], "uint32"),
        ):
            with T.Kernel(T.ceildiv(rows_var, 32), num_heads_var, threads=128) as (bx, by):
                data_shared = T.alloc_shared([32, head_dim], "bfloat16")
                g_min_sh = T.alloc_shared([head_dim], "float32")   # shared → any lane reads any j
                g_inv_sh = T.alloc_shared([head_dim], "float32")
                for j in T.Parallel(head_dim):
                    gm = Global_Min[by, j]
                    diff = T.max(Global_Max[by, j] - gm, 1e-5)
                    g_min_sh[j] = gm
                    g_inv_sh[j] = (Max_Value[by] - 1.0) / diff
                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        data_shared[i, j] = Had_In[row_idx, by, j]
                    else:
                        data_shared[i, j] = 0.0
                mvv = Max_Value[by]
                for i, j4 in T.Parallel(32, hd4):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        b = j4 * 4
                        n0 = (T.cast(data_shared[i, b + 0], "float32") - g_min_sh[b + 0]) * g_inv_sh[b + 0]
                        n1 = (T.cast(data_shared[i, b + 1], "float32") - g_min_sh[b + 1]) * g_inv_sh[b + 1]
                        n2 = (T.cast(data_shared[i, b + 2], "float32") - g_min_sh[b + 2]) * g_inv_sh[b + 2]
                        n3 = (T.cast(data_shared[i, b + 3], "float32") - g_min_sh[b + 3]) * g_inv_sh[b + 3]
                        c0 = T.cast(T.min(T.max(T.floor(n0 + 0.5), 0.0), mvv), "uint32")
                        c1 = T.cast(T.min(T.max(T.floor(n1 + 0.5), 0.0), mvv), "uint32")
                        c2 = T.cast(T.min(T.max(T.floor(n2 + 0.5), 0.0), mvv), "uint32")
                        c3 = T.cast(T.min(T.max(T.floor(n3 + 0.5), 0.0), mvv), "uint32")
                        Q_Out32[row_idx, by, j4] = (
                            c0 | T.shift_left(c1, T.uint32(8))
                            | T.shift_left(c2, T.uint32(16))
                            | T.shift_left(c3, T.uint32(24))
                        )

        return tilelang.compile(channel_quantize_packed, target="cuda")

    def _compile_token_quantize_kernel(self):
        """
        TileLang 融合内核：Hadamard + Token 轴量化（完全融合，单次内核）
        
        Token 轴 = 每行独立找 min/max，完美适配 Tile 并行！
        与原版 quantizer_func.quantize(axis="token") 对齐！
        
        原版公式：
          reduce dims = [2, 3] (num_heads, head_dim)
          → 每个 (block, token) 位置只有 1 个 scale，所有 head 共享
          scale = (max - min) / (max_value - 1)
          Q = clamp(round((x - min) / scale), 0, max_value)
        
        因此 Token 轴也需要 2 阶段：
          Kernel A: Hadamard + 每行每头的 min/max → [rows, heads]
          Python:   跨 head 归约 → 每行 1 个全局 min/max → [rows]
          Kernel B: 用全局 per-row min/max 量化
        """
        head_dim = self.head_dim

        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")
        num_heads_var = tvm.tir.SizeVar("num_heads_var", "int32")

        @T.prim_func
        def hadamard_token_reduce(
            KV:       T.Buffer([rows_var, num_heads_var, head_dim], "float16"),
            H_Batch:  T.Buffer([num_heads_var, head_dim, head_dim], "float16"),
            Had_Out:  T.Buffer([rows_var, num_heads_var, head_dim], "float16"),
            Row_Min:  T.Buffer([rows_var, num_heads_var], "float32"),
            Row_Max:  T.Buffer([rows_var, num_heads_var], "float32"),
        ):
            with T.Kernel(T.ceildiv(rows_var, 32), num_heads_var, threads=128) as (bx, by):
                kv_shared  = T.alloc_shared([32, head_dim], "float16")
                h_shared   = T.alloc_shared([head_dim, head_dim], "float16")
                out_frag   = T.alloc_fragment([32, head_dim], "float32")
                out_shared = T.alloc_shared([32, head_dim], "float32")
                max_shared = T.alloc_shared([32], "float32")
                min_shared = T.alloc_shared([32], "float32")

                T.clear(out_frag)

                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        kv_shared[i, j] = KV[row_idx, by, j]
                    else:
                        kv_shared[i, j] = 0.0

                for i, j in T.Parallel(head_dim, head_dim):
                    h_shared[i, j] = H_Batch[by, i, j]

                # Hadamard (Tensor Core)
                T.gemm(kv_shared, h_shared, out_frag)
                T.copy(out_frag, out_shared)

                # 每行在当前 head 内求 min/max（沿 head_dim 方向归约）
                T.reduce_max(out_shared, max_shared, dim=-1, clear=True)
                T.reduce_min(out_shared, min_shared, dim=-1, clear=True)

                # 写出 Hadamard 结果 + 每行每头的局部极值
                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        Had_Out[row_idx, by, j] = T.cast(out_shared[i, j], "float16")

                for i in T.Parallel(32):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        Row_Min[row_idx, by] = min_shared[i]
                        Row_Max[row_idx, by] = max_shared[i]

        return tilelang.compile(hadamard_token_reduce, target="cuda")

    def _compile_token_quantize_global_kernel(self):
        """
        TileLang 内核 B（Token 轴）：用全局 per-row min/max 做量化
        
        每个 (token, head) 共享同一个 scale（与原版一致）。
        Global_Min/Max 是 [rows] 维度（跨所有 head 归约后的结果）。
        """
        head_dim = self.head_dim

        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")
        num_heads_var = tvm.tir.SizeVar("num_heads_var", "int32")

        @T.prim_func
        def token_quantize_global_perhead(
            Had_In:     T.Buffer([rows_var, num_heads_var, head_dim], "float16"),
            Global_Min: T.Buffer([rows_var], "float32"),
            Global_Max: T.Buffer([rows_var], "float32"),
            Max_Value:  T.Buffer([num_heads_var], "float32"),  # 每头量化上限
            Q_Out:      T.Buffer([rows_var, num_heads_var, head_dim], "uint8"),
        ):
            with T.Kernel(T.ceildiv(rows_var, 32), num_heads_var, threads=128) as (bx, by):
                data_shared  = T.alloc_shared([32, head_dim], "float16")
                g_min_shared = T.alloc_shared([32], "float32")
                g_scale_shared = T.alloc_shared([32], "float32")

                # 加载全局 per-row min/max → 计算 scale
                for i in T.Parallel(32):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        g_min = Global_Min[row_idx]
                        g_max = Global_Max[row_idx]
                        diff = T.max(g_max - g_min, 1e-5)
                        g_min_shared[i] = g_min
                        g_scale_shared[i] = diff / (Max_Value[by] - 1.0)

                # 加载 Hadamard 数据
                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        data_shared[i, j] = Had_In[row_idx, by, j]
                    else:
                        data_shared[i, j] = 0.0

                # 量化（与原版公式一致，每行所有 head 共享同一个 scale）
                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        val = T.cast(data_shared[i, j], "float32")
                        normalized = (val - g_min_shared[i]) / g_scale_shared[i]
                        clamped = T.min(T.max(T.floor(normalized + 0.5), 0.0), Max_Value[by])
                        Q_Out[row_idx, by, j] = T.cast(clamped, "uint8")

        return tilelang.compile(token_quantize_global_perhead, target="cuda")

    def _compile_token_quantize_perhead_v2_kernel(self):
        """
        Token 轴量化 v2：per-row per-head min/max（无需 head-split）
        
        Row_Min/Max: [rows, heads] — 每行每头独立的 min/max
        Max_Value:   [heads]       — per-head 量化上限
        精度 ≥ 原版（更细粒度的量化参数）
        """
        head_dim = self.head_dim

        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")
        num_heads_var = tvm.tir.SizeVar("num_heads_var", "int32")

        @T.prim_func
        def token_quantize_perhead_v2(
            Had_In:    T.Buffer([rows_var, num_heads_var, head_dim], "bfloat16"),
            Row_Min:   T.Buffer([rows_var, num_heads_var], "float32"),
            Row_Max:   T.Buffer([rows_var, num_heads_var], "float32"),
            Max_Value: T.Buffer([num_heads_var], "float32"),
            Q_Out:     T.Buffer([rows_var, num_heads_var, head_dim], "uint8"),
        ):
            with T.Kernel(T.ceildiv(rows_var, 32), num_heads_var, threads=128) as (bx, by):
                data_shared    = T.alloc_shared([32, head_dim], "bfloat16")
                g_min_shared   = T.alloc_shared([32], "float32")
                g_scale_shared = T.alloc_shared([32], "float32")

                for i in T.Parallel(32):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        g_min = Row_Min[row_idx, by]
                        g_max = Row_Max[row_idx, by]
                        diff = T.max(g_max - g_min, 1e-5)
                        g_min_shared[i] = g_min
                        g_scale_shared[i] = diff / (Max_Value[by] - 1.0)

                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        data_shared[i, j] = Had_In[row_idx, by, j]
                    else:
                        data_shared[i, j] = 0.0

                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        val = T.cast(data_shared[i, j], "float32")
                        normalized = (val - g_min_shared[i]) / g_scale_shared[i]
                        clamped = T.min(T.max(T.floor(normalized + 0.5), 0.0), Max_Value[by])
                        Q_Out[row_idx, by, j] = T.cast(clamped, "uint8")

        return tilelang.compile(token_quantize_perhead_v2, target="cuda")

    def _compile_token_quantize_grouped_kernel(self):
        """
        TileLang 融合分组量化内核：在 GPU 内核内部完成跨 Head 的分组归约 + 量化。
        
        相比 token_quantize_perhead_v2 的改进：
          - 接受 GroupMap [num_heads] 指定分组
          - 每个线程在寄存器内对 8 个 Head 的 Row_Min/Row_Max 做跨组 min/max
          - 输出分组后的 G_Min/G_Scale 供解压缩使用
          - 消除了 Python 层面的 scatter_reduce + gather（省 ~1.2ms / 32层）
        """
        head_dim = self.head_dim
        num_heads_const = self.num_heads  # 编译期常量（如 8），用于循环展开

        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")
        num_heads_var = tvm.tir.SizeVar("num_heads_var", "int32")

        @T.prim_func
        def token_quantize_grouped(
            Had_In:      T.Buffer([rows_var, num_heads_var, head_dim], "bfloat16"),
            Row_Min:     T.Buffer([rows_var, num_heads_var], "float32"),
            Row_Max:     T.Buffer([rows_var, num_heads_var], "float32"),
            GroupMap:     T.Buffer([num_heads_var], "int32"),
            Max_Value:   T.Buffer([num_heads_var], "float32"),
            Q_Out:       T.Buffer([rows_var, num_heads_var, head_dim], "uint8"),
            G_Min_Out:   T.Buffer([rows_var, num_heads_var], "float32"),
            G_Scale_Out: T.Buffer([rows_var, num_heads_var], "float32"),
        ):
            with T.Kernel(T.ceildiv(rows_var, 32), num_heads_var, threads=128) as (bx, by):
                data_shared    = T.alloc_shared([32, head_dim], "bfloat16")
                g_min_shared   = T.alloc_shared([32], "float32")
                g_max_tmp      = T.alloc_shared([32], "float32")
                g_scale_shared = T.alloc_shared([32], "float32")

                # ─── 跨 Head 分组归约（用 shared memory 做累积器） ───
                for i in T.Parallel(32):
                    g_min_shared[i] = 1e10
                    g_max_tmp[i] = -1e10

                for i in T.Parallel(32):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        for h in range(num_heads_const):
                            if GroupMap[h] == GroupMap[by]:
                                g_min_shared[i] = T.min(g_min_shared[i], Row_Min[row_idx, h])
                                g_max_tmp[i] = T.max(g_max_tmp[i], Row_Max[row_idx, h])

                for i in T.Parallel(32):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        diff = T.max(g_max_tmp[i] - g_min_shared[i], 1e-5)
                        g_scale_shared[i] = diff / (Max_Value[by] - 1.0)
                        G_Min_Out[row_idx, by] = g_min_shared[i]
                        G_Scale_Out[row_idx, by] = g_scale_shared[i]

                # ─── 加载数据到共享内存 ───
                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        data_shared[i, j] = Had_In[row_idx, by, j]
                    else:
                        data_shared[i, j] = 0.0

                # ─── 量化 ───
                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        val = T.cast(data_shared[i, j], "float32")
                        normalized = (val - g_min_shared[i]) / g_scale_shared[i]
                        clamped = T.min(T.max(T.floor(normalized + 0.5), 0.0), Max_Value[by])
                        Q_Out[row_idx, by, j] = T.cast(clamped, "uint8")

        return tilelang.compile(token_quantize_grouped, target="cuda")

    def _compile_token_quantize_grouped_kernel_light(self):
        """[H100 co-design] Light-footprint token grouped quantize — bit-identical
        to `_compile_token_quantize_grouped_kernel` but WITHOUT the 8 KB `data_shared`
        staging (read Had_In straight from global), so the block's SMEM drops 8.2 KB
        → ~0.4 KB and it can co-reside in the persistent nvjet GEMM's per-SM slack
        (which an 8 KB-SMEM block cannot). Math/metadata UNCHANGED (same grouped
        min/max, same divide, same round/clamp) → maxdiff(Q_Out)=0 vs the original.
        """
        head_dim = self.head_dim
        num_heads_const = self.num_heads
        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")
        num_heads_var = tvm.tir.SizeVar("num_heads_var", "int32")

        @T.prim_func
        def token_quantize_grouped_light(
            Had_In:      T.Buffer([rows_var, num_heads_var, head_dim], "bfloat16"),
            Row_Min:     T.Buffer([rows_var, num_heads_var], "float32"),
            Row_Max:     T.Buffer([rows_var, num_heads_var], "float32"),
            GroupMap:    T.Buffer([num_heads_var], "int32"),
            Max_Value:   T.Buffer([num_heads_var], "float32"),
            Q_Out:       T.Buffer([rows_var, num_heads_var, head_dim], "uint8"),
            G_Min_Out:   T.Buffer([rows_var, num_heads_var], "float32"),
            G_Scale_Out: T.Buffer([rows_var, num_heads_var], "float32"),
        ):
            with T.Kernel(T.ceildiv(rows_var, 32), num_heads_var, threads=128) as (bx, by):
                # tiny per-row accumulators only (no 8KB data_shared staging)
                g_min_shared   = T.alloc_shared([32], "float32")
                g_max_tmp      = T.alloc_shared([32], "float32")
                g_scale_shared = T.alloc_shared([32], "float32")

                for i in T.Parallel(32):
                    g_min_shared[i] = 1e10
                    g_max_tmp[i] = -1e10

                for i in T.Parallel(32):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        for h in range(num_heads_const):
                            if GroupMap[h] == GroupMap[by]:
                                g_min_shared[i] = T.min(g_min_shared[i], Row_Min[row_idx, h])
                                g_max_tmp[i] = T.max(g_max_tmp[i], Row_Max[row_idx, h])

                for i in T.Parallel(32):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        diff = T.max(g_max_tmp[i] - g_min_shared[i], 1e-5)
                        g_scale_shared[i] = diff / (Max_Value[by] - 1.0)
                        G_Min_Out[row_idx, by] = g_min_shared[i]
                        G_Scale_Out[row_idx, by] = g_scale_shared[i]

                # quantize reading Had_In DIRECTLY (no SMEM staging) — identical math
                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        val = T.cast(Had_In[row_idx, by, j], "float32")
                        normalized = (val - g_min_shared[i]) / g_scale_shared[i]
                        clamped = T.min(T.max(T.floor(normalized + 0.5), 0.0), Max_Value[by])
                        Q_Out[row_idx, by, j] = T.cast(clamped, "uint8")

        return tilelang.compile(token_quantize_grouped_light, target="cuda")

    def _compile_token_quantize_grouped_kernel_warp(self, warps: int = 4):
        """[H100 co-design v2] warp-per-row, register-only token grouped quantize.
        One row → one warp (32 lanes × head_dim/32 elems). Grouped min/max done in
        REGISTERS (each lane redundantly scans the ≤num_heads group entries of
        Row_Min/Row_Max — tiny, L2-broadcast — no `g_*` shared, no cross-thread
        reduce, no 2D staging). Strips registers + SMEM to the floor → fits nvjet's
        per-SM slack. Math/metadata IDENTICAL to the staged kernel (same grouped
        min/max, same divide, same round/clamp) → maxdiff(Q_Out)=0.
        """
        head_dim = self.head_dim
        num_heads_const = self.num_heads
        WARP = 32
        per_lane = head_dim // WARP
        assert per_lane * WARP == head_dim
        threads = warps * WARP
        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")
        num_heads_var = tvm.tir.SizeVar("num_heads_var", "int32")

        @T.prim_func
        def token_quantize_grouped_warp(
            Had_In:      T.Buffer([rows_var, num_heads_var, head_dim], "bfloat16"),
            Row_Min:     T.Buffer([rows_var, num_heads_var], "float32"),
            Row_Max:     T.Buffer([rows_var, num_heads_var], "float32"),
            GroupMap:    T.Buffer([num_heads_var], "int32"),
            Max_Value:   T.Buffer([num_heads_var], "float32"),
            Q_Out:       T.Buffer([rows_var, num_heads_var, head_dim], "uint8"),
            G_Min_Out:   T.Buffer([rows_var, num_heads_var], "float32"),
            G_Scale_Out: T.Buffer([rows_var, num_heads_var], "float32"),
        ):
            with T.Kernel(T.ceildiv(rows_var, warps), num_heads_var, threads=threads) as (bx, by):
                tid = T.get_thread_binding()
                warp = tid // WARP
                lane = tid % WARP
                row_idx = bx * warps + warp
                if row_idx < rows_var:
                    gmin = T.alloc_local([1], "float32")
                    gmax = T.alloc_local([1], "float32")
                    gmin[0] = T.float32(1e10)
                    gmax[0] = T.float32(-1e10)
                    for h in range(num_heads_const):              # grouped reduce in registers
                        if GroupMap[h] == GroupMap[by]:
                            gmin[0] = T.min(gmin[0], Row_Min[row_idx, h])
                            gmax[0] = T.max(gmax[0], Row_Max[row_idx, h])
                    mv = Max_Value[by]
                    scale = T.max(gmax[0] - gmin[0], 1e-5) / (mv - 1.0)
                    if lane == 0:
                        G_Min_Out[row_idx, by] = gmin[0]
                        G_Scale_Out[row_idx, by] = scale
                    for c in range(per_lane):
                        j = lane + c * WARP                        # coalesced
                        val = T.cast(Had_In[row_idx, by, j], "float32")
                        normalized = (val - gmin[0]) / scale
                        clamped = T.min(T.max(T.floor(normalized + 0.5), 0.0), mv)
                        Q_Out[row_idx, by, j] = T.cast(clamped, "uint8")

        return tilelang.compile(token_quantize_grouped_warp, target="cuda")

    # ===================================================================
    # Standalone reduce-only kernels（不含 Hadamard，接受已变换数据）
    # ===================================================================
    def _compile_channel_reduce_only_kernel(self):
        """Tile-level channel min/max reduce（无 Hadamard gemm）"""
        head_dim = self.head_dim
        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")
        num_tiles_var = tvm.tir.SizeVar("num_tiles_var", "int32")
        num_heads_var = tvm.tir.SizeVar("num_heads_var", "int32")

        @T.prim_func
        def channel_reduce_only(
            Data:     T.Buffer([rows_var, num_heads_var, head_dim], "float16"),
            Tile_Min: T.Buffer([num_tiles_var, num_heads_var, head_dim], "float32"),
            Tile_Max: T.Buffer([num_tiles_var, num_heads_var, head_dim], "float32"),
        ):
            with T.Kernel(num_tiles_var, num_heads_var, threads=128) as (bx, by):
                data_shared = T.alloc_shared([32, head_dim], "float16")
                ch_min = T.alloc_fragment([head_dim], "float32")
                ch_max = T.alloc_fragment([head_dim], "float32")

                for j in T.Parallel(head_dim):
                    ch_min[j] = 1e10
                    ch_max[j] = -1e10

                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        data_shared[i, j] = Data[row_idx, by, j]
                    else:
                        data_shared[i, j] = 0.0

                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        val = T.cast(data_shared[i, j], "float32")
                        ch_min[j] = T.min(ch_min[j], val)
                        ch_max[j] = T.max(ch_max[j], val)

                for j in T.Parallel(head_dim):
                    Tile_Min[bx, by, j] = ch_min[j]
                    Tile_Max[bx, by, j] = ch_max[j]

        return tilelang.compile(channel_reduce_only, target="cuda")

    def _compile_token_reduce_only_kernel(self):
        """Per-row per-head min/max reduce（无 Hadamard gemm）"""
        head_dim = self.head_dim
        import tvm
        rows_var = tvm.tir.SizeVar("rows_var", "int32")
        num_heads_var = tvm.tir.SizeVar("num_heads_var", "int32")

        @T.prim_func
        def token_reduce_only(
            Data:    T.Buffer([rows_var, num_heads_var, head_dim], "float16"),
            Row_Min: T.Buffer([rows_var, num_heads_var], "float32"),
            Row_Max: T.Buffer([rows_var, num_heads_var], "float32"),
        ):
            with T.Kernel(T.ceildiv(rows_var, 32), num_heads_var, threads=128) as (bx, by):
                data_shared = T.alloc_shared([32, head_dim], "float16")
                max_shared  = T.alloc_shared([32], "float32")
                min_shared  = T.alloc_shared([32], "float32")

                for i, j in T.Parallel(32, head_dim):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        data_shared[i, j] = Data[row_idx, by, j]
                    else:
                        data_shared[i, j] = 0.0

                T.reduce_max(data_shared, max_shared, dim=-1, clear=True)
                T.reduce_min(data_shared, min_shared, dim=-1, clear=True)

                for i in T.Parallel(32):
                    row_idx = bx * 32 + i
                    if row_idx < rows_var:
                        Row_Min[row_idx, by] = min_shared[i]
                        Row_Max[row_idx, by] = max_shared[i]

        return tilelang.compile(token_reduce_only, target="cuda")

    # ===================================================================
    # FWHT 兼容的压缩方法（接受已 FWHT 变换的数据，只做量化）
    # ===================================================================
    def _get_uniform_max_value_vec(
        self,
        value: float,
        num_heads: int,
        device,
    ) -> torch.Tensor:
        key = (str(device), num_heads, float(value))
        if key not in self._uniform_max_vals_cache:
            self._uniform_max_vals_cache[key] = torch.full(
                (num_heads,), float(value), dtype=torch.float32, device=device)
        return self._uniform_max_vals_cache[key]

    def _get_token_group_map(
        self,
        max_value_vec,
        *,
        layer_id: int,
        device,
        num_heads: int,
    ) -> torch.Tensor:
        """Return a per-head group map without GPU->CPU sync in the hot path.

        The async compression worker may run across multiple request lifetimes.
        Cached device tensors must be recorded on the stream that consumes them;
        otherwise PyTorch's allocator can miss cross-stream lifetime edges.
        """
        disable_cache = os.environ.get(
            "KVSERVE_TILELANG_DISABLE_TOKEN_GROUP_CACHE", ""
        ).strip().lower() in {"1", "true", "yes", "on"}

        cpu_key = (layer_id, num_heads, self.split_type)
        if cpu_key not in self._token_group_map_cpu_cache or disable_cache:
            if self.head_scores_mask is not None and hasattr(self, "precomputed_v"):
                max_values = self.precomputed_v[layer_id].tolist()
            elif isinstance(max_value_vec, torch.Tensor):
                # Fallback path for unusual callers.  The KVServe hot path uses
                # precomputed CPU values above, so it does not pay this sync.
                max_values = max_value_vec.detach().cpu().tolist()
            else:
                max_values = [float(max_value_vec)] * num_heads

            val_to_gid = {}
            group_map = []
            for value in max_values:
                value = float(value)
                if value not in val_to_gid:
                    val_to_gid[value] = len(val_to_gid)
                group_map.append(val_to_gid[value])
            if disable_cache:
                return torch.tensor(group_map, dtype=torch.int32, device=device)
            self._token_group_map_cpu_cache[cpu_key] = tuple(group_map)

        group_map = self._token_group_map_cpu_cache[cpu_key]
        device_key = (str(device),) + cpu_key
        if not disable_cache:
            tensor = self._token_group_map_cache.get(device_key)
            if tensor is None:
                tensor = torch.tensor(group_map, dtype=torch.int32, device=device)
                self._token_group_map_cache[device_key] = tensor
            tensor.record_stream(torch.cuda.current_stream(device))
            return tensor
        return torch.tensor(group_map, dtype=torch.int32, device=device)

    def _channel_compress_fwht(self, data_fp16, signs_fp16, max_value_vec, q_out=None):
        """Channel 轴 FWHT 融合压缩（Hadamard + Reduce + Quantize 真融合）。
        q_out 可传入预分配的 [rows,num_heads,head_dim] uint8 缓冲（如 out[0] 切片），
        量化核直接写入它，免去外层 torch.stack 的 272MB memcpy。"""
        rows, num_heads, head_dim = data_fp16.shape
        # vec_shfl 的 channel-reduce 用 bm=16 行/tile(overlap 甜点);f32 版用 32。
        ch_bm = 16 if self.fused_variant == "vec_shfl" else 32
        num_tiles = (rows + ch_bm - 1) // ch_bm
        device = data_fp16.device

        if not hasattr(self, '_fwht_ch_reduce_k'):
            if self.fused_variant == "vec_shfl":
                if self.throttle_gx > 0:
                    self._fwht_ch_reduce_k = self._compile_vec_channel_reduce_persistent(self.throttle_gx)
                else:
                    self._fwht_ch_reduce_k = self._compile_vec_channel_reduce_kernel()
            elif self.fused_variant == "fwht_f16":
                self._fwht_ch_reduce_k = self._compile_fwht_channel_reduce_kernel_f16()
            else:
                self._fwht_ch_reduce_k = self._compile_fwht_channel_reduce_kernel()
        if not hasattr(self, '_ch_quantize_kernel'):
            self._ch_quantize_kernel = self._compile_channel_quantize_kernel()

        had_out = torch.empty((rows, num_heads, head_dim), dtype=torch.bfloat16, device=device)
        tile_min = torch.empty((num_tiles, num_heads, head_dim), dtype=torch.float32, device=device)
        tile_max = torch.empty((num_tiles, num_heads, head_dim), dtype=torch.float32, device=device)

        # Kernel A: FWHT 蝶形 + Channel Reduce（全融合）
        self._fwht_ch_reduce_k(data_fp16, signs_fp16, had_out, tile_min, tile_max)

        global_min = tile_min.amin(dim=0)
        global_max = tile_max.amax(dim=0)
        del tile_min, tile_max

        # Kernel B: 量化
        if q_out is None:
            q_out = torch.empty((rows, num_heads, head_dim), dtype=torch.uint8, device=device)
        if not isinstance(max_value_vec, torch.Tensor):
            max_value_vec = self._get_uniform_max_value_vec(
                max_value_vec, num_heads, device)
        self._ch_quantize_kernel(had_out, global_min, global_max, max_value_vec, q_out)
        del had_out

        scale = (global_max - global_min).clamp_(min=1e-5) / (max_value_vec.unsqueeze(1) - 1)
        meta = {"min_val": global_min, "quant_scale": scale, "axis": "channel"}
        return q_out, meta

    def _token_compress_fwht(self, data_fp16, signs_fp16, max_value_vec, layer_id: int, q_out=None):
        """Token 轴 FWHT 融合压缩（分组归约已融入 GPU 内核，零 Python 开销）。
        q_out 可传入预分配的 [rows,num_heads,head_dim] uint8 缓冲（如 out[1] 切片）。"""
        rows, num_heads, head_dim = data_fp16.shape
        device = data_fp16.device

        if not hasattr(self, '_fwht_tok_reduce_k'):
            if self.fused_variant == "vec_shfl":
                if self.throttle_gx > 0:
                    self._fwht_tok_reduce_k = self._compile_vec_token_reduce_persistent(self.throttle_gx)
                else:
                    self._fwht_tok_reduce_k = self._compile_vec_token_reduce_kernel()
            elif self.fused_variant == "fwht_f16":
                self._fwht_tok_reduce_k = self._compile_fwht_token_reduce_kernel_f16()
            else:
                self._fwht_tok_reduce_k = self._compile_fwht_token_reduce_kernel()
        if not hasattr(self, '_tok_quantize_grouped_k'):
            self._tok_quantize_grouped_k = self._compile_token_quantize_grouped_kernel()

        had_out = torch.empty((rows, num_heads, head_dim), dtype=torch.bfloat16, device=device)
        row_min = torch.empty((rows, num_heads), dtype=torch.float32, device=device)
        row_max = torch.empty((rows, num_heads), dtype=torch.float32, device=device)

        # Kernel A: FWHT 蝶形 + Token Reduce
        self._fwht_tok_reduce_k(data_fp16, signs_fp16, had_out, row_min, row_max)

        # ====== 预计算 GroupMap（仅首次，之后直接查缓存） ======
        if not isinstance(max_value_vec, torch.Tensor):
            max_value_vec = self._get_uniform_max_value_vec(
                max_value_vec, num_heads, device)
        group_map = self._get_token_group_map(
            max_value_vec, layer_id=layer_id, device=device, num_heads=num_heads)

        # Kernel B: 融合分组量化（分组归约 + 量化 + meta 输出，一个 kernel 全搞定）
        if q_out is None:
            q_out = torch.empty((rows, num_heads, head_dim), dtype=torch.uint8, device=device)
        g_min_out = torch.empty((rows, num_heads), dtype=torch.float32, device=device)
        g_scale_out = torch.empty((rows, num_heads), dtype=torch.float32, device=device)
        self._tok_quantize_grouped_k(had_out, row_min, row_max, group_map, max_value_vec,
                                     q_out, g_min_out, g_scale_out)
        del had_out, row_min, row_max

        meta = {
            "min_val": g_min_out,
            "quant_scale": g_scale_out,
            "axis": "token_perhead",
        }
        return q_out, meta

    def _channel_fused_compress(self, data_flat, h_batched, max_value_vec):
        """
        Channel 轴融合压缩：2 个 TileLang 内核 + 1 次微型归约
        
        Args:
            data_flat:     [rows, num_heads, head_dim] float16
            h_batched:     [num_heads, head_dim, head_dim] float16
            max_value_vec: [num_heads] float32 — 每头的量化上限
        """
        rows, num_heads, head_dim = data_flat.shape
        num_tiles = (rows + 31) // 32
        device = data_flat.device

        # 懒编译
        if not hasattr(self, '_had_ch_reduce_kernel'):
            self._had_ch_reduce_kernel = self._compile_hadamard_channel_reduce_kernel()
        if not hasattr(self, '_ch_quantize_kernel'):
            self._ch_quantize_kernel = self._compile_channel_quantize_kernel()

        # 预分配输出
        had_out = torch.empty((rows, num_heads, head_dim), dtype=torch.float16, device=device)
        tile_min = torch.empty((num_tiles, num_heads, head_dim), dtype=torch.float32, device=device)
        tile_max = torch.empty((num_tiles, num_heads, head_dim), dtype=torch.float32, device=device)

        # ─── Kernel A: Hadamard + 局部 Channel Reduce ───
        self._had_ch_reduce_kernel(data_flat, h_batched, had_out, tile_min, tile_max)

        # ─── 微型归约 ───
        global_min = tile_min.amin(dim=0)  # [heads, dim]
        global_max = tile_max.amax(dim=0)  # [heads, dim]
        del tile_min, tile_max

        # ─── Kernel B: Channel 量化（per-head max_value）───
        q_out = torch.empty((rows, num_heads, head_dim), dtype=torch.uint8, device=device)
        if not isinstance(max_value_vec, torch.Tensor):
            max_value_vec = torch.full((num_heads,), max_value_vec, dtype=torch.float32, device=device)
        self._ch_quantize_kernel(had_out, global_min, global_max, max_value_vec, q_out)
        del had_out

        # 元数据
        scale = (global_max - global_min).clamp_(min=1e-5)
        # max_value_vec: [heads] → [heads, 1] for broadcast
        scale = scale / (max_value_vec.unsqueeze(1) - 1)
        meta = {
            "min_val": global_min,
            "quant_scale": scale,
            "axis": "channel",
        }

        return q_out, meta

    def _token_fused_compress(self, data_flat, h_batched, max_value_vec):
        """
        Token 轴融合压缩：2 个 TileLang 内核 + 1 次微型归约
        
        Args:
            data_flat:     [rows, num_heads, head_dim] float16
            h_batched:     [num_heads, head_dim, head_dim] float16
            max_value_vec: [num_heads] float32 — 每头的量化上限
        """
        rows, num_heads, head_dim = data_flat.shape
        device = data_flat.device

        # 懒编译
        if not hasattr(self, '_had_tok_reduce_kernel'):
            self._had_tok_reduce_kernel = self._compile_token_quantize_kernel()
        if not hasattr(self, '_tok_quantize_global_kernel'):
            self._tok_quantize_global_kernel = self._compile_token_quantize_global_kernel()

        # 预分配输出
        had_out = torch.empty((rows, num_heads, head_dim), dtype=torch.float16, device=device)
        row_min = torch.empty((rows, num_heads), dtype=torch.float32, device=device)
        row_max = torch.empty((rows, num_heads), dtype=torch.float32, device=device)

        # ─── Kernel A: Hadamard + 每行每头 min/max ───
        self._had_tok_reduce_kernel(data_flat, h_batched, had_out, row_min, row_max)

        # ─── 微型归约 ───
        global_min = row_min.amin(dim=1)  # [rows]
        global_max = row_max.amax(dim=1)  # [rows]
        del row_min, row_max

        # ─── Kernel B: Token 量化（per-head max_value）───
        q_out = torch.empty((rows, num_heads, head_dim), dtype=torch.uint8, device=device)
        if not isinstance(max_value_vec, torch.Tensor):
            max_value_vec = torch.full((num_heads,), max_value_vec, dtype=torch.float32, device=device)
        self._tok_quantize_global_kernel(had_out, global_min, global_max, max_value_vec, q_out)
        del had_out

        # 元数据
        avg_max_val = max_value_vec.mean().item()
        scale = (global_max - global_min).clamp_(min=1e-5).div_(avg_max_val - 1)
        meta = {
            "min_val": global_min.unsqueeze(1).unsqueeze(2),
            "quant_scale": scale.unsqueeze(1).unsqueeze(2),
            "axis": "token",
        }

        return q_out, meta

    def compress_v3(
        self,
        kv_layer: torch.Tensor,   # [2, blocks, block_size, heads, dim]
        layer_id: int,
        axis_key:   str = "channel",
        axis_value: str = "token",
    ) -> torch.Tensor:
        """
        真融合算子 v2：FWHT 蝶形 + TileLang 量化，零 head-split
        
        Key:   FWHT → channel reduce → channel quantize（per-head max_value）
        Value: FWHT → token reduce → token quantize v2（per-row per-head，无需 head-split）
        
        总计只需 4 次 TileLang kernel launch（vs 原版 ~31 次 PyTorch kernel）
        精度：float32 蝶形 + per-head 量化 ≥ 原版 0.963
        """
        keys = kv_layer[0]
        values = kv_layer[1]
        num_blocks, block_size, num_heads, head_dim = keys.shape
        rows = num_blocks * block_size
        device = keys.device

        signs = self._get_signs(layer_id, device)

        # per-head max_value
        if self.head_scores_mask is not None and hasattr(self, 'precomputed_k'):
            if device not in self._max_vals_cache:
                self._max_vals_cache[device] = (
                    self.precomputed_k.to(device),
                    self.precomputed_v.to(device)
                )
            cache_k, cache_v = self._max_vals_cache[device]
            key_max_vals = cache_k[layer_id]
            val_max_vals = cache_v[layer_id]
        else:
            key_max_vals = torch.full((num_heads,), self.high_key_max_value,
                                     dtype=torch.float32, device=device)
            val_max_vals = torch.full((num_heads,), self.high_value_max_value,
                                     dtype=torch.float32, device=device)

        # 预分配统一输出 [2, blocks, block_size, heads, dim]，量化核直接写进 out[0]/out[1]
        # 的连续切片，消除原 torch.stack 的 272MB 纯 memcpy（占 compress_v3 ~14-17%）。
        out = torch.empty((2, num_blocks, block_size, num_heads, head_dim),
                          dtype=torch.uint8, device=device)
        k_buf = out[0].reshape(rows, num_heads, head_dim)   # 连续 view，零拷贝
        v_buf = out[1].reshape(rows, num_heads, head_dim)

        # ─── Key：FWHT + channel reduce + quantize（2 kernels）───
        keys_bf16 = keys.reshape(rows, num_heads, head_dim)
        _, k_meta = self._channel_compress_fwht(keys_bf16, signs, key_max_vals, q_out=k_buf)

        # ─── Value：FWHT + token reduce + quantize v2（2 kernels，零 head-split）───
        vals_bf16 = values.reshape(rows, num_heads, head_dim)
        _, v_meta = self._token_compress_fwht(
            vals_bf16, signs, val_max_vals, layer_id, q_out=v_buf)
        metadata = {
            "key_meta": k_meta,
            "value_meta": v_meta,
            "key_max_vals": key_max_vals,
            "value_max_vals": val_max_vals,
        }
        return out, metadata

    def decompress(
        self,
        q_kv:    torch.Tensor,    # uint8 压缩后的 KV Cache
        metadata: dict,           # compress() 返回的元数据字典
        layer_id: int,
        max_val: float = None,    # 遗留参数，已被 metadata["max_vals"] 替代
    ) -> torch.Tensor:
        """
        解压缩：先反量化（将 uint8 还原成浮点数），再施加逆哈达玛变换。
        返回与原始 KV Cache 尺寸相同的 bfloat16 张量。
        """
        import fast_hadamard_transform as _fht
        
        num_blocks, block_size, num_heads, head_dim = q_kv.shape
        rows = num_blocks * block_size
        
        # 将 uint8 反量化回浮点数
        device = q_kv.device
        q_f = q_kv.float()
        # 【对应 compress 里的 .cpu()】：解压缩时把 metadata 从 CPU RAM 搬回 GPU，
        # 这样后续的 GPU 矩阵运算才能正常进行。
        s_view = metadata["scales"].to(device).view(num_blocks, block_size, num_heads, 1)
        m_vals = metadata["max_vals"].to(device).view(1, 1, num_heads, 1)
        
        if self.quant_type == "absmax":
            # 对称反量化公式： (uint8_val - m_val) / m_val * scale
            dequant = ((q_f - m_vals) / m_vals) * s_view
        elif self.quant_type == "minmax":
            # 非对称反量化公式： uint8_val * scale + min_val
            m_view = metadata["mins"].to(device).view(num_blocks, block_size, num_heads, 1)
            dequant = q_f * s_view + m_view
            
        hkv = dequant.to(self.dtype)

        # 施加逆哈达玛变换还原到应用哈达玛变换前的空间
        # 注意：这里用的是封装好的快速 Walsh-Hadamard Transform，不是 TileLang
        scale = 1.0 / math.sqrt(head_dim)
        rotated = _fht.hadamard_transform(hkv.contiguous(), scale=scale)
        # 将随机符号再乘回去（因为压缩时乘了符号，解压缩时要再乘一次去抚平）
        signs = self._get_signs(layer_id, q_kv.device)
        kv_restored = rotated * signs.view(1, 1, num_heads, head_dim)
        
        return kv_restored

    def decompress_v3(
        self,
        q_kv:     torch.Tensor,   # [2, blocks, block_size, heads, dim] uint8
        metadata: dict,           # compress_v3() 返回的元数据
        layer_id: int,
    ) -> torch.Tensor:
        """
        解压缩 compress_v3 的输出（精度对齐版）。
        
        流程：
          1. 用 key_meta / value_meta 做反量化（支持 per-group）
          2. 施加逆 Hadamard 变换 + 逆随机符号，还原到原始 KV 空间
        """
        import fast_hadamard_transform as _fht
        
        device = q_kv.device
        keys = q_kv[0]     # [blocks, block_size, heads, dim]
        values = q_kv[1]
        num_blocks, block_size, num_heads, head_dim = keys.shape
        
        # ─── Step 1: 反量化 ───
        def _dequant(q_data, meta):
            """反量化，支持单组和 per-group 两种 metadata 格式"""
            if "per_group_meta" in meta:
                # per-group 模式：每组头有独立的 min_val/scale
                result = torch.empty_like(q_data, dtype=self.dtype)
                nb, bs = q_data.shape[0], q_data.shape[1]
                for mv_val, group_meta in meta["per_group_meta"].items():
                    head_idx = group_meta["head_idx"].to(device)
                    subset = q_data[:, :, head_idx, :]
                    q_f = subset.to(self.dtype)
                    min_val = group_meta["min_val"].to(device=device, dtype=self.dtype)
                    scale   = group_meta["quant_scale"].to(device=device, dtype=self.dtype)
                    # token 轴 meta 形状 [rows, 1, 1] → reshape 为 [blocks, block_size, 1, 1]
                    if min_val.dim() == 3 and min_val.shape[0] == nb * bs:
                        min_val = min_val.reshape(nb, bs, 1, 1)
                        scale   = scale.reshape(nb, bs, 1, 1)
                    result[:, :, head_idx, :] = q_f * scale + min_val
                return result
            else:
                # 单组模式
                if meta.get("min_val") is None or q_data.numel() == 0:
                    return q_data.to(self.dtype)
                
                q_f = q_data.to(self.dtype)
                min_val = meta["min_val"].to(device=device, dtype=self.dtype)
                scale   = meta["quant_scale"].to(device=device, dtype=self.dtype)
                nb, bs = q_data.shape[0], q_data.shape[1]
                axis = meta.get("axis", "")
                
                if axis == "token_perhead" and min_val.dim() == 2:
                    # per-row per-head: [rows, heads] → [blocks, block_size, heads, 1]
                    min_val = min_val.reshape(nb, bs, -1, 1)
                    scale   = scale.reshape(nb, bs, -1, 1)
                elif min_val.dim() == 2:
                    # channel: [heads, dim] → [1, 1, heads, dim]
                    min_val = min_val.unsqueeze(0).unsqueeze(0)
                    scale   = scale.unsqueeze(0).unsqueeze(0)
                elif min_val.dim() == 3 and min_val.shape[0] == nb * bs:
                    # token: [rows, 1, 1] → [blocks, block_size, 1, 1]
                    min_val = min_val.reshape(nb, bs, 1, 1)
                    scale   = scale.reshape(nb, bs, 1, 1)
                return q_f * scale + min_val
        
        keys_f   = _dequant(keys,   metadata["key_meta"])
        values_f = _dequant(values, metadata["value_meta"])
        
        # ─── Step 2: 逆 Hadamard 变换 + 逆随机符号 ───
        scale_h = 1.0 / math.sqrt(head_dim)
        signs = self._get_signs(layer_id, device)
        signs_view = signs.view(1, 1, num_heads, head_dim)
        
        keys_r   = _fht.hadamard_transform(keys_f.contiguous(), scale=scale_h) * signs_view
        values_r = _fht.hadamard_transform(values_f.contiguous(), scale=scale_h) * signs_view
        del keys_f, values_f
        
        return torch.stack([keys_r, values_r], dim=0)


# ===========================================================================
# 精度验证 — 使用 compress_v3 + decompress_v3 完整闭环测试
# ===========================================================================

if __name__ == "__main__":
    import os
    import sys
    
    # 动态添加项目根目录到 Python 路径，以防找不到 kvserve 模块
    workspace_dir = "/workspace"
    if os.path.exists(workspace_dir) and workspace_dir not in sys.path:
        sys.path.insert(0, workspace_dir)
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        
    torch.manual_seed(42)

    test_files = [
        "/workspace/req_0_kv.pkl",
        "/workspace/req_1_kv.pkl",
    ]

    for pkl_path in test_files:
        print(f"\n{'='*60}")
        if not os.path.exists(pkl_path):
            print(f"{pkl_path} not found, skipping.")
            continue

        print(f"Loading KV payload from {pkl_path}...")
        try:
            kv_data = torch.load(pkl_path, map_location="cpu", weights_only=False)
        except Exception:
            import pickle
            with open(pkl_path, "rb") as f:
                kv_data = pickle.load(f)

        # 解析数据结构
        if isinstance(kv_data, dict):
            kv = kv_data.get("compressed_kv", kv_data.get("kv", None))
            if kv is None:
                for v in kv_data.values():
                    if isinstance(v, torch.Tensor):
                        kv = v
                        break
        else:
            kv = kv_data

        if not isinstance(kv, torch.Tensor):
            print(f"  Could not resolve tensor, skipping.")
            continue

        kv = kv.to(torch.bfloat16).cuda()
        print(f"  原始 shape: {kv.shape}, 大小={kv.nbytes/1024/1024:.1f} MB")

        # 期望 shape: [layers, 2, blocks, block_size, heads, dim]
        if len(kv.shape) != 6:
            print(f"  Unexpected shape {kv.shape}, skipping.")
            continue

        num_layers, kv_dim, blocks, block_size, num_heads, head_dim = kv.shape
        assert kv_dim == 2, f"Expected kv_dim=2, got {kv_dim}"

        # 初始化算子（与原版 KVServe 完全一致的混合精度参数）
        op = KVHadamardQuantOp(
            num_heads=num_heads,
            head_dim=head_dim,
            quant_type="minmax",
            model_name="Llama-3.1-8B-Instruct",
            hybrid_ratio=0.8,
            high_key_max_value=12.0,
            high_value_max_value=8.0,
            low_key_max_value=6.0,
            low_value_max_value=4.0,
            axis_key="channel",
            axis_value="token",
            split_type="head",
        )


        print(f"\n[ ENCODE ] 逐层 compress_v3 ({num_layers} layers)...")
        all_q = []
        all_meta = []
        total_input_bytes = 0
        total_output_bytes = 0

        for l in range(num_layers):
            layer_data = kv[l]  # [2, blocks, block_size, heads, dim]
            q_layer, meta = op.compress_v3(layer_data, layer_id=l)
            all_q.append(q_layer)
            all_meta.append(meta)
            total_input_bytes += layer_data.nbytes
            total_output_bytes += q_layer.nbytes
            if (l + 1) % 8 == 0:
                print(f"    Layer {l+1}/{num_layers} done")

        ratio = total_input_bytes / total_output_bytes
        print(f"  Input: {total_input_bytes/1024/1024:.1f} MB → Output: {total_output_bytes/1024/1024:.1f} MB")
        print(f"  🚀 Compression Ratio: {ratio:.2f}x")

        print(f"\n[ DECODE ] 逐层 decompress_v3 ({num_layers} layers)...")
        all_restored = []
        for l in range(num_layers):
            restored = op.decompress_v3(all_q[l], all_meta[l], layer_id=l)
            all_restored.append(restored)
            if (l + 1) % 8 == 0:
                print(f"    Layer {l+1}/{num_layers} done")

        # 拼接所有层计算全局精度
        original_flat = kv.float().flatten()
        restored_flat = torch.cat([r.float().flatten() for r in all_restored])

        sim = torch.cosine_similarity(original_flat, restored_flat, dim=0)
        mse = torch.nn.functional.mse_loss(original_flat, restored_flat)

        print(f"\n  🎯 Cosine Similarity (Precision): {sim:.6f}")
        print(f"  🎯 Mean Squared Error (Loss)    : {mse:.6f}")
        print(f"  ✅ 已完成 {pkl_path}")

        # 释放显存
        del all_q, all_meta, all_restored, original_flat, restored_flat

    print("\nAll tasks done!")
