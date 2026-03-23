# KV 通信压缩框架设计文档（基于 vLLM V1）

## 目标与约束

**目标**：在两个独立的 vLLM V1 实例（Prefill / Decode）之间做 KV 缓存的压缩传输，降低跨实例网络带宽。

**核心约束**：
- 不 fork vLLM，不修改 vLLM 任何代码
- 不自定义调度器和 KV block 管理，全部由 vLLM V1 原生负责
- 只实现一个 `KVConnectorBase_V1` 子类
- 复用 kvserve 现有 `CompressionManager` 模块（微小适配）

---

## 一、vLLM V1 PD 分离机制（精确描述）

vLLM V1 通过 `KVConnectorBase_V1`（`vllm/distributed/kv_transfer/kv_connector/v1/base.py`）实现 PD 分离，该类按 `role` 参数分为两个角色，运行在不同进程：

```
Prefill vLLM Instance               Decode vLLM Instance
┌─────────────────────────┐         ┌─────────────────────────┐
│  Scheduler Process      │         │  Scheduler Process      │
│  ┌──────────────────┐   │         │  ┌──────────────────┐   │
│  │ Connector        │   │         │  │ Connector        │   │
│  │ role=SCHEDULER   │   │         │  │ role=SCHEDULER   │   │
│  └──────────────────┘   │         │  └──────────────────┘   │
│  Worker Process(es)     │   KV    │  Worker Process(es)     │
│  ┌──────────────────┐   │ ──────► │  ┌──────────────────┐   │
│  │ Connector        │   │         │  │ Connector        │   │
│  │ role=WORKER      │   │         │  │ role=WORKER      │   │
│  └──────────────────┘   │         │  └──────────────────┘   │
└─────────────────────────┘         └─────────────────────────┘
```

### 1.1 精确调用时序

**Prefill 实例（Producer，`kv_role="kv_producer"`）**：

```
每轮调度（scheduler.py:164 schedule()）：
  connector[SCHEDULER].get_num_new_matched_tokens(request, computed)
    → 返回 (0, False)                          # producer 不从远端 load
  connector[SCHEDULER].update_state_after_alloc(request, blocks, 0)
  connector[SCHEDULER].build_connector_meta(scheduler_output)
    → 打包 {request_id, token_ids, block_ids, slot_mapping}

worker.execute_model(scheduler_output)（mixin.py:43）：
  connector[WORKER].start_load_kv(forward_context)  # producer: no-op
  for each layer:
    unified_attention_with_output()                  # layer.py:484
      → wait_for_kv_layer_from_connector(layer_name) # producer: no-op
      → [attention compute，KV 写入 paged buffer]
      → maybe_save_kv_layer_to_connector(layer_name, kv_cache)
          → connector[WORKER].save_kv_layer(...)     # ← 压缩在这里触发
  connector[WORKER].wait_for_save()                  # 阻塞直到发送完成

request_finished()（scheduler.py:1101）：
  connector[SCHEDULER].request_finished(request, block_ids)
    → 发送未完成 → 返回 (True, None)，延迟 block 释放
    → get_finished() 返回该 request_id 后，block 才释放
```

**Decode 实例（Consumer，`kv_role="kv_consumer"`）**：

```
每轮调度：
  connector[SCHEDULER].get_num_new_matched_tokens(request, 0)
    → 返回 (prompt_len - 1, True)              # True = async load
    → 调度器将 request 置为 WAITING_FOR_REMOTE_KVS，跳过本轮（scheduler.py:339）
  connector[SCHEDULER].update_state_after_alloc(request, blocks, num_external)
    → 记录 request_id → block_ids（用于后续注入时定位 slot）

后台 Transport 线程接收压缩 KV，解压后缓存。
接收完成 → 将 request_id 加入 finished_recving_kv_req_ids。

下轮调度：
  _update_waiting_for_remote_kv(request) 检测到完成（scheduler.py:1107）
  → request 状态 WAITING_FOR_REMOTE_KVS → WAITING → 正常排入调度

worker.execute_model()：
  connector[WORKER].start_load_kv(forward_context)
    → 将已解压的 KV 注入 paged buffer（按 slot_mapping scatter）
  for each layer:
    wait_for_layer_load(layer_name)    # 同步注入方案：no-op
    [attention compute，正常使用已注入的 KV]
    save_kv_layer → consumer: no-op
  wait_for_save() → consumer: no-op
```

### 1.2 关键约束（来自实际代码）

- `save_kv_layer` 收到的 `kv_layer` 是**整层 paged 大张量** `[2, num_pages, page_size, num_kv_heads, head_size]`，包含当前 batch 所有 request 的所有 page。必须用 `slot_mapping` gather 提取本 request 的 tokens，再压缩发送
- `maybe_save_kv_layer_to_connector` 只在 `torch.ops.vllm.unified_attention_with_output`（CUDA 编译路径）中调用，GPU 平台 `use_direct_call=False`，正常走此路径
- Block 分配完全由 `KVCacheManager` 负责，connector 只能在 `update_state_after_alloc` 中被动接收已分配的 block_ids，不能干涉分配
- `get_num_new_matched_tokens` 返回 `(n, True)` → 异步（请求进入 `WAITING_FOR_REMOTE_KVS`）；返回 `(n, False)` → 同步（要求 `start_load_kv` 立即注入）
- 注册自定义 Connector 无需修改 vLLM：`KVTransferConfig.kv_connector_module_path` 指向模块路径，工厂用 `importlib.import_module` 动态加载

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      用户应用层                                  │
│            (HTTP API / OpenAI-compatible endpoint)              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  PDCoordinator     │  ← 新增（轻量）
                    │  (路由 + 连接管理)  │    负责路由请求到 P/D
                    └──────┬──────┬──────┘
                           │      │
              ┌────────────▼──┐  ┌▼──────────────────┐
              │ vLLM V1       │  │ vLLM V1            │
              │ AsyncLLM      │  │ AsyncLLM           │
              │ (Prefill)     │  │ (Decode)           │
              │               │  │                    │
              │ KVTransferCfg │  │ KVTransferCfg      │
              │ kv_role=      │  │ kv_role=           │
              │ "kv_producer" │  │ "kv_consumer"      │
              │               │  │                    │
              │ ┌───────────┐ │  │ ┌───────────────┐ │
              │ │Compressed │ │  │ │ Compressed    │ │
              │ │KVConnector│ │  │ │ KVConnector   │ │
              │ └─────┬─────┘ │  │ └──────┬────────┘ │
              └───────┼───────┘  └─────────┼──────────┘
                      │                    │
                      └────────┬───────────┘
                     压缩 KV 传输（变长字节流）
```

vLLM V1 负责全部调度、KV block 分配/回收、prefix cache、TP/PP。本框架只实现传输层的压缩。

---

## 三、模块设计

### 3.1 目录结构

```
kvserve/
├── connector/
│   ├── __init__.py
│   ├── compressed_kv_connector.py     # 新建：KVConnectorBase_V1 实现
│   ├── transport/
│   │   ├── __init__.py
│   │   ├── base.py                    # 新建：Transport 抽象接口
│   │   └── zmq_transport.py           # 新建：ZMQ 实现
│   └── utils.py                       # 新建：KV 提取/注入工具函数
├── manager/
│   └── compression_manager.py         # 复用：CompressionManager（微小改动）
└── coordinator/
    └── pd_coordinator.py              # 新建：轻量路由层
```

### 3.2 `CompressedKVConnector`（核心模块）

```python
# kvserve/connector/compressed_kv_connector.py

from dataclasses import dataclass, field
from typing import Optional
import threading
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.v1.core.sched.output import SchedulerOutput
from kvserve.manager.compression_manager import CompressionManager, CompressionConfig
from kvserve.connector.utils import (
    _extract_kv_from_paged_buffer, _inject_kv_to_paged_buffer, _make_slot_mapping)


@dataclass
class ReqKVBuffer:
    """单个 request 的 KV 数据缓存（逐层积累 / 解压后存放）"""
    request_id: str
    token_ids: list
    block_ids: list
    slot_mapping: torch.Tensor
    layer_kv_buffers: dict = field(default_factory=dict)   # producer: 逐层积累
    decompressed_kv: Optional[dict] = None                 # consumer: 解压后待注入


@dataclass
class CompressedKVConnectorMetadata(KVConnectorMetadata):
    reqs_to_send: list = field(default_factory=list)    # producer worker 侧
    reqs_to_inject: list = field(default_factory=list)  # consumer worker 侧


class CompressedKVConnector(KVConnectorBase_V1):
    """
    实现 KVConnectorBase_V1。
    压缩复用 kvserve.manager.compression_manager.CompressionManager。
    传输通过 Transport 接口解耦，默认 ZMQ。
    """

    def __init__(self, vllm_config, role: KVConnectorRole):
        super().__init__(vllm_config, role)
        cfg = vllm_config.kv_transfer_config
        self.is_producer = cfg.is_kv_producer
        self.block_size = vllm_config.cache_config.block_size

        extra = cfg.kv_connector_extra_config
        comp_cfg_dict = extra.get("compression_config", {})
        self.comp_config = CompressionConfig(**comp_cfg_dict) if comp_cfg_dict \
                           else CompressionConfig()

        # 调度器侧状态
        self._pending_send: dict = {}
        self._pending_inject: dict = {}
        self._waiting_recv: dict = {}
        self._finished_recving: set = set()
        self._finished_sending: set = set()

        # Worker 侧资源
        if role == KVConnectorRole.WORKER:
            self._compression_manager = CompressionManager(self.comp_config)
            self._transport = self._build_transport(cfg, extra)
            if not self.is_producer:
                self._recv_thread = threading.Thread(
                    target=self._recv_loop, daemon=True)
                self._recv_thread.start()

    # ── Scheduler-side ────────────────────────────────────────────

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if self.is_producer:
            return 0, False
        num_external = len(request.prompt_token_ids) - 1 - num_computed_tokens
        if num_external <= 0:
            return 0, False
        return num_external, True  # True → WAITING_FOR_REMOTE_KVS

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if self.is_producer or num_external_tokens == 0:
            return
        block_ids = blocks.get_block_ids()[0]
        slot_mapping = _make_slot_mapping(
            request.prompt_token_ids, block_ids, self.block_size)
        self._waiting_recv[request.request_id] = ReqKVBuffer(
            request_id=request.request_id,
            token_ids=list(request.prompt_token_ids),
            block_ids=list(block_ids),
            slot_mapping=slot_mapping,
        )

    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        meta = CompressedKVConnectorMetadata()
        if self.is_producer:
            for req in scheduler_output.scheduled_new_reqs:
                slot_mapping = _make_slot_mapping(
                    req.prompt_token_ids, req.block_ids[0], self.block_size)
                buf = ReqKVBuffer(
                    request_id=req.req_id,
                    token_ids=list(req.prompt_token_ids),
                    block_ids=list(req.block_ids[0]),
                    slot_mapping=slot_mapping,
                )
                self._pending_send[req.req_id] = buf
                meta.reqs_to_send.append(buf)
        else:
            for req in scheduler_output.scheduled_new_reqs:
                rid = req.req_id
                if rid in self._pending_inject:
                    meta.reqs_to_inject.append(self._pending_inject.pop(rid))
        return meta

    def update_connector_output(self, connector_output):
        if connector_output.finished_sending:
            self._finished_sending.update(connector_output.finished_sending)
        if connector_output.finished_recving:
            for rid in connector_output.finished_recving:
                if rid in self._waiting_recv:
                    self._pending_inject[rid] = self._waiting_recv.pop(rid)
            self._finished_recving.update(connector_output.finished_recving)

    def request_finished(self, request, block_ids):
        if self.is_producer:
            rid = request.request_id
            if rid in self._pending_send and rid not in self._finished_sending:
                return True, None   # 发送未完成，延迟 block 释放
        return False, None

    # ── Worker-side ───────────────────────────────────────────────

    def start_load_kv(self, forward_context, **kwargs):
        """Consumer: forward pass 开始前将已解压 KV 注入 paged buffer"""
        if self.is_producer:
            return
        meta = self._get_connector_metadata()
        assert isinstance(meta, CompressedKVConnectorMetadata)
        for req_buf in meta.reqs_to_inject:
            if req_buf.decompressed_kv is None:
                continue
            for layer_name, kv_tensor in req_buf.decompressed_kv.items():
                layer = forward_context.no_compile_layers.get(layer_name)
                if layer is None:
                    continue
                kv_cache_layer = layer.kv_cache[forward_context.virtual_engine]
                _inject_kv_to_paged_buffer(
                    kv_cache_layer, kv_tensor, req_buf.slot_mapping)

    def wait_for_layer_load(self, layer_name: str):
        # start_load_kv 已全部同步注入，no-op
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        """
        Producer: 从 paged buffer 提取本 request 的 KV tokens 并缓存。
        kv_layer: [2, num_pages, page_size, num_kv_heads, head_size]
        在 wait_for_save() 里统一压缩全部层后发送。
        """
        if not self.is_producer:
            return
        meta = self._get_connector_metadata()
        assert isinstance(meta, CompressedKVConnectorMetadata)
        for req_buf in meta.reqs_to_send:
            extracted = _extract_kv_from_paged_buffer(
                kv_layer, req_buf.slot_mapping)  # [2, num_tokens, num_kv_heads, head_size]
            req_buf.layer_kv_buffers[layer_name] = extracted

    def wait_for_save(self):
        """Producer: 所有层积累完毕，压缩并发送。"""
        if not self.is_producer:
            return
        meta = self._get_connector_metadata()
        assert isinstance(meta, CompressedKVConnectorMetadata)
        for req_buf in meta.reqs_to_send:
            if not req_buf.layer_kv_buffers:
                continue
            layer_names_ordered = sorted(req_buf.layer_kv_buffers.keys())
            stacked = torch.stack(
                [req_buf.layer_kv_buffers[n] for n in layer_names_ordered],
                dim=0)  # [num_layers, 2, num_tokens, num_kv_heads, head_size]

            # 复用 CompressionManager（kvserve/manager/compression_manager.py）
            compressed = self._compression_manager.compress_all_layers(stacked)

            self._transport.send(
                request_id=req_buf.request_id,
                layer_names=layer_names_ordered,
                compressed_data=self._compression_manager.compressed_to_bytes(compressed),
                original_shape=tuple(stacked.shape),
                original_dtype=str(stacked.dtype),
            )

    def get_finished(self, finished_req_ids):
        newly_sent, newly_recv = self._transport.poll_finished()
        sent = newly_sent & finished_req_ids if newly_sent else None
        recv = newly_recv if newly_recv else None
        return sent, recv

    # ── 内部辅助 ──────────────────────────────────────────────────

    def _recv_loop(self):
        """Consumer 后台线程：持续接收压缩 KV，解压后写入 req_buf"""
        while True:
            pkt = self._transport.recv_blocking()
            stacked = self._compression_manager.decompress_all_layers(
                self._compression_manager.bytes_to_compressed(pkt.compressed_bytes),
                pkt.original_shape,
                pkt.original_dtype,
            )  # [num_layers, 2, num_tokens, ...]
            if pkt.request_id in self._waiting_recv:
                req_buf = self._waiting_recv[pkt.request_id]
                req_buf.decompressed_kv = {
                    name: stacked[i] for i, name in enumerate(pkt.layer_names)}
                self._finished_recving.add(pkt.request_id)

    @staticmethod
    def _build_transport(cfg, extra):
        from kvserve.connector.transport.zmq_transport import ZmqTransport
        return ZmqTransport(
            is_sender=cfg.is_kv_producer,
            kv_ip=cfg.kv_ip,
            kv_port=cfg.kv_port,
        )
```

### 3.3 KV 提取 / 注入工具

```python
# kvserve/connector/utils.py
import torch

def _extract_kv_from_paged_buffer(
    kv_layer: torch.Tensor,      # [2, num_pages, page_size, num_kv_heads, head_size]
    slot_mapping: torch.Tensor,  # [num_tokens]，绝对 slot 索引
) -> torch.Tensor:
    """从 paged buffer 中 gather 出该 request 的 KV tokens"""
    num_pages, page_size = kv_layer.shape[1], kv_layer.shape[2]
    flat = kv_layer.reshape(2, num_pages * page_size, -1)  # [2, total_slots, H*D]
    return flat[:, slot_mapping, :]                        # [2, num_tokens, H*D]


def _inject_kv_to_paged_buffer(
    kv_cache_layer: torch.Tensor,  # [2, num_pages, page_size, num_kv_heads, head_size]
    kv_data: torch.Tensor,          # [2, num_tokens, H*D]
    slot_mapping: torch.Tensor,     # [num_tokens]
) -> None:
    num_pages, page_size = kv_cache_layer.shape[1], kv_cache_layer.shape[2]
    flat = kv_cache_layer.reshape(2, num_pages * page_size, -1)
    flat[:, slot_mapping, :] = kv_data


def _make_slot_mapping(
    token_ids, block_ids, block_size: int
) -> torch.Tensor:
    num_tokens = len(token_ids)
    block_ids_t = torch.tensor(block_ids)
    offsets = torch.arange(block_size)
    slots = (block_ids_t.unsqueeze(1) * block_size + offsets).flatten()
    return slots[:num_tokens]
```

### 3.4 Transport 层

Transport 是纯字节流传输，与 vLLM 无关。

```python
# kvserve/connector/transport/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class KVPacket:
    request_id: str
    layer_names: list
    compressed_bytes: bytes
    original_shape: tuple
    original_dtype: str

class TransportBase(ABC):
    @abstractmethod
    def send(self, request_id, layer_names, compressed_data,
             original_shape, original_dtype): ...

    @abstractmethod
    def recv_blocking(self) -> KVPacket: ...

    @abstractmethod
    def poll_finished(self) -> tuple:
        """返回 (已确认发送的 request_id set, 新到达已接收的 request_id set)"""
        ...
```

**ZMQ 消息格式**（发送端 PUSH，接收端 PULL）：

```
[header_size: 4 bytes LE]
[header: JSON，含 request_id, layer_names, original_shape, original_dtype]
[compressed_bytes: N bytes]
```

ZMQ 原生支持变长消息，不需要像 NCCL 一样填充到固定长度。

### 3.5 CompressionManager 复用（微小改动）

现有 `kvserve/manager/compression_manager.py` 的 `compress_all_layers` / `decompress_all_layers` 直接复用，只需新增两个序列化方法：

```python
# 在 CompressionManager 中新增（不改动现有逻辑）
def compressed_to_bytes(self, compressed: CompressedKVData) -> bytes:
    import pickle
    return pickle.dumps(compressed)

def bytes_to_compressed(self, data: bytes) -> CompressedKVData:
    import pickle
    return pickle.loads(data)
```

---

## 四、配置方法

完全通过 `KVTransferConfig` 注入，零改 vLLM 代码：

```python
from vllm import AsyncLLMEngine, AsyncEngineArgs
from vllm.config import KVTransferConfig

# Prefill 实例
prefill_engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
    model="meta-llama/Llama-3-8B",
    kv_transfer_config=KVTransferConfig(
        kv_connector="CompressedKVConnector",
        kv_connector_module_path="kvserve.connector.compressed_kv_connector",
        kv_role="kv_producer",
        kv_rank=0,
        kv_parallel_size=2,
        kv_ip="127.0.0.1",
        kv_port=14579,
        kv_connector_extra_config={
            "compression_config": {
                "quantizer": {"method": "hybrid", "bits": 8},
                "codec": {"method": "lz4"},
                "transformer": {"method": "hadamard"},
            }
        },
    ),
))

# Decode 实例（compression_config 保持一致）
decode_engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
    model="meta-llama/Llama-3-8B",
    kv_transfer_config=KVTransferConfig(
        kv_connector="CompressedKVConnector",
        kv_connector_module_path="kvserve.connector.compressed_kv_connector",
        kv_role="kv_consumer",
        kv_rank=1,
        kv_parallel_size=2,
        kv_ip="127.0.0.1",
        kv_port=14579,
        kv_connector_extra_config={
            "compression_config": { ... },
        },
    ),
))
```

---

## 五、模块关系与复用声明

```
本框架模块                        状态          说明
────────────────────────────────────────────────────────────────────
CompressedKVConnector             新建          实现 KVConnectorBase_V1
  ├─ Scheduler 侧                  新建          调用点：scheduler.py:384,584,1104
  └─ Worker 侧                     新建          调用点：layer.py:413-508, mixin.py:43

CompressionManager                复用          kvserve/manager/compression_manager.py
  compress_all_layers()           复用          输入 [L,2,T,H,D]，输出 CompressedKVData
  decompress_all_layers()         复用          逆过程
  compressed_to_bytes()           新增（2行）   序列化给 Transport
  bytes_to_compressed()           新增（2行）   反序列化

CompressedKVData / CompressionConfig  复用      通过 kv_connector_extra_config 传入

TransportBase / ZmqTransport      新建          替代 kvserve NCCL P2P（适合变长）
utils.py（extract/inject/slot）   新建          替代 worker.extract_kv_blocks()

PDCoordinator                     新建（轻量）  替代 stage_engine.py + backend.py
                                               仅做 HTTP 路由 + engine 生命周期管理

────────────────────── 退役 ────────────────────────────────────────
kvserve/engine/worker.py          退役          vLLM V1 Worker + CacheEngine 取代
kvserve/engine/stage_engine.py    退役          vLLM V1 Scheduler + KVCacheManager 取代
kvserve/engine/block_manager.py   退役          vLLM V1 KVCacheManager 取代
kvserve/engine/kv_transfer.py     退役          CompressedKVConnector 取代
kvserve/engine/worker_steps.py    退役          vLLM V1 model runner 取代
kvserve/engine/backend.py         大幅简化       只保留进程管理和 HTTP 路由
```

---

## 六、关键设计决策与权衡

### 6.1 Buffer-then-compress vs Layer-by-layer

**当前设计**：`save_kv_layer` 逐层积累，`wait_for_save` 统一压缩发送。

| | Buffer-then-compress（当前） | Layer-by-layer（后续优化） |
|---|---|---|
| CompressionManager 复用 | 直接复用，无改动 | 需要拆分为单层接口 |
| 跨层联合量化精度 | 好（全局统计） | 差（每层独立） |
| 延迟 | forward pass 结束后才压缩 | 压缩可与后续层 compute 并行 |
| 实现复杂度 | 低 | 高（需 CUDA stream 管理） |

Phase 1 用 buffer-then-compress；Phase 3 升级为 CUDA stream 异步提取。

### 6.2 ZMQ vs NCCL 传输

| | ZMQ | NCCL |
|---|---|---|
| 变长消息 | 原生支持 | 需定长 header + 填充 |
| GPU Direct | 不支持 | 支持（可降延迟） |
| 实现复杂度 | 低 | 中 |
| 适合场景 | 高压缩比（数据小） | 低压缩比（数据大） |

压缩后数据通常较小，ZMQ 足够。若压缩比低（< 2x），切换 NcclTransport。

### 6.3 Consumer 侧注入时机与内存压力

`get_num_new_matched_tokens` 返回 `True` → 请求在 `WAITING_FOR_REMOTE_KVS` 等待接收完成后才进入调度。接收完成时，解压后 KV 缓存在 GPU 内存直到 forward pass 注入。

缓冲内存估算：`num_layers × 2 × num_tokens × num_kv_heads × head_size × dtype_bytes`

对 Llama-3-8B（32层，8 KV heads，128 head_size，fp16）、4096 tokens：
`32 × 2 × 4096 × 8 × 128 × 2 bytes ≈ 4.3 GB`

并发接收请求数需设上限，避免 OOM。

---

## 七、已知限制

### 7.1 Chunked Prefill

若 Prefill 实例开启 chunked prefill，一个 request 的 KV 跨多个 scheduler step 产生。`build_connector_meta` 需要跨步积累 block_ids，类似 `P2pNcclConnector.chunked_prefill` dict 的处理。

**当前处理**：Phase 1 先禁用 chunked prefill（`--max-num-batched-tokens` 足够大）。

### 7.2 Tensor Parallelism（TP > 1）

TP > 1 时每个 TP rank 的 worker 有独立的 Connector WORKER 实例，各自持有各自 shard 的 KV。需要：
- 每个 TP rank 建立到对端对应 TP rank 的 transport 连接
- `kv_port + tp_rank` 区分端口

当前设计假设 TP = 1，多 TP 支持后续扩展。

### 7.3 Prefix Cache 去重

若 Decode 侧开启了 prefix cache，本地已命中部分 prompt KV 时，`get_num_new_matched_tokens` 应只请求未命中的部分（而非全部 prompt），避免冗余传输。

后续可在 `get_num_new_matched_tokens` 中查询本地 `KVCacheManager.get_computed_blocks`，返回 `(prompt_len - 1 - local_cached, True)`。

### 7.4 压缩错误回退

当前无回退到未压缩重传的机制。Phase 2 实施时加入：
- 发送端在 `KVPacket` header 中附加 checksum
- 接收端校验失败后通过 control channel 请求重传

---

## 八、实现顺序

**Phase 1 — 框架正确性验证（无压缩，identity pass-through）**
1. 实现 `CompressedKVConnector`（compression_manager 用 identity：原样返回）
2. 实现 `ZmqTransport`
3. 实现 `utils.py`（extract / inject / slot_mapping）
4. 与 `P2pNcclConnector` 的输出做数值对比，验证注入正确性

**Phase 2 — 接入真实压缩**
5. 接入 `CompressionManager.compress_all_layers`（quantizer + lz4 codec）
6. 通过 `kv_connector_extra_config` 暴露压缩策略参数
7. 测量压缩比、解压速度，关注 perplexity 变化

**Phase 3 — 性能优化**
8. `save_kv_layer` 改为 CUDA stream 异步提取（与后续层 compute 并行）
9. 多 request 并发压缩
10. 按当前网络带宽动态选择压缩策略（`CompressionManager` 已有 adaptive 框架）
11. 支持 Chunked Prefill 和 TP > 1
