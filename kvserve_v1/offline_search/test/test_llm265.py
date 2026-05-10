import torch
import numpy as np
import PyNvVideoCodec as nvc
import cupy as cp

class LLM265Compressor:
    def __init__(self, device_id=0, width=1920, height=1080):
        """
        初始化 LLM.265 压缩器
        :param width: 视频帧宽度 (需根据 Tensor 大小计算，最好是 32 的倍数)
        :param height: 视频帧高度
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = 60
        
        # 1. 创建编码器
        # 注意：PyNvVideoCodec 的参数可能随版本更新，以下配置针对常见用例
        # format='NV12' 是 LLM.265 论文中使用的格式 (仅使用 Y 平面存储数据)
        # codec=nvc.CudaVideoCodec.HEVC 指定使用 H.265
        self.enc = nvc.CreateEncoder(
            width=self.width,
            height=self.height,
            fps=self.fps,
            fmt="NV12", 
            codec="hevc",
            preset="P1",  # P1 = 最快/最低延迟
            tuning_info="low_latency",
            # 设置 GOP 为 1，强制每一帧都是关键帧 (Intra-only)，禁用帧间预测
            gop=1,  
            idrperiod=1,
            gpuid=self.device_id,
            usecpuinputbuffer=False,
        )

        # 2. 创建解码器
        # 解码器通常不需要指定太多参数，它会自动解析比特流头信息
        self.dec = nvc.CreateDecoder(
            gpuid=self.device_id,
            codec=nvc.cudaVideoCodec.HEVC,
        )

    def quantize(self, tensor: torch.Tensor):
        """
        [LLM.265 核心步骤 1] 量化
        将 FP16 Tensor 映射到 uint8 [0, 255]
        """
        orig_shape = tensor.shape
        # 展平以便处理
        flat_tensor = tensor.flatten()
        
        # 计算统计量
        min_val = flat_tensor.min()
        max_val = flat_tensor.max()
        
        # 线性量化
        scale = (max_val - min_val) / 255.0
        # 防止除零
        scale = torch.where(scale == 0, torch.tensor(1.0, device=tensor.device, dtype=tensor.dtype), scale)
        
        quantized = (flat_tensor - min_val) / scale
        quantized = quantized.clamp(0, 255).round().to(torch.uint8)
        
        return quantized, scale, min_val, orig_shape

    def dequantize(self, quantized_tensor, scale, min_val, original_shape):
        """反量化回 FP16"""
        dequantized = quantized_tensor.to(torch.float16) * scale + min_val
        return dequantized.reshape(original_shape)

    def prepare_nv12_surface(self, quantized_tensor):
        """
        [LLM.265 核心步骤 2] 数据排布
        将 1D 的 uint8 tensor 填充到 NV12 Surface 的 Y 平面中。
        """
        # 确保数据在 CuPy 中 (零拷贝转换，如果 tensor 在 GPU 上)
        # PyNvVideoCodec 支持 __cuda_array_interface__
        
        total_pixels = self.width * self.height
        current_size = quantized_tensor.numel()
        
        # 1. Padding: 补零以匹配视频分辨率
        if current_size < total_pixels:
            padding = torch.zeros(total_pixels - current_size, dtype=torch.uint8, device=quantized_tensor.device)
            padded_data = torch.cat([quantized_tensor, padding])
        else:
            padded_data = quantized_tensor[:total_pixels] # 截断（理论上不应发生，需初始化正确的 width/height）

        # 2. Reshape 为 2D Y-plane
        y_plane = padded_data.reshape(self.height, self.width)
        
        # 3. 创建 UV 平面 (填充 128 代表无色度，或者 0)
        # NV12 中 UV 平面高度是 Y 的一半
        uv_plane = torch.full((self.height // 2, self.width), 128, dtype=torch.uint8, device=quantized_tensor.device)
        
        return y_plane, uv_plane

    def compress(self, tensor):
        """
        执行压缩：Quantize -> NV12 Layout -> H.265 Encode
        """
        # 1. 量化
        q_tensor, scale, min_val, orig_shape = self.quantize(tensor)
        
        # 2. 准备 YUV 数据
        y_tensor, uv_tensor = self.prepare_nv12_surface(q_tensor)
        
        # 3. 编码
        # PyNvVideoCodec 的 Encode 方法通常接受一个列表或对象，支持 CUDA Array Interface
        # 我们构建一个符合 NV12 结构的 RawFrame
        # 注意：具体 API 可能要求传入 list of planes [Y, UV]
        raw_frame = [y_tensor, uv_tensor]
        
        # 编码并获取 Packet
        # Encode() 返回 numpy 数组 (比特流) 或者 Packet 对象
        # 注意：第一帧可能需要发送一些 SPS/PPS 头信息，通常包含在比特流中
        encoded_packet = self.enc.Encode(raw_frame)
        
        # 如果是 Packet 对象，转换为 bytes
        if hasattr(encoded_packet, 'tobytes'):
            bitstream = encoded_packet.tobytes()
        elif isinstance(encoded_packet, np.ndarray):
            bitstream = encoded_packet.tobytes()
        else:
            # 某些版本可能直接返回 bytes 或 list of packets
            bitstream = bytes(encoded_packet) if encoded_packet is not None else b""

        metadata = {
            "scale": scale,
            "min_val": min_val,
            "shape": orig_shape,
            "valid_len": q_tensor.numel()
        }
        
        return bitstream, metadata

    def decompress(self, bitstream, metadata):
        """
        执行解压：H.265 Decode -> Extract Y -> Dequantize
        """
        if not bitstream:
            return None

        # 1. 解码
        # Decode 接受 bytes 或 numpy array
        # 返回 Surface 或 Frame 对象
        packet = np.frombuffer(bitstream, dtype=np.uint8)
        decoded_surface = self.dec.Decode(packet)
        
        if decoded_surface is None or (hasattr(decoded_surface, 'Empty') and decoded_surface.Empty()):
             # 有时需要 flush 或者数据不足一帧（虽然 intra-only 应该是一帧一个包）
             return None

        # 2. 提取数据
        # Decode 返回的对象通常支持 CUDA Array Interface 或有导出方法
        # 假设返回的是 list of planes [Y, UV]
        # 或者是一个 Surface 对象，我们需要提取 Y plane
        
        # 伪代码：适配不同的 PyNvVideoCodec 返回结构
        if isinstance(decoded_surface, list):
            y_plane_view = decoded_surface[0]
        else:
            # 如果是 Surface 对象，尝试导出
            # 注意：需查阅具体版本的 API 获取 Plane 0
            # 这里假设它支持切片或转换
            y_plane_view = decoded_surface.Plane(0) 

        # 将 CUDA memory view 转回 Torch Tensor
        # 利用 __cuda_array_interface__
        y_tensor = torch.as_tensor(y_plane_view, device=self.device_id)
        
        # 3. 恢复数据
        flat_data = y_tensor.flatten()
        valid_data = flat_data[:metadata["valid_len"]]
        
        # 4. 反量化
        recon_tensor = self.dequantize(
            valid_data, 
            metadata["scale"], 
            metadata["min_val"], 
            metadata["shape"]
        )
        
        return recon_tensor

# --- 使用示例 ---
if __name__ == "__main__":
    # 假设 GPU 0 可用
    device = torch.device('cuda:0')
    
    # 模拟一个 KV Cache (Batch=1, Heads=32, Seq=128, Dim=128)
    kv_cache = torch.randn(1, 32, 128, 128, dtype=torch.float16, device=device)
    
    # 计算所需分辨率
    num_elements = kv_cache.numel() # 524,288
    # 寻找合适的宽和高，例如 1024x512 = 524,288 (正好)
    width = 1024
    height = 512
    
    print(f"初始化 LLM.265 压缩器 (Resolution: {width}x{height}, Device: {device})...")
    compressor = LLM265Compressor(device_id=0, width=width, height=height)
    
    # 压缩
    print("正在压缩...")
    bitstream, meta = compressor.compress(kv_cache)
    
    # 统计
    original_size_kb = kv_cache.element_size() * kv_cache.numel() / 1024
    compressed_size_kb = len(bitstream) / 1024
    print(f"原始大小 (FP16): {original_size_kb:.2f} KB")
    print(f"压缩后大小 (H.265): {compressed_size_kb:.2f} KB")
    print(f"压缩比: {original_size_kb / compressed_size_kb:.2f}x")
    
    # 解压
    print("正在解压...")
    recon_kv = compressor.decompress(bitstream, meta)
    
    if recon_kv is not None:
        # 验证误差 (MSE)
        mse = torch.nn.functional.mse_loss(kv_cache.float(), recon_kv.float())
        print(f"重建 MSE: {mse.item():.6f}")
    else:
        print("解压失败或无输出帧")