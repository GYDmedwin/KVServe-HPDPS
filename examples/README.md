# External vLLM Connector Example

`external_connector_config.py` shows the stable import path and the minimal
`KVTransferConfig` needed to use KVServe as an out-of-tree vLLM V1 connector.

Key points:

- Install KVServe into the same Python environment as vLLM.
- Set `kv_connector_module_path` to
  `kvserve_v1.connector.compressed_kv_connector`.
- Use the decode node's reachable IP as `kv_ip`.
- For homogeneous TP, KVServe opens one rank-to-rank channel per TP rank:
  `kv_port + tp_rank`.
- The same logical request must carry the same `transfer_id` on prefill and
  decode. A router or request admission layer should generate it once and pass
  it to both sides through `SamplingParams.extra_args`.

## Custom Compression Component

Custom transformer, quantizer, and codec components operate on extracted
vLLM-managed KV blocks, not on vLLM scheduler state.

```text
one layer KV: [2, num_blocks, block_size, num_kv_heads, head_size]
all layers : [num_layers, 2, num_blocks, block_size, num_kv_heads, head_size]
```

A minimal custom quantizer:

```python
class MyQuantizer:
    def __init__(self, **kwargs):
        self.scale = kwargs.get("scale", 1.0)

    def update_params(self, **kwargs):
        self.scale = kwargs.get("scale", self.scale)

    def quantize(self, layer_id, tensor, **kwargs):
        # tensor: [2, num_blocks, block_size, num_kv_heads, head_size]
        q = tensor
        meta = {"original_dtype": str(tensor.dtype).replace("torch.", "")}
        return q, meta

    def dequantize(self, layer_id, tensor, meta, **kwargs):
        return tensor
```

Select it in `KVCompressionAdapter._build_manager()`:

```python
if "quantizer" in pipeline:
    if cfg_dict.get("quantizer_impl") == "my_quantizer":
        from my_package import MyQuantizer
        quantizer_cls = MyQuantizer
    else:
        from kvserve_v1.compression.quantizer.kvserve_quantizer import KVServeQuantizer
        quantizer_cls = KVServeQuantizer
```

Then use it in a compression profile:

```python
"compression": {
    "enabled": True,
    "pipeline": ["quantizer", "codec"],
    "quantizer_impl": "my_quantizer",
    "quantizer_config": {"scale": 1.0},
    "codec_config": {"codec_type": "nvcomp", "nvcomp_algorithm": "ANS"},
}
```
