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
