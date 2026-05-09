# KVServe

[![vLLM V1](https://img.shields.io/badge/vLLM-V1-4b5563)](https://github.com/vllm-project/vllm)
![KV compression](https://img.shields.io/badge/KV%20compression-9x-2563eb)
![PD communication](https://img.shields.io/badge/PD%20comm-8x%20faster-16a34a)
![Latency](https://img.shields.io/badge/E2E%20latency-7.5x%20lower-9333ea)
![Accuracy](https://img.shields.io/badge/accuracy-preserved-15803d)
![License](https://img.shields.io/badge/license-Apache--2.0-64748b)

KVServe is a vLLM V1 KV connector extension for disaggregated prefill/decode
serving with optional KV communication compression. It keeps scheduling and KV
block management inside vLLM and only handles KV transfer plus compression.

```text
KV COMPRESSION          █████████  9x
PD COMM TIME            ████████   8x lower
END-TO-END LATENCY      ███████▌   7.5x lower
ACCURACY                █████████  preserved
```

- **Plug into vLLM**: use `kv_connector_module_path`, no vLLM fork required.
- **Compress only KV traffic**: vLLM keeps native scheduling and KV block management.
- **Support PD + TP**: validated for two-engine PD and homogeneous TP.

## Installation

Use an environment that already has a compatible vLLM installation, then install
KVServe from the repository root:

```bash
cd /path/to/KVServe
pip install -e .
pip install -r requirements.txt
```

If you do not install editable mode, set `PYTHONPATH` before running tests:

```bash
cd /path/to/KVServe
export PYTHONPATH="$(pwd)"
```

## External vLLM Connector

KVServe can be used as an out-of-tree vLLM V1 connector. Install this package in
the same Python environment as vLLM, then configure vLLM with:

```python
KVTransferConfig(
    kv_connector="CompressedKVConnector",
    kv_connector_module_path="kvserve_v1.connector.compressed_kv_connector",
    kv_role="kv_producer",  # or "kv_consumer"
    kv_rank=0,              # 0 for producer, 1 for consumer
    kv_parallel_size=2,
    kv_ip="decode-node-ip",
    kv_port=25010,
    kv_connector_extra_config={"compression": None},
)
```

See `examples/external_connector_config.py` for a copyable producer/consumer
configuration helper.

For homogeneous TP, use the same tensor parallel size on prefill and decode.
KVServe opens one rank-to-rank channel per TP rank: `kv_port + tp_rank`.

`transfer_id` is the stable wire key for one logical request. In production it
should be generated once by the router or request admission layer, then passed
to both prefill and decode through `SamplingParams.extra_args`. The connector
cannot safely invent matching IDs independently on two different engines.

## Testing

All commands below should be run from the repository root. Override the model
path with `--model /path/to/model` when the default path is not available.

Baseline PD separation:

```bash
python tests/test_pd_prefill_decode.py --model /path/to/model
```

Simulator without compression:

```bash
python tests/test_simulator.py --mode none --model /path/to/model
python tests/test_simulator.py --mode none --model /path/to/model \
  --lmeval-task wikitext --num-requests 20
```

Simulator with compression:

```bash
python tests/test_simulator.py --mode custom --model /path/to/model
python tests/test_simulator.py --mode default --model /path/to/model
```

Controller mode requires a profile library:

```bash
python tests/test_simulator.py --mode controller --model /path/to/model \
  --library-path /path/to/profiles.json
```

`--lmeval-task` requires `lm-eval` and a locally available dataset cache unless
`--online` is passed.

## Notes

- KVServe expects the PD orchestration layer to attach a stable `transfer_id`
  to each logical request. This is handled by the test simulator; external
  integrations should do the same in their router or request admission layer.
- The current connector validates the standard two-engine PD path with one
  producer and one consumer instance. Homogeneous TP is supported when prefill
  and decode use the same TP size.
- The ZMQ/NCCL control plane should run on trusted network interfaces only.

## License

Apache-2.0.
