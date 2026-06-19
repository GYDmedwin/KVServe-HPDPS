# KVServe

[![vLLM V1](https://img.shields.io/badge/vLLM-V1-4b5563)](https://github.com/vllm-project/vllm)
![KV compression](https://img.shields.io/badge/KV%20compression-9x-2563eb)
![PD communication](https://img.shields.io/badge/PD%20comm-8x%20faster-16a34a)
![Latency](https://img.shields.io/badge/E2E%20latency-7.5x%20lower-9333ea)
![Accuracy](https://img.shields.io/badge/accuracy-preserved-15803d)
![License](https://img.shields.io/badge/license-Apache--2.0-64748b)

**Service-aware KV-cache compression for bandwidth-efficient disaggregated LLM serving.**

KVServe is a **vLLM KV connector extension** that reduces KV-cache traffic in disaggregated prefill/decode serving. It plugs into vLLM without forking the runtime, keeps scheduling and KV block management inside vLLM, and only handles KV transfer plus optional compression.

## 🔥 News

- **[2026-05-13]** KVServe is now on arXiv! Read the paper here: [arXiv:2605.13734](https://arxiv.org/abs/2605.13734). 🚀
- **[2026-05-12]** KVServe v1 code has been released, with plug-and-play integration with vLLM. ⚡
- **[2026-05-11]** KVServe has been accepted by **ACM SIGCOMM 2026**! 🎉

KVServe is built around a modular KV compression abstraction:

```text
Raw KV Cache  →  Transform  →  Quantizer  →  Codec  →  Compressed KV
```

This makes KV compression **configurable, extensible, and service-aware**: KVServe can choose different compression profiles based on bandwidth, SLO, and quality budget.

```text
KV COMPRESSION          █████████  up to 10x
PD COMM TIME            ████████   9x lower
END-TO-END LATENCY      ███████▌   8.5x lower
ACCURACY                █████████  preserved
```

## Where KVServe Helps

KVServe is most useful when KV transfer is on the critical path.

| Scenario | Bandwidth | Communication Path | KVServe Benefit |
|---|---:|---|---|
| Distributed edge system | ≤10 Gbps | Robot → robot over wireless | Highest |
| Remote KV-cache storage | 5–25 Gbps | Remote storage → HBM over Ethernet | High |
| Cross datacenter / cluster PD serving | 10–100 Gbps | Prefill cluster → decode cluster over Ethernet | High |
| Cross node PD serving | 50–200+ Gbps | Prefill node → decode node over RoCE / InfiniBand | Medium |

When bandwidth is tight, KVServe uses stronger compression to reduce transfer time.
When bandwidth is abundant, KVServe can select lighter compression or disable compression to avoid unnecessary overhead.

## Key Features

- **Plug into vLLM**: use `kv_connector_module_path`, no vLLM fork required.
- **Compress only KV traffic**: vLLM keeps native scheduling and KV block management.
- **Service-aware selection**: choose compression profiles based on bandwidth, SLO, and quality budget.
- **Modular compression pipeline**: configure Transform / Quantizer / Codec independently.
- **Support PD + TP**: validated for two-engine PD and homogeneous tensor parallelism.

## Installation

Use an environment that already has a compatible vLLM installation, then install KVServe from the repository root:

```bash
cd /path/to/KVServe
pip install -e .
pip install -r requirements.txt
```

If you do not install in editable mode, set `PYTHONPATH` before running tests:

```bash
cd /path/to/KVServe
export PYTHONPATH="$(pwd)"
```

## External vLLM Connector

KVServe can be used as an out-of-tree vLLM V1 connector. Install this package in the same Python environment as vLLM, then configure vLLM with:

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

See `examples/external_connector_config.py` for a copyable producer/consumer configuration helper.

For homogeneous TP, use the same tensor parallel size on prefill and decode. KVServe opens one rank-to-rank channel per TP rank:

```text
kv_port + tp_rank
```

`transfer_id` is the stable wire key for one logical request. In production, it should be generated once by the router or request admission layer, then passed to both prefill and decode through `SamplingParams.extra_args`. The connector cannot safely invent matching IDs independently on two different engines.

## Choose a Compression Mode

KVServe supports three compression modes. The recommended path is controller
mode: the online selector chooses the **optimal compression profile** for the current
bandwidth, SLO, and quality budget. 

For quick experiments, use the built-in
**default mode**. For research or deployment-specific tuning, pass a custom
pipeline.

```python
kv_connector_extra_config={"compression": "default"}
```

Default mode runs the fused **TileLang `compress_v3`** pipeline (Hadamard
transform + quantize fused in one GPU kernel, then codec). The fused kernels are
fast enough that compressed KV transfer is a net win on fast InfiniBand as well
as on slow links. If TileLang is not installed, default mode falls back to the
quantizer + codec pipeline automatically.

**Controller mode** uses a profile library:

```python
kv_connector_extra_config={
    "compression": {
        "mode": "controller",
        "library_path": "/path/to/profiles.json",
        "service_config": {
            "slo_ms": 200.0,
            "accuracy_requirement": 0.92,
        },
    }
}
```

**Custom mode** directly defines the ordered compression pipeline. The connector
still receives vLLM-managed KV blocks; the pipeline only transforms the
extracted KV payload before transport.

```python
kv_connector_extra_config={
    "compression": {
        "enabled": True,
        "pipeline": ["quantizer", "codec"],
        "quantizer_config": {
            "model_name": "Qwen2.5-7B-Instruct",
            "hybrid_ratio": 0.3,
            "high_key_max_value": 16,
            "high_value_max_value": 16,
            "low_key_max_value": 12,
            "low_value_max_value": 12,
            "axis_key": "channel",
            "axis_value": "token",
            "split_type": "head",
        },
        "codec_config": {
            "codec_type": "nvcomp",
            "nvcomp_algorithm": "ANS",
            "data_type": "|u1",
        },
        "min_compress_size": 1024,
    }
}
```

Add a transform stage when your profile needs one:

```python
"compression": {
    "enabled": True,
    "pipeline": ["transformer", "quantizer", "codec"],
    "transformer_config": {"transform_type": "hadamard", "seed": 0x3333},
    "quantizer_config": {...},
    "codec_config": {...},
}
```

Built-in stages:

- `transformer`: `KVServeTransformer`, currently Hadamard transform.
- `quantizer`: `KVServeQuantizer`, hybrid head/layer precision quantization.
- `codec`: `KVServeCodec`, currently nvCOMP-backed lossless payload coding.

### Build Your Own Compression Component

KVServe allows new transformer, quantizer, or codec implementations without
touching vLLM scheduling or KV block allocation. Custom components should follow
the same KV shape contract and be selected inside
`KVCompressionAdapter._build_manager()`.

See `examples/README.md` for the minimal custom quantizer example and wiring
snippet.

## Testing

All commands below should be run from the repository root. Override the model path with `--model /path/to/model` when the default path is not available.

Baseline PD separation:

```bash
python tests/test_pd_prefill_decode.py --model /path/to/model
```

KVServe test without compression:

```bash
python tests/test_kvserve.py --mode none --model /path/to/model
python tests/test_kvserve.py --mode none --model /path/to/model \
  --lmeval-task wikitext --num-requests 20
```

KVServe test with compression:

```bash
python tests/test_kvserve.py --mode custom --model /path/to/model
python tests/test_kvserve.py --mode default --model /path/to/model
```

Controller mode with a profile library:

```bash
python tests/test_kvserve.py --mode controller --model /path/to/model \
  --library-path /path/to/profiles.json
```

`--lmeval-task` requires `lm-eval` and a locally available dataset cache unless `--online` is passed.

## Citation

If you find KVServe useful for your research, please consider citing our paper:

```bibtex
@article{liu2026kvserve,
  title={KVServe: Service-Aware KV Cache Compression for Communication-Efficient Disaggregated LLM Serving},
  author={Liu, Zedong and Ma, Xinyang and Luo, Dejun and Zhao, Hairui and Lu, Bing and Huang, Wenjing and Gu, Yida and Liu, Xingchen and Wei, Zheng and Liu, Jinyang and others},
  journal={arXiv preprint arXiv:2605.13734},
  year={2026}
}
```

## Notes

- KVServe expects the PD orchestration layer to attach a stable `transfer_id` to each logical request. This is handled by the KVServe test; external integrations should do the same in their router or request admission layer.
- Homogeneous TP is supported when prefill and decode use the same TP size.
- The ZMQ/NCCL control plane should run on trusted network interfaces only.

## License

Apache-2.0
