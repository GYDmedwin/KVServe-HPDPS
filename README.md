# KVServe

We present *KVServe*, the first service-aware and adaptive KV communication compression framework for disaggregated LLM serving: KVServe (1) unifies KV compression into a modular strategy space with new components and cross-method recomposition; (2) introduces Bayesian Profiling Engine that efficiently searches this space and distills a 3D Pareto candidate set, reducing `50x` offline search overhead; and (3) deploys a Service-Aware Online Controller that combines an analytical latency model with a lightweight bandit to select profiles under constraints and correct offline-to-online mismatch.

KVServe works as an upper-layer extension on top of vLLM. It is plug-and-play (`kv_connector_module_path` based), and can be adapted to new vLLM versions with minimal integration changes.

## Installation

From the repository root:

```bash
cd /data/lzd/kvserve_v1
pip install -r requirements.txt
```

Use any Python environment that matches your vLLM / CUDA stack. This repository does not prescribe how you create that environment.

## Run Guide

From the repository root:

```bash
cd /data/lzd/kvserve_v1
```

These scripts live under `tests/`. When you run `python tests/...py`, Python puts `tests/` first on `sys.path`, so the top-level package `kvserve_v1` is **not** on the import path unless you add the repo root. You do **not** need `pip install -e .`; set `PYTHONPATH` to the repo root (`.` is enough after `cd`):

```bash
export PYTHONPATH=/data/lzd/kvserve_v1
# or, equivalently, after cd:
export PYTHONPATH=.
```

### 1) Validate PD separation (baseline)

```bash
PYTHONPATH=. python tests/test_pd_prefill_decode.py
```

Override the default model path if needed, for example:

```bash
PYTHONPATH=. python tests/test_pd_prefill_decode.py --model /path/to/your/model
```

### 2) Validate PD + compression

```bash
PYTHONPATH=. python tests/test_pd_with_compression.py --mode custom
PYTHONPATH=. python tests/test_pd_with_compression.py --mode default
PYTHONPATH=. python tests/test_pd_with_compression.py --mode controller --library-path /path/to/profiles.json
```

### 3) Run dataset-enabled simulation test

```bash
PYTHONPATH=. python tests/test_simulator.py --mode none
PYTHONPATH=. python tests/test_simulator.py --mode custom
PYTHONPATH=. python tests/test_simulator.py --lmeval-task wikitext --num-requests 20
```

`--lmeval-task` requires `lm-eval` to be installed (see `requirements.txt` optional line).

## Difference Between the Three Test Files

- `tests/test_pd_prefill_decode.py`
  - Verifies PD separation end-to-end (prefill/producer + decode/consumer).
  - Focuses on communication and functional correctness.

- `tests/test_pd_with_compression.py`
  - Verifies PD separation with KV compression enabled.
  - Focuses on compression mode behavior (`custom/default/controller`).

- `tests/test_simulator.py`
  - Adds dataset-driven testing and richer experiment controls.
  - Supports lm-eval prompts, CSV export, and optional KV dump mode.
