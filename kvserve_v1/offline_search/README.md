# Offline Search

This directory contains an offline parameter-search workflow for KV-cache
compression. The main entry point is
`evaluation/param_search/test_search.py`.

The workflow does two things:

- Profiles the compression ratio of candidate KV compression configs.
- Uses accuracy-constrained Bayesian search to find configs with high
  compression ratio while keeping task accuracy within the configured tolerance.

## What Is Included

- `duo_config/`: per-model head score files used by the custom cache policy.
- `src/cache/`: cache and quantization utilities used by the offline evaluator.
- `src/models/`: local model patches used by the `transformers` path.
- `evaluation/param_search/`: the main search script and evaluators.
- `evaluation/compression_ratio/`: compression-ratio utilities.
- `evaluation/lm_eval/`: lm-eval integration for accuracy checks.
- `evaluation/component_speed/`: optional component-level speed scripts.

## Requirements

Recommended environment:

- `transformers==4.50.0`
- `lm-eval`
- `numpy`, `pandas`, `scipy`, `scikit-learn`, and `datasets`
- PyTorch with CUDA support
- `cupy` and NVIDIA `nvcomp` for the current compression-ratio evaluator
- Local model checkpoints
- A matching score CSV under `duo_config/`, for example
  `duo_config/Qwen2.5-7B-Instruct_scores.csv`

## Configure

Edit the configuration block at the top of
`evaluation/param_search/test_search.py`.

The most important fields are:

- `BASE_MODEL_PATH`: directory containing local model checkpoints.
- `MODEL_NAME`: model directory name under `BASE_MODEL_PATH`.
- `BASE_CONFIG_PATH`: directory containing the score CSV files.
- `TASK_TO_SEARCH`: lm-eval tasks used for accuracy evaluation.
- `DATASET_LIMIT`: number of examples sampled per task.
- `ACC_TOLERANCE`: allowed accuracy degradation from the baseline.
- `MAX_ITER`: Bayesian search budget after cold start.
- `SEARCH_SPACE`: candidate compression parameters.

Supported score files currently include:

- `Llama-3.1-8B-Instruct_scores.csv`
- `Qwen2.5-7B-Instruct_scores.csv`
- `Qwen2.5-32B-Instruct_scores.csv`

## Run

Run the script from its directory so the default relative paths resolve
correctly:

```bash
cd kvserve_v1/offline_search/evaluation/param_search
python test_search.py
```

The script first builds or reuses a compression-ratio cache, then evaluates
candidate configs with lm-eval and writes feasible results.

## Outputs

Results are written under:

```text
evaluation/param_search/results/<MODEL_NAME>/
```

Typical files:

- `search_space.csv`: cached compression-ratio values.
- `tolerance_<ACC_TOLERANCE>_results.json`: feasible configs with accuracy and
  compression-ratio fields.

The CR cache can be reused as long as the model, task, compression
implementation, and search space stay compatible.
