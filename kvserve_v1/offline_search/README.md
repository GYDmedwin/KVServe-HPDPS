# Offline Search

This directory contains the offline parameter-search workflow for KV-cache compression. The current search implementation is centered on `evaluation/param_search/test_search.py`: it enumerates a user-defined compression search space, profiles compression ratio once per candidate, and then uses Bayesian optimization to find configurations that preserve accuracy while maximizing compression ratio.

The implementation currently runs through the Hugging Face `transformers` stack. It has not yet been adapted to the `vllm` path used by `kvserve_v1`.

## Supported Models

The current offline search setup is intended for the following local model names:

- `Llama-3.1-8B-Instruct`
- `Qwen2.5-7B-Instruct`
- `Qwen2.5-32B-Instruct`

Each model must have a matching score file under `duo_config`, for example:

```text
duo_config/Qwen2.5-7B-Instruct_scores.csv
```

## Directory Layout

```text
offline_search/
|-- duo_config/                         # Per-model score files used by the custom cache policy
|-- src/cache/                          # Cache and quantization utilities
|-- src/models/                         # Local model implementations
`-- evaluation/
    |-- param_search/
    |   |-- test_search.py              # Main offline search entry point
    |   |-- acc_evaluator.py            # Accuracy evaluator based on lm-eval
    |   |-- cr_evaluator.py             # Compression-ratio evaluator
    |   `-- results/                    # Cached CR values and final search results
    |-- lm_eval/                        # lm-eval integration
    |-- compression_ratio/              # Compression utilities and wrappers
    `-- component_speed/                # Component-level speed benchmarks
```

## Requirements

Recommended environment:

- `transformers==4.50.0`
- `lm-eval`, used by `acc_evaluator.py` for accuracy evaluation
- `numpy`, `pandas`, `scipy`, `scikit-learn`, and `datasets`
- PyTorch with CUDA support
- `cupy` and NVIDIA `nvcomp` for the current compression-ratio path
- Access to the target model checkpoints under `BASE_MODEL_PATH`
- Score CSV files under `BASE_CONFIG_PATH`

The compression-ratio evaluator uses the custom cache implementation and the default codec path configured in `cr_evaluator.py`.

## Configuration

Before running the search, edit the configuration block at the top of `evaluation/param_search/test_search.py`.

```python
# ================= Configuration =================

# Target constraints and search budget.
BASELINE_ACC = 100
ACC_TOLERANCE = 3
TARGET_ACC_THRESHOLD = BASELINE_ACC - ACC_TOLERANCE
PRUNING_EPSILON = 0.2
MAX_ITER = 5
EXPLORATION_WEIGHT = 1
SEED = 42
WHETHER_TO_EXPLORE = True

# Parameter search space.
SEARCH_SPACE = {
    "transform_type": ["hadamard"],
    "heads_selection": [0.3, 0.5, 0.7, 0.9],
    "high_key_max_value": [12, 10, 8],
    "high_value_max_value": [12, 10, 8],
    "low_key_max_value": [6, 4],
    "low_value_max_value": [6, 4],
    "axis_key_options": [(2,)],
    "axis_value_options": [(1, 3)],
}

MODEL_NAME = "Qwen2.5-7B-Instruct"
TASK_TO_SEARCH = ["longbench_2wikimqa"]
DATASET_LIMIT = 5
BATCH_SIZE = 2
CACHE_CSV_PATH = "search_space.csv"
FINAL_JSON_PATH = f"tolerance_{ACC_TOLERANCE}_results.json"
BASE_MODEL_PATH = "/root/data/models"
BASE_CONFIG_PATH = "../../duo_config"
```

Important fields:

- `BASE_MODEL_PATH`: directory containing local model checkpoints.
- `BASE_CONFIG_PATH`: directory containing per-model score CSV files.
- `MODEL_NAME`: model directory name under `BASE_MODEL_PATH`.
- `TASK_TO_SEARCH`: lm-eval task list used for accuracy evaluation.
- `DATASET_LIMIT`: number of examples sampled per task.
- `BATCH_SIZE`: evaluation batch size.
- `ACC_TOLERANCE`: allowed accuracy degradation from `BASELINE_ACC`.
- `MAX_ITER`: maximum number of Bayesian optimization iterations after cold start.
- `SEARCH_SPACE`: candidate parameter grid for compression settings.

## Running the Search

Run the search from the `kvserve_v1` directory:

```bash
python offline_search/evaluation/param_search/test_search.py
```

The script performs two main phases:

1. Compression-ratio profiling

   The script enumerates all valid configurations from `SEARCH_SPACE`, evaluates compression ratio through `CompressionEvaluator`, and writes reusable CR values to:

   ```text
   offline_search/evaluation/param_search/results/<MODEL_NAME>/search_space.csv
   ```

   Compression ratio is expensive but deterministic for the same model, task, and search space. Once this CSV is generated, later runs reuse the cached CR values and only profile newly added configurations.

2. Accuracy-constrained Bayesian search

   The script initializes `AccuracyEvaluator`, evaluates a small cold-start set, then fits a Gaussian-process surrogate model over the observed accuracies. It proposes new candidates by combining:

   - candidate compression ratio
   - predicted probability of meeting the target accuracy threshold
   - an optional uncertainty-based exploration bonus

   Feasible configurations are saved to:

   ```text
   offline_search/evaluation/param_search/results/<MODEL_NAME>/tolerance_<ACC_TOLERANCE>_results.json
   ```

## Search Logic

The search target is:

```text
maximize compression ratio
subject to accuracy >= BASELINE_ACC - ACC_TOLERANCE
```

The workflow is:

1. Generate the Cartesian product from `SEARCH_SPACE`.
2. Filter invalid candidates, such as inconsistent high/low quantization bounds.
3. Load cached compression-ratio values if available.
4. Evaluate CR for missing candidates and append them to the cache CSV.
5. One-hot encode categorical or tuple-valued parameters for the Gaussian-process model.
6. Run a cold start using high-, median-, and low-CR candidates.
7. Iteratively fit the GP model and propose the next candidate.
8. Evaluate actual accuracy with lm-eval.
9. Record feasible configurations and apply pruning rules.
10. Stop when `MAX_ITER` is reached, no candidates remain, or the early-stopping condition is triggered.

The pruning logic uses two assumptions:

- If a configuration is feasible, more conservative configurations with lower CR may be dominated.
- If a configuration is clearly infeasible, sufficiently more aggressive configurations may be skipped when they share the same isolation columns.

The current isolation columns are `axis_key` and `axis_value`.

## Current Compression Components

The current search configuration supports:

- Transformer mode: `none` and `hadamard`
- Quantizer: `mixhq`
- Codec: `nvcomp-ans` by default

The default codec is configured in `evaluation/param_search/cr_evaluator.py` through:

```python
nvcomp_wrapper = NVCompWrapper("ANS", data_type="|u1")
```

At the moment, the search space is mainly exposed through transformer and quantization-related parameters in `test_search.py`. Full joint search over transformer, quantizer, and codec components is planned but not yet implemented.

## Outputs

For each model, results are written under:

```text
evaluation/param_search/results/<MODEL_NAME>/
```

Typical files:

- `search_space.csv`: cached compression-ratio values for all profiled candidates.
- `tolerance_<ACC_TOLERANCE>_results.json`: feasible search results with accuracy and configuration fields.

The CR cache can be reused across runs as long as the model, task, compression implementation, and search-space definitions remain compatible.

## Limitations

- The current implementation is based on `transformers`; it is not yet adapted to the `kvserve_v1` `vllm` runtime.
- Accuracy evaluation depends on `lm-eval` and the task definitions available in that package.
- The default compression-ratio path uses `nvcomp-ans`; changing codecs currently requires editing `cr_evaluator.py`.
- The search currently covers a limited set of compression parameters instead of a full component-level search across transformer, quantizer, and codec choices.
- Model checkpoints and score CSV files must be prepared locally before running the script.

## Roadmap

Planned improvements:

- [ ] Adapt the offline search workflow to the `kvserve_v1` `vllm` implementation.
- [ ] Further clean up the code structure and documentation.
- [ ] Support full search across transformer, quantizer, and codec components.
