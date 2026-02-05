# Benchmark Evaluation Guide

This guide explains how to run multi-benchmark evaluation on the 7 math test datasets:
- **AIME 2024** - American Invitational Mathematics Examination (2024)
- **AMC** - American Mathematics Competition
- **MATH** - Hendrycks MATH benchmark
- **MINERVA** - Minerva Math benchmark
- **OLYMPIAD_BENCH** - Olympiad-level math problems
- **AIME 2025** - American Invitational Mathematics Examination (2025)
- **AMC 2025** - American Mathematics Competition (2025)

Both approaches generate K responses per question and compute Pass@n for all n from 1 to K.

## Approach A: Manual vLLM Server

This approach uses `AgentExecutionEngine` with a manually launched vLLM server.

### Step 1: Launch vLLM Server

**Note:** The DeepScaleR training uses **TP=1** (no tensor parallelism) with **data parallelism** across 8 GPUs. Each GPU runs an independent model replica processing different requests.

**For multi-GPU with data parallelism (recommended, matches training):**
```bash
HYDRA_FULL_ERROR=1 NCCL_NET=Socket python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/model \
    --port 30000 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 \
    --dtype bfloat16
```
This creates 8 independent model replicas (one per GPU), matching the training configuration.

**For single GPU:**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/model \
    --port 30000 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16
```

**For larger models requiring tensor parallelism (e.g., 7B+ models that don't fit on single GPU):**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/model \
    --port 30000 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16
```

### Step 2: Run Generation

```bash
python examples/deepscaler/generate_benchmarks_openai.py \
    --model_path /path/to/your/model \
    --output_path ~/data/benchmark_results.parquet \
    --n_samples 16 \
    --seed 42 \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_tokens 32768 \
    --base_url http://localhost:30000/v1 \
    --n_parallel_agents 64
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | (required) | Model name/path (must match vLLM server) |
| `--output_path` | `~/data/benchmark_results_openai.parquet` | Output file path |
| `--n_samples` | 16 | Number of responses per question (K) |
| `--seed` | 42 | Random seed |
| `--temperature` | 0.6 | Sampling temperature (matches training) |
| `--top_p` | 0.95 | Top-p sampling parameter (matches training) |
| `--max_tokens` | 32768 | Max total tokens (prompt + response) |
| `--base_url` | `http://localhost:30000/v1` | vLLM server URL |
| `--n_parallel_agents` | 64 | Concurrent agent count |
| `--run_evaluation` | False | Run evaluation after generation |

**Note on defaults:** Temperature (0.6) and top_p (0.95) match the training validation config. The training uses n=8 samples during validation; n=16 is used here to compute pass@16.

### Step 3: Run Evaluation

```bash
python examples/deepscaler/evaluate_benchmarks.py \
    --input_path ~/data/benchmark_results.parquet \
    --output_path ~/data/benchmark_results_eval.json
```

---

## Approach B: AgentTrainer (Auto-launches vLLM)

This approach uses `AgentTrainer` which automatically launches vLLM through Ray.

### Step 1: Run Generation + Evaluation

```bash
python examples/deepscaler/generate_benchmarks_trainer.py \
    actor_rollout_ref.model.path=/path/to/your/model \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.max_response_length=24576 \
    +benchmark.output_path=~/data/benchmark_results_trainer.parquet \
    +benchmark.save_val_results=True
```

**Common configuration overrides:**
| Config | Default | Description |
|--------|---------|-------------|
| `actor_rollout_ref.model.path` | (required) | Model path |
| `actor_rollout_ref.rollout.val_kwargs.n` | 1 | Samples per question (K) |
| `+benchmark.output_path` | `~/data/benchmark_results_trainer.parquet` | Output path |
| `+benchmark.save_val_results` | True | Enable saving results |
| `actor_rollout_ref.rollout.val_kwargs.temperature` | 0.6 | Sampling temperature |
| `actor_rollout_ref.rollout.val_kwargs.top_p` | 0.95 | Top-p sampling parameter |
| `actor_rollout_ref.rollout.val_kwargs.max_response_length` | 24576 | Max response length |

### Step 2: Run Evaluation (if needed separately)

```bash
python examples/deepscaler/evaluate_benchmarks.py \
    --input_path ~/data/benchmark_results_trainer.parquet
```

---

## Output Files

### Parquet File (Generation Output)

Contains one row per unique problem with columns:
- `data_source`: Dataset name (AIME, AMC, MATH, MINERVA, OLYMPIAD_BENCH)
- `problem`: The problem text
- `ground_truth`: Expected answer
- `responses`: List of K responses
- `rewards`: List of K reward scores (0 or 1)

### JSON File (Evaluation Output)

Contains pass@k metrics for each dataset:
```json
{
  "AIME": {
    "num_problems": 30,
    "num_samples": 16,
    "mean_reward": 0.45,
    "pass_at_k": [0.35, 0.48, 0.62, 0.71, ...]
  },
  ...
  "_aggregate": {
    "num_problems": 1000,
    "pass_at_k": [0.42, 0.55, 0.68, ...]
  }
}
```

### Summary Table

The evaluation script prints a summary table:
```
================================================================================
BENCHMARK EVALUATION RESULTS
================================================================================
Dataset              #Prob   Pass@1   Pass@2   Pass@4   Pass@8  Pass@16
--------------------------------------------------------------------------------
AIME                    30    35.0%    48.0%    62.0%    71.0%    85.0%
AMC                     50    45.0%    58.0%    70.0%    82.0%    90.0%
MATH                   500    55.0%    65.0%    75.0%    83.0%    91.0%
MINERVA                200    40.0%    52.0%    64.0%    74.0%    84.0%
OLYMPIAD_BENCH         220    25.0%    38.0%    52.0%    65.0%    78.0%
--------------------------------------------------------------------------------
AGGREGATE             1000    42.0%    55.0%    68.0%    78.0%    87.0%
================================================================================
```

---

## Comparing Approaches

| Feature | Approach A (OpenAI) | Approach B (Trainer) |
|---------|---------------------|----------------------|
| vLLM Management | Manual | Automatic |
| Multi-GPU Support | Yes (`--data-parallel-size`) | Yes (via Ray) |
| Setup Complexity | Medium | Low |
| Flexibility | High | Medium |
| Use Case | Production eval, custom servers | Quick evaluation |
| Ray Required | No | Yes |

**When to use Approach A:**
- You need fine-grained control over the vLLM server (TP, DP settings)
- You want to reuse an existing vLLM server
- You're running in a production environment

**When to use Approach B:**
- You want a quick, all-in-one solution
- You're doing development/experimentation
- You prefer minimal setup

---

## Troubleshooting

### Approach A Issues

**Connection refused:**
```
Check that vLLM server is running:
  curl http://localhost:30000/v1/models
```

**OOM errors:**
```
Reduce parallel agents or max_tokens:
  --n_parallel_agents 32
  --max_tokens 16384
```

**Using data parallelism:**
```
# 8 GPUs with data parallelism (matches training setup)
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/model \
    --data-parallel-size 8 \
    --tensor-parallel-size 1 \
    ...

# Verify all GPUs are being used:
nvidia-smi
```

### Approach B Issues

**Ray initialization errors:**
```
Ensure Ray is installed and GPUs are visible:
  ray start --head
  python -c "import ray; ray.init(); print(ray.cluster_resources())"
```

**Config errors:**
```
Check Hydra syntax:
  - Use = for existing keys: actor_rollout_ref.model.path=/path
  - Use += for new keys: +benchmark.output_path=/path
```
