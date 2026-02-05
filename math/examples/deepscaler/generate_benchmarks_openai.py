"""
Approach A: Generate benchmark responses using AgentExecutionEngine with a manually launched vLLM server.

This script runs all 5 math benchmarks (AIME, AMC, MATH, MINERVA, OLYMPIAD_BENCH)
and generates K responses per question for pass@k evaluation.

Prerequisites:
    1. Launch a vLLM server manually:
       python -m vllm.entrypoints.openai.api_server \
           --model /path/to/your/model \
           --port 30000 \
           --tensor-parallel-size 1 \
           --max-model-len 32768

    2. Run this script:
       python generate_benchmarks_openai.py \
           --model_path /path/to/your/model \
           --output_path ~/data/results.parquet \
           --n_samples 16

Usage:
    python generate_benchmarks_openai.py --model_path MODEL_NAME --output_path OUTPUT.parquet
"""

import argparse
import asyncio
import hashlib
import json
import os
import pickle
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from rllm.agents.math_agent import MathAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_fn import math_reward_fn


# All math test benchmarks to evaluate (dataset names registered via DatasetRegistry)
ALL_TEST_DATASETS = [
    "aime2024",
    "amc",
    "math500",
    "minerva",
    "olympiad_bench",
    "aime2025",
    "amc2025",
]


def save_with_fallback(data, output_path: str, description: str = "data"):
    """
    Save data with multiple fallback methods. Will try parquet, then JSON, then pickle.
    GUARANTEES the data will be saved somewhere.

    Args:
        data: DataFrame or list of dicts to save
        output_path: Primary output path (will try .parquet, then .json, then .pkl)
        description: Description for logging

    Returns:
        str: Path where data was actually saved
    """
    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # If data is a list, convert to DataFrame
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data

    # Normalize ground_truth column if it exists
    if "ground_truth" in df.columns:
        df = df.copy()
        df["ground_truth"] = df["ground_truth"].apply(
            lambda x: json.dumps(x) if isinstance(x, (list, dict)) else str(x) if x is not None else ""
        )

    # Try 1: Parquet
    parquet_path = output_path.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"Successfully saved {description} to {parquet_path}")
        return str(parquet_path)
    except Exception as e:
        print(f"WARNING: Failed to save as parquet: {e}")
        print(f"Trying JSON fallback...")

    # Try 2: JSON
    json_path = output_path.with_suffix(".json")
    try:
        # Convert DataFrame to list of dicts for JSON
        records = df.to_dict(orient="records")
        with open(json_path, "w") as f:
            json.dump(records, f, indent=2, default=str)
        print(f"Successfully saved {description} to {json_path}")
        return str(json_path)
    except Exception as e:
        print(f"WARNING: Failed to save as JSON: {e}")
        print(f"Trying pickle fallback...")

    # Try 3: Pickle (should always work)
    pkl_path = output_path.with_suffix(".pkl")
    try:
        with open(pkl_path, "wb") as f:
            pickle.dump(df, f)
        print(f"Successfully saved {description} to {pkl_path}")
        return str(pkl_path)
    except Exception as e:
        print(f"CRITICAL: Failed to save as pickle: {e}")

        # Last resort: dump raw data to a text file
        txt_path = output_path.with_suffix(".txt")
        try:
            with open(txt_path, "w") as f:
                f.write(str(df.to_dict()))
            print(f"Emergency save to {txt_path}")
            return str(txt_path)
        except Exception as e2:
            print(f"FATAL: Could not save data anywhere: {e2}")
            raise RuntimeError(f"Failed to save data using any method. Last error: {e2}")


def load_all_benchmarks(aime_only: bool = False) -> list[dict]:
    """
    Load all test benchmarks and combine them into a single list.

    Each item must have required fields: 'data_source', 'question', 'ground_truth'.

    Args:
        aime_only: If True, only load AIME dataset for quick testing.

    Returns:
        List of task dictionaries with required fields
    """
    if aime_only:
        dataset_names = ["aime2024"]
    else:
        dataset_names = ALL_TEST_DATASETS

    all_tasks = []

    for dataset_name in dataset_names:
        print(f"Loading {dataset_name}...")
        dataset = DatasetRegistry.load_dataset(dataset_name, "test")
        if dataset is None:
            raise ValueError(f"Dataset '{dataset_name}' not found in registry. Run prepare_all_math_data.py first.")
        data = dataset.get_data()
        print(f"  Loaded {len(data)} problems")

        for i, item in enumerate(data):
            for field in ["data_source", "question", "ground_truth"]:
                if field not in item:
                    raise ValueError(f"Item {i} missing required field '{field}' in dataset '{dataset_name}': {item}")
            all_tasks.append(item)

    print(f"\nTotal: {len(all_tasks)} problems across {len(dataset_names)} benchmarks")
    return all_tasks


def process_results(results, n_samples: int) -> list[dict]:
    """
    Process raw trajectory results into grouped problem results.

    Args:
        results: List of trajectory objects from engine.execute_tasks
        n_samples: Expected number of samples per problem

    Returns:
        List of dicts with keys: data_source, problem, ground_truth, responses, rewards
    """
    # Handle None or empty results
    if results is None:
        results = []

    problem_results = defaultdict(lambda: {
        "data_source": None,
        "problem": None,
        "ground_truth": None,
        "responses": [],
        "rewards": [],
    })

    for trajectory in results:
        # Safely extract task from trajectory
        try:
            task = trajectory.task if hasattr(trajectory, "task") else None
        except Exception:
            task = None

        # Generate unique problem ID
        if isinstance(task, dict):
            problem_str = json.dumps(task, sort_keys=True)
            data_source = task.get("data_source", "unknown")
            problem_text = task.get("question", str(task))
            ground_truth = task.get("ground_truth", "")
        else:
            problem_str = str(task) if task is not None else "unknown_task"
            data_source = "unknown"
            problem_text = str(task) if task is not None else ""
            ground_truth = ""
        problem_hash = hashlib.md5(problem_str.encode()).hexdigest()

        # Store problem info (first time only)
        if problem_results[problem_hash]["problem"] is None:
            problem_results[problem_hash]["data_source"] = data_source
            problem_results[problem_hash]["problem"] = problem_text
            problem_results[problem_hash]["ground_truth"] = ground_truth

        # Collect response and reward (with explicit None handling)
        try:
            if hasattr(trajectory, "steps") and trajectory.steps and len(trajectory.steps) > 0:
                step = trajectory.steps[-1]
                response = getattr(step, "model_response", None)
                response = str(response) if response is not None else ""
            else:
                response = ""
        except Exception:
            response = ""

        try:
            reward = getattr(trajectory, "reward", None)
            reward = float(reward) if reward is not None else 0.0
        except Exception:
            reward = 0.0

        problem_results[problem_hash]["responses"].append(response)
        problem_results[problem_hash]["rewards"].append(reward)

    # Convert to list
    results_list = list(problem_results.values())

    # === VALIDATION: Catch serious data quality issues ===

    # Check if we got any results at all
    if len(results_list) == 0:
        raise ValueError("FATAL: No results produced. process_results returned empty list.")

    # Check for sample count mismatches
    mismatched_count = 0
    for i, item in enumerate(results_list):
        if len(item["rewards"]) != n_samples:
            print(f"Warning: Problem {i} has {len(item['rewards'])} samples, expected {n_samples}")
            mismatched_count += 1

    if mismatched_count > 0:
        print(f"WARNING: {mismatched_count}/{len(results_list)} problems have incorrect sample counts")

    # Check for excessive "unknown" data sources (indicates task extraction failures)
    unknown_count = sum(1 for item in results_list if item["data_source"] == "unknown")
    if unknown_count > 0:
        unknown_pct = unknown_count / len(results_list) * 100
        print(f"WARNING: {unknown_count}/{len(results_list)} ({unknown_pct:.1f}%) problems have 'unknown' data_source")
        if unknown_pct > 10:
            raise ValueError(f"FATAL: {unknown_pct:.1f}% of problems have 'unknown' data_source. Task extraction is failing.")

    # Check for excessive empty responses (indicates model response extraction failures)
    total_responses = sum(len(item["responses"]) for item in results_list)
    empty_responses = sum(sum(1 for r in item["responses"] if r == "") for item in results_list)
    if total_responses > 0:
        empty_pct = empty_responses / total_responses * 100
        print(f"Response statistics: {empty_responses}/{total_responses} ({empty_pct:.1f}%) empty responses")
        if empty_pct > 50:
            raise ValueError(f"FATAL: {empty_pct:.1f}% of responses are empty strings. Model response extraction is failing.")

    return results_list


def run_generation(
    model_path: str,
    output_path: str,
    n_samples: int = 16,
    seed: int = 42,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 32768,
    base_url: str = "http://localhost:30000/v1",
    n_parallel_agents: int = 64,
    max_prompt_length: int = 2048,
    aime_only: bool = False,
):
    """
    Run generation on all benchmarks.

    Args:
        model_path: Model name/path for the vLLM server
        output_path: Where to save results parquet file
        n_samples: Number of responses per question (K)
        seed: Random seed
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Max response length
        base_url: vLLM server URL
        n_parallel_agents: Number of concurrent agent trajectories
        max_prompt_length: Maximum prompt length
        aime_only: If True, only run on AIME dataset for quick testing
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a timestamp for backup files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_prefix = output_dir / f"backup_{timestamp}"

    # Load tokenizer
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load all benchmarks
    all_tasks = load_all_benchmarks(aime_only=aime_only)

    # Repeat tasks for n_samples
    print(f"\nRepeating each problem {n_samples} times for pass@k evaluation...")
    repeated_tasks = []
    for task in all_tasks:
        for _ in range(n_samples):
            repeated_tasks.append(task.copy())
    print(f"Total tasks after repeat: {len(repeated_tasks)}")

    # Create engine
    print(f"\nInitializing AgentExecutionEngine...")
    print(f"  Model: {model_path}")
    print(f"  Base URL: {base_url}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-p: {top_p}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Parallel agents: {n_parallel_agents}")

    sampling_params = {
        "temperature": temperature,
        "top_p": top_p,
        "model": model_path,
        "seed": seed,
    }

    engine = AgentExecutionEngine(
        agent_class=MathAgent,
        env_class=SingleTurnEnvironment,
        agent_args={},
        env_args={"reward_fn": math_reward_fn},
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": base_url,
            "api_key": "None",
        },
        max_response_length=max_tokens,
        max_prompt_length=max_prompt_length,
        n_parallel_agents=n_parallel_agents,
    )

    # Execute tasks
    print(f"\nRunning generation on {len(repeated_tasks)} tasks...")
    results = None
    try:
        results = asyncio.run(engine.execute_tasks(repeated_tasks))
        if results is None:
            raise ValueError("FATAL: engine.execute_tasks returned None")
        print(f"Generation complete. Got {len(results)} results.")

        # Validate we got approximately the expected number of results
        expected = len(repeated_tasks)
        actual = len(results)
        if actual == 0:
            raise ValueError("FATAL: Generation produced 0 results")
        if actual < expected * 0.5:
            raise ValueError(f"FATAL: Got only {actual}/{expected} results ({actual/expected*100:.1f}%). Too many failures.")
        if actual != expected:
            print(f"WARNING: Expected {expected} results, got {actual} ({actual/expected*100:.1f}%)")

    except Exception as e:
        print(f"ERROR during generation: {e}")
        traceback.print_exc()
        if results is None:
            raise
        print(f"Partial results available: {len(results)} trajectories")

    # IMMEDIATELY save raw results before any processing
    print("\n" + "=" * 60)
    print("SAVING RAW RESULTS (backup)")
    print("=" * 60)
    try:
        raw_backup_path = f"{backup_prefix}_raw_trajectories.pkl"
        with open(raw_backup_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Raw trajectories backed up to: {raw_backup_path}")
    except Exception as e:
        print(f"WARNING: Could not backup raw trajectories: {e}")

    # Process results
    print("\nGrouping results by problem...")
    try:
        results_list = process_results(results, n_samples)
    except Exception as e:
        print(f"ERROR during result processing: {e}")
        traceback.print_exc()
        # Try to save what we have
        print("Attempting emergency save of raw results...")
        if results:
            save_with_fallback(
                [{"raw": str(r)} for r in results],
                str(backup_prefix) + "_emergency",
                "emergency raw data"
            )
        raise

    # Save processed results with fallback
    print("\n" + "=" * 60)
    print("SAVING PROCESSED RESULTS")
    print("=" * 60)
    saved_path = save_with_fallback(results_list, str(output_path), "benchmark results")

    # Print summary statistics
    try:
        df = pd.DataFrame(results_list)
        print("\n" + "=" * 60)
        print("GENERATION SUMMARY")
        print("=" * 60)
        for data_source in df["data_source"].unique():
            subset = df[df["data_source"] == data_source]
            n_problems = len(subset)
            mean_reward = subset["rewards"].apply(lambda x: sum(r >= 1.0 for r in x) / len(x)).mean()
            print(f"{data_source:<20}: {n_problems:>4} problems, Pass@1 approx: {mean_reward*100:.1f}%")
        print("=" * 60)
        print(f"\nResults saved to: {saved_path}")
    except Exception as e:
        print(f"Could not print summary: {e}")
        print(f"But results were saved to: {saved_path}")

    return results_list


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark responses using AgentExecutionEngine with vLLM server"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Model name/path (must match the model running in vLLM server)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="~/data/benchmark_results_openai.parquet",
        help="Path to save results parquet file",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=16,
        help="Number of responses per question (K for pass@k)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32768,
        help="Max response length",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:30000/v1",
        help="vLLM server URL",
    )
    parser.add_argument(
        "--n_parallel_agents",
        type=int,
        default=64,
        help="Number of concurrent agent trajectories",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=2048,
        help="Maximum prompt length",
    )
    parser.add_argument(
        "--run_evaluation",
        action="store_true",
        help="Run evaluation after generation",
    )
    parser.add_argument(
        "--aime_only",
        action="store_true",
        help="Only run on AIME dataset for quick testing",
    )

    args = parser.parse_args()

    # Expand user home directory
    output_path = os.path.expanduser(args.output_path)

    # Run generation
    run_generation(
        model_path=args.model_path,
        output_path=output_path,
        n_samples=args.n_samples,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
        n_parallel_agents=args.n_parallel_agents,
        max_prompt_length=args.max_prompt_length,
        aime_only=args.aime_only,
    )

    # Optionally run evaluation
    if args.run_evaluation:
        print("\nRunning evaluation...")
        # Import from the same directory
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from evaluate_benchmarks import evaluate
        evaluate(output_path)


if __name__ == "__main__":
    main()
