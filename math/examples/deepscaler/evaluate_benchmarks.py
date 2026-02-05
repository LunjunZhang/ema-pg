"""
Shared evaluation script for benchmark results.

This script evaluates generation results from either:
- Approach A: generate_benchmarks_openai.py (manual vLLM server)
- Approach B: generate_benchmarks_trainer.py (AgentTrainer)

It computes Pass@n for all n from 1 to K and outputs both JSON results
and a summary table.

Usage:
    python evaluate_benchmarks.py --input_path results.parquet --output_path results_eval.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator of pass@k from the Codex paper.

    Computes the probability that at least one of k randomly selected
    samples from n total samples (where c are correct) is correct.

    Args:
        n: Total number of samples for this problem
        c: Number of correct samples
        k: Number of samples to select

    Returns:
        Estimated pass@k probability
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def evaluate(input_path: str, output_path: str | None = None) -> dict:
    """
    Evaluate benchmark results from a parquet or JSON file.

    Args:
        input_path: Path to parquet or JSON file with results
        output_path: Path to save JSON evaluation results (optional)

    Returns:
        Dictionary containing evaluation results for each dataset
    """
    # Support both parquet and JSON formats
    input_path = str(input_path)
    if input_path.endswith(".json"):
        with open(input_path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif input_path.endswith(".pkl"):
        import pickle
        with open(input_path, "rb") as f:
            df = pickle.load(f)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
    else:
        df = pd.read_parquet(input_path)

    # Validate required columns
    required_columns = ["data_source", "problem", "ground_truth", "responses", "rewards"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    results = {}

    for data_source in df["data_source"].unique():
        subset = df[df["data_source"] == data_source]
        n_problems = len(subset)

        # Skip empty datasets
        if n_problems == 0:
            continue

        # Get number of samples per problem (K)
        first_rewards = subset.iloc[0]["rewards"]
        K = len(first_rewards)

        # Validate all problems have the same number of samples
        sample_counts = subset["rewards"].apply(len)
        if sample_counts.nunique() > 1:
            print(f"Warning: {data_source} has varying sample counts: {sample_counts.unique().tolist()}")
            K = sample_counts.min()  # Use minimum to avoid index errors

        # Compute pass@k for all k from 1 to K
        pass_at_k_list = []
        for k in range(1, K + 1):
            scores = []
            for _, row in subset.iterrows():
                rewards = row["rewards"]
                if isinstance(rewards, np.ndarray):
                    rewards = rewards.tolist()
                n = len(rewards)
                # Count number of correct samples (reward >= 1.0)
                c = sum(1 for r in rewards if r >= 1.0)
                scores.append(pass_at_k(n, c, k))
            pass_at_k_list.append(float(np.mean(scores)))

        # Also compute mean reward (greedy accuracy equivalent)
        mean_rewards = []
        for _, row in subset.iterrows():
            rewards = row["rewards"]
            if isinstance(rewards, np.ndarray):
                rewards = rewards.tolist()
            mean_rewards.append(np.mean(rewards))
        mean_reward = float(np.mean(mean_rewards))

        results[data_source] = {
            "num_problems": n_problems,
            "num_samples": K,
            "mean_reward": mean_reward,
            "pass_at_k": pass_at_k_list,  # [pass@1, pass@2, ..., pass@K]
        }

    # Add aggregate results across all datasets
    total_problems = sum(r["num_problems"] for r in results.values())
    if total_problems > 0 and results:
        # Weighted average of pass@k
        aggregate_pass_at_k = []
        max_k = max(len(r["pass_at_k"]) for r in results.values())
        for k_idx in range(max_k):
            weighted_sum = 0
            weight_sum = 0
            for data_source, data in results.items():
                if k_idx < len(data["pass_at_k"]):
                    weighted_sum += data["pass_at_k"][k_idx] * data["num_problems"]
                    weight_sum += data["num_problems"]
            if weight_sum > 0:
                aggregate_pass_at_k.append(weighted_sum / weight_sum)

        results["_aggregate"] = {
            "num_problems": total_problems,
            "num_samples": max(r["num_samples"] for r in results.values()),
            "pass_at_k": aggregate_pass_at_k,
        }

    # Save full results to JSON
    if output_path is None:
        output_path = str(Path(input_path).with_suffix("")) + "_eval.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation results saved to: {output_path}")

    # Print summary table
    print_summary_table(results)

    return results


def print_summary_table(results: dict):
    """
    Print a summary table of pass@k results.

    Args:
        results: Dictionary of evaluation results by dataset
    """
    print("\n" + "=" * 80)
    print("BENCHMARK EVALUATION RESULTS")
    print("=" * 80)

    # Header
    header = f"{'Dataset':<20} {'#Prob':>6} {'Pass@1':>8} {'Pass@2':>8} {'Pass@4':>8} {'Pass@8':>8} {'Pass@16':>8}"
    print(header)
    print("-" * 80)

    # Sort datasets: _aggregate last, others alphabetically
    sorted_keys = sorted([k for k in results.keys() if not k.startswith("_")])
    if "_aggregate" in results:
        sorted_keys.append("_aggregate")

    for ds in sorted_keys:
        data = results[ds]
        p = data["pass_at_k"]
        K = len(p)
        n_prob = data["num_problems"]

        # Format display name
        display_name = "AGGREGATE" if ds == "_aggregate" else ds

        # Get pass@k values (0-indexed: pass@1 is p[0], pass@2 is p[1], etc.)
        p1 = p[0] * 100 if K >= 1 else 0
        p2 = p[1] * 100 if K >= 2 else 0
        p4 = p[3] * 100 if K >= 4 else 0
        p8 = p[7] * 100 if K >= 8 else 0
        p16 = p[15] * 100 if K >= 16 else 0

        row = f"{display_name:<20} {n_prob:>6} {p1:>7.1f}% {p2:>7.1f}% {p4:>7.1f}% {p8:>7.1f}% {p16:>7.1f}%"

        # Print separator before aggregate
        if ds == "_aggregate":
            print("-" * 80)
        print(row)

    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate benchmark results and compute Pass@k metrics"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to parquet file with generation results",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save JSON evaluation results (default: {input_path}_eval.json)",
    )
    args = parser.parse_args()

    evaluate(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
