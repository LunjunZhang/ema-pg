from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry


def prepare_train_data():
    """Download and register the DeepScaleR training dataset."""
    train_dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")

    def preprocess_fn(example, idx):
        return {
            "question": example["problem"],
            "ground_truth": example["answer"],
            "data_source": "deepscaler",
        }

    train_dataset = train_dataset.map(preprocess_fn, with_indices=True)
    train_dataset = DatasetRegistry.register_dataset("deepscaler_math", train_dataset, "train")
    return train_dataset


def prepare_all_test_datasets():
    """Download and register all 7 math test datasets."""

    # AIME (HuggingFaceH4/aime_2024)
    print("Downloading AIME dataset...")
    aime = load_dataset("HuggingFaceH4/aime_2024", split="train")
    aime_data = [
        {
            "question": ex["problem"],
            "ground_truth": ex["answer"],
            "data_source": "aime2024",
        }
        for ex in aime
    ]
    print(f"  Loaded {len(aime_data)} AIME problems")
    DatasetRegistry.register_dataset("aime2024", aime_data, "test")

    # AMC (from opencompass/LiveMathBench, subset v202412_AMC_en)
    print("Downloading AMC dataset from LiveMathBench...")
    amc = load_dataset("opencompass/LiveMathBench", "v202412_AMC_en", split="test")
    amc_data = [
        {
            "question": ex["question"],
            "ground_truth": ex["answer"],
            "data_source": "amc",
        }
        for ex in amc
    ]
    print(f"  Loaded {len(amc_data)} AMC problems")
    DatasetRegistry.register_dataset("amc", amc_data, "test")

    # MATH-500 (HuggingFaceH4/MATH-500)
    print("Downloading MATH-500 dataset...")
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    math_data = [
        {
            "question": ex["problem"],
            "ground_truth": ex["answer"],
            "data_source": "math500",
        }
        for ex in math500
    ]
    print(f"  Loaded {len(math_data)} MATH-500 problems")
    DatasetRegistry.register_dataset("math500", math_data, "test")

    # MINERVA (math-ai/minervamath)
    print("Downloading MINERVA dataset...")
    minerva = load_dataset("math-ai/minervamath", split="test")
    minerva_data = [
        {
            "question": ex["question"],
            "ground_truth": ex["answer"],
            "data_source": "minerva",
        }
        for ex in minerva
    ]
    print(f"  Loaded {len(minerva_data)} MINERVA problems")
    DatasetRegistry.register_dataset("minerva", minerva_data, "test")

    # OLYMPIAD_BENCH (Hothan/OlympiadBench, subset OE_TO_maths_en_COMP)
    print("Downloading OLYMPIAD_BENCH dataset...")
    olympiad = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
    olympiad_data = [
        {
            "question": ex["question"],
            "ground_truth": ex["final_answer"],
            "data_source": "olympiad_bench",
        }
        for ex in olympiad
    ]
    print(f"  Loaded {len(olympiad_data)} OLYMPIAD_BENCH problems")
    DatasetRegistry.register_dataset("olympiad_bench", olympiad_data, "test")

    # AIME 2025 (MathArena/aime_2025)
    print("Downloading AIME 2025 dataset...")
    aime2025 = load_dataset("MathArena/aime_2025", split="train")
    aime2025_data = [
        {
            "question": ex["problem"],
            "ground_truth": ex["answer"],
            "data_source": "aime2025",
        }
        for ex in aime2025
    ]
    print(f"  Loaded {len(aime2025_data)} AIME 2025 problems")
    DatasetRegistry.register_dataset("aime2025", aime2025_data, "test")

    # AMC 2025 (from opencompass/LiveMathBench, subset v202505_hard_en)
    print("Downloading AMC 2025 dataset from LiveMathBench...")
    amc2025 = load_dataset("opencompass/LiveMathBench", "v202505_hard_en", split="test")
    amc2025_data = [
        {
            "question": ex["question"],
            "ground_truth": ex["answer"],
            "data_source": "amc2025",
        }
        for ex in amc2025
    ]
    print(f"  Loaded {len(amc2025_data)} AMC 2025 problems")
    DatasetRegistry.register_dataset("amc2025", amc2025_data, "test")

    print(f"\nTotal: Registered 7 test datasets")


def prepare_all_math_data():
    """Download and register training dataset and all 7 test datasets."""
    train_dataset = prepare_train_data()
    prepare_all_test_datasets()
    return train_dataset


if __name__ == "__main__":
    train_dataset = prepare_all_math_data()
    print(f"\nTraining data path: {train_dataset.get_data_path()}")
    print("\nAll registered test datasets:")
    for name in ["aime2024", "amc", "math500", "minerva", "olympiad_bench", "aime2025", "amc2025"]:
        dataset = DatasetRegistry.load_dataset(name, "test")
        if dataset:
            print(f"  {name}: {len(dataset)} problems at {dataset.get_data_path()}")
