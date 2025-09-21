import json
from pathlib import Path
import argparse

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def join_tokens(tokens):
    # Just join tokens with space, simple baseline
    return " ".join(tokens)

def main():
    parser = argparse.ArgumentParser(description="Check overlap between train, val, and test datasets")
    parser.add_argument("--suffix", type=str, default="", help="Optional suffix for files, e.g. '100k'")
    args = parser.parse_args()

    base_dir = Path("data/processed/encoded")

    suffix = f"_{args.suffix}" if args.suffix else ""

    train_path = base_dir / f"train{suffix}.jsonl"
    val_path = base_dir / f"val{suffix}.jsonl"
    test_path = base_dir / f"test{suffix}.jsonl"

    train = load_jsonl(train_path)
    val = load_jsonl(val_path)
    test = load_jsonl(test_path)

    train_texts = {join_tokens(ex["tokens"]) for ex in train}
    val_texts = {join_tokens(ex["tokens"]) for ex in val}
    test_texts = {join_tokens(ex["tokens"]) for ex in test}

    overlap_train_val = train_texts & val_texts
    overlap_train_test = train_texts & test_texts
    overlap_val_test = val_texts & test_texts

    print(f"Overlap train-val: {len(overlap_train_val)}")
    print(f"Overlap train-test: {len(overlap_train_test)}")
    print(f"Overlap val-test: {len(overlap_val_test)}")

    if overlap_train_test:
        print("\nExamples in both train & test:")
        for example in list(overlap_train_test)[:5]:
            print(example)

if __name__ == "__main__":
    main()