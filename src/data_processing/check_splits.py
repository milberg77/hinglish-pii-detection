import json
from pathlib import Path

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def join_tokens(tokens):
    # Just join tokens with space, simple baseline
    return " ".join(tokens)

if __name__ == "__main__":
    base_dir = Path("data/processed/encoded")
    train = load_jsonl(base_dir / "train.jsonl")
    val = load_jsonl(base_dir / "val.jsonl")
    test = load_jsonl(base_dir / "test.jsonl")

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
