import json
import random
import os

def split_dataset(input_file, output_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    random.seed(seed)

    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    random.shuffle(data)
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_data = data[:n_train]
    val_data = data[n_train:n_train+n_val]
    test_data = data[n_train+n_val:]

    os.makedirs(output_dir, exist_ok=True)
    for name, subset in zip(["train", "val", "test"], [train_data, val_data, test_data]):
        with open(os.path.join(output_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for rec in subset:
                f.write(json.dumps(rec) + "\n")
        print(f"Saved {len(subset)} → {output_dir}/{name}.jsonl")

if __name__ == "__main__":
    split_dataset(
        input_file="data/processed/hinglish_banking_20k_clean.jsonl",
        output_dir="data/processed",
        train_ratio=0.8,
        val_ratio=0.1,
    )
