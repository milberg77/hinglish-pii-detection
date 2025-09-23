import json
import random
from pathlib import Path
from collections import defaultdict
import argparse

def controlled_split(
    in_file,
    out_dir,
    seed=42,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    suffix=""
):
    random.seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all records
    records = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            records.append(rec)

    print(f"Loaded {len(records)} records")

    # Group by template "skeleton" (masked_text is a good proxy for template)
    groups = defaultdict(list)
    for rec in records:
        key = rec["masked_text"]
        groups[key].append(rec)

    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    n = len(group_keys)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    n_test = n - n_train - n_val

    train_keys = set(group_keys[:n_train])
    val_keys = set(group_keys[n_train:n_train+n_val])
    test_keys = set(group_keys[n_train+n_val:])

    print(f"Groups: train={len(train_keys)}, val={len(val_keys)}, test={len(test_keys)}")

    # Helper to add suffix before .jsonl
    def suffixed_filename(base_name):
        return f"{base_name}{('_' + suffix) if suffix else ''}.jsonl"

    # Write splits
    split_map = {
        suffixed_filename("train"): train_keys,
        suffixed_filename("val"): val_keys,
        suffixed_filename("test"): test_keys
    }

    for fname, keyset in split_map.items():
        out_path = out_dir / fname
        with open(out_path, "w", encoding="utf-8") as f:
            count = 0
            for key in keyset:
                random.shuffle(groups[key])
                for rec in groups[key]:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    count += 1
        print(f"Wrote {count} → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Controlled split of JSONL data by masked_text template")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for splits")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Proportion of training groups")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Proportion of validation groups")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Proportion of test groups")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--suffix", type=str, default="", help="Suffix to add before .jsonl in output filenames")

    args = parser.parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not (0.99 <= total_ratio <= 1.01):
        raise ValueError("Train/val/test ratios must sum to 1.0")

    controlled_split(
        in_file=args.input,
        out_dir=args.output,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        suffix=args.suffix
    )
