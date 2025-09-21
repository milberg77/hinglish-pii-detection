import argparse
import json
import os

def process_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line)

            # Extract required fields
            tokens = record.get("mBERT_tokens", [])
            labels = record.get("mBERT_token_classes", [])

            # Safety check
            if len(tokens) != len(labels):
                print(f"Skipping UID {record.get('uid')} due to mismatch (tokens={len(tokens)}, labels={len(labels)})")
                continue

            out_record = {
                "id": record.get("uid"),
                "tokens": tokens,
                "labels": labels
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    def suffixed_path(split_name):
        return os.path.join(args.out_dir, f"{split_name}{('_' + args.suffix) if args.suffix else ''}.jsonl")

    if args.train:
        process_file(args.train, suffixed_path("train"))
    if args.val:
        process_file(args.val, suffixed_path("val"))
    if args.test:
        process_file(args.test, suffixed_path("test"))

    print(f"Preprocessing complete. Encoded files saved to {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Hinglish PII dataset into token-label format")
    parser.add_argument("--train", type=str, help="Path to training JSONL file")
    parser.add_argument("--val", type=str, help="Path to validation JSONL file")
    parser.add_argument("--test", type=str, help="Path to test JSONL file")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--suffix", type=str, default="", help="Suffix to add before .jsonl in output filenames")
    args = parser.parse_args()

    main(args)
