#!/usr/bin/env python3
"""
src/data_processing/data_validator.py

Usage:
    python src/data_processing/data_validator.py --input data/synthetic/hinglish_banking_2k.jsonl --schema data/docs/data_schema.json --out failures_2k.jsonl --sample 500
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import re
from typing import Dict, Any


def load_schema(schema_path: str) -> Dict[str, Any]:
    if not Path(schema_path).exists():
        return {}
    with open(schema_path, "r", encoding="utf-8") as fh:
        schema = json.load(fh)
    compiled = {}
    for label, info in schema.items():
        pat = info.get("regex")
        if pat:
            try:
                compiled[label] = re.compile(pat)
            except re.error:
                # fallback: compile with UNICODE
                compiled[label] = re.compile(pat, flags=re.UNICODE)
    return compiled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL (generator output)")
    parser.add_argument("--schema", required=False, default="data/docs/data_schema.json", help="Schema JSON (for regex validation)")
    parser.add_argument("--out", required=False, default=None, help="Optional path to write failing samples JSONL")
    parser.add_argument("--sample", type=int, default=500, help="How many failing examples to save (max)")
    args = parser.parse_args()

    schema = load_schema(args.schema)

    stats = Counter()
    per_label_counts = Counter()
    failing_records = []
    total = 0
    line_errors = defaultdict(int)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    with input_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                line_errors["json_decode_error"] += 1
                continue

            src = rec.get("source_text", "")
            privacy_mask = rec.get("privacy_mask", [])
            if isinstance(privacy_mask, str):
                try:
                    privacy_mask = json.loads(privacy_mask)
                except Exception:
                    privacy_mask = []

            tokens = rec.get("mBERT_tokens", [])
            labels = rec.get("mBERT_token_classes", [])
            entities = rec.get("entities", {})
            if isinstance(entities, str):
                try:
                    entities = json.loads(entities)
                except Exception:
                    entities = {}

            # token/label count check
            if len(tokens) != len(labels):
                stats["token_label_mismatch"] += 1
                if len(failing_records) < args.sample:
                    rec["_error"] = "token_label_mismatch"
                    failing_records.append(rec)
                continue

            # privacy mask span integrity
            pm_bad = False
            for m in privacy_mask:
                try:
                    s = int(m["start"])
                    e = int(m["end"])
                    v = m.get("value", "")
                except Exception:
                    pm_bad = True
                    break
                if src[s:e] != v:
                    pm_bad = True
                    break
                # count labels
                stats[f"mask_label_{m['label']}"] += 1
                per_label_counts[m["label"]] += 1

            if pm_bad:
                stats["privacy_mask_mismatch"] += 1
                if len(failing_records) < args.sample:
                    rec["_error"] = "privacy_mask_mismatch"
                    failing_records.append(rec)
                continue

            # schema regex validation for each entity (if schema provided)
            schema_bad = False
            for etype, val in entities.items():
                pat = schema.get(etype)
                if pat:
                    if not pat.fullmatch(val):
                        schema_bad = True
                        line_errors[f"schema_mismatch_{etype}"] += 1
            if schema_bad:
                stats["schema_mismatch"] += 1
                if len(failing_records) < args.sample:
                    rec["_error"] = "schema_mismatch"
                    failing_records.append(rec)
                continue

            # passed all checks
            stats["ok"] += 1

    # Summary
    print("Validation summary")
    print("==================")
    print(f"Input file: {input_path}")
    print(f"Total records processed: {total}")
    print()
    print("Main outcomes:")
    for k, v in stats.most_common():
        print(f"  {k}: {v}")
    print()
    print("Per-label mask counts (top 20):")
    for label, cnt in per_label_counts.most_common(20):
        print(f"  {label}: {cnt}")

    if line_errors:
        print()
        print("Other errors / mismatches (sample):")
        for k, v in list(line_errors.items())[:20]:
            print(f"  {k}: {v}")

    # optionally write failing records for manual inspection
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as fh:
            for rec in failing_records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print()
        print(f"Wrote {len(failing_records)} failing records to {outp}")

if __name__ == "__main__":
    main()