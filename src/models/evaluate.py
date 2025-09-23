#!/usr/bin/env python3
"""
Evaluate token-classification checkpoint on test.jsonl (encoded format).

Writes:
 - printed classification report (seqeval if available)
 - JSONL file with token-level true/pred labels
 - summary metrics saved as JSON

Usage:
python src/models/evaluate.py \
  --model_dir models/bert_tokenclf_v1/checkpoint-5000 \
  --test_file data/processed/encoded/test.jsonl \
  --batch_size 32 \
  --max_seq_length 128 \
  --out_dir results/eval_checkpoint-5000
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments

try:
    from seqeval.metrics import classification_report as cr_dict, precision_score, recall_score, f1_score
    HAS_SEQEVAL = True
except Exception:
    HAS_SEQEVAL = False


def read_jsonl_examples(path: str) -> List[Dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            tokens = rec.get("tokens") or rec.get("mBERT_tokens")
            labels = rec.get("labels") or rec.get("mBERT_token_classes")
            uid = rec.get("id") or rec.get("uid") or rec.get("uid".upper())
            if tokens is None or labels is None:
                raise ValueError(f"Missing tokens/labels in record keys: {list(rec.keys())}")
            if len(tokens) != len(labels):
                raise ValueError(f"Token/label length mismatch uid={uid}: {len(tokens)} vs {len(labels)}")
            examples.append({"id": uid, "tokens": tokens, "labels": labels})
    return examples


def build_label_maps(model_dir: str):
    p_lab2id = Path(model_dir) / "label2id.json"
    p_id2lab = Path(model_dir) / "id2label.json"
    if p_id2lab.exists():
        with open(p_id2lab, "r", encoding="utf-8") as fh:
            id2label = json.load(fh)
        # convert keys to int if necessary
        id2label = {int(k): v for k, v in id2label.items()}
        label_list = [id2label[i] for i in range(len(id2label))]
        label2id = {v: k for k, v in id2label.items()}
        return label_list, label2id, id2label
    elif p_lab2id.exists():
        with open(p_lab2id, "r", encoding="utf-8") as fh:
            label2id = json.load(fh)
        label_list = sorted(label2id.keys(), key=lambda x: label2id[x])
        id2label = {v: k for k, v in label2id.items()}
        return label_list, label2id, id2label
    else:
        return None, None, None


def tokenize_and_align_labels(batch, tokenizer, label2id, max_seq_length):
    tokenized = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True,
                          padding="max_length", max_length=max_seq_length)
    aligned = []
    for i, labs in enumerate(batch["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word = None
        label_ids = []
        for widx in word_ids:
            if widx is None:
                label_ids.append(-100)
            else:
                lab = labs[widx]
                if widx != prev_word:
                    label_to_use = lab
                else:
                    if lab == "O":
                        label_to_use = "O"
                    elif lab.startswith("B-"):
                        ent = lab.split("-", 1)[1]
                        label_to_use = f"I-{ent}"
                    else:
                        label_to_use = lab
                label_ids.append(label2id.get(label_to_use, label2id.get("O", 0)))
            prev_word = widx
        aligned.append(label_ids)
    tokenized["labels"] = aligned
    return tokenized

def convert_to_serializable(obj):
    """Recursively convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate token-classification model")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--out_dir", type=str, default="results/eval")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model and tokenizer from:", args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    # load label maps (fall back to model config)
    label_list, label2id, id2label = build_label_maps(args.model_dir)
    if label_list is None:
        # try model config
        id2label = {int(k): v for k, v in getattr(model.config, "id2label", {}).items()}
        label_list = [id2label[i] for i in range(len(id2label))]
        label2id = {v: k for k, v in id2label.items()}

    print("Labels:", label_list)

    print("Loading test examples...")
    test_examples = read_jsonl_examples(args.test_file)
    test_ds = Dataset.from_list(test_examples)

    print("Tokenizing and aligning labels (this uses the same logic as training)...")
    def map_fn(batch): return tokenize_and_align_labels(batch, tokenizer, label2id, args.max_seq_length)
    tokenized_test = test_ds.map(map_fn, batched=True, remove_columns=["id", "tokens", "labels"])

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # lightweight Trainer for prediction
    training_args = TrainingArguments(output_dir=str(out_dir / "tmp_trainer"), per_device_eval_batch_size=args.batch_size,
                                      do_train=False, do_eval=False, logging_dir=str(out_dir / "logs"), report_to="none")
    trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer, data_collator=data_collator)

    print("Running predictions...")
    pred_output = trainer.predict(tokenized_test)
    logits = pred_output.predictions
    label_ids = pred_output.label_ids

    preds = np.argmax(logits, axis=-1)

    true_label_seqs = []
    pred_label_seqs = []

    jsonl_out = out_dir / "predictions_token_level.jsonl"
    with open(jsonl_out, "w", encoding="utf-8") as outf:
        for i in range(label_ids.shape[0]):
            true_seq = []
            pred_seq = []
            # iterate through sequence length
            for j in range(label_ids.shape[1]):
                lid = int(label_ids[i, j])
                if lid == -100:
                    continue
                true_label = id2label[lid]
                pred_label = id2label[int(preds[i, j])]
                true_seq.append(true_label)
                pred_seq.append(pred_label)
            true_label_seqs.append(true_seq)
            pred_label_seqs.append(pred_seq)

            # also write token-level arrays (original tokens)
            out_rec = {
                "index": i,
                "true_labels": true_seq,
                "pred_labels": pred_seq,
            }
            outf.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    if HAS_SEQEVAL:
        p = precision_score(true_label_seqs, pred_label_seqs)
        r = recall_score(true_label_seqs, pred_label_seqs)
        f = f1_score(true_label_seqs, pred_label_seqs)

        # full report as dict
        report_dict = cr_dict(true_label_seqs, pred_label_seqs, output_dict=True)
        report_text = cr_dict(true_label_seqs, pred_label_seqs)

        print("=== Seqeval Report ===")
        print(report_text)

        summary = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "per_class": report_dict
        }

    else:
        # token accuracy fallback
        total = 0
        correct = 0
        for t_seq, p_seq in zip(true_label_seqs, pred_label_seqs):
            for a, b in zip(t_seq, p_seq):
                total += 1
                if a == b:
                    correct += 1
        acc = correct / total if total > 0 else 0.0
        print(f"Token-level accuracy: {acc:.4f}")
        summary = {"token_accuracy": float(acc)}

    # Save summary
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as fh:
        json.dump(convert_to_serializable(summary), fh, indent=2, ensure_ascii=False)

    print("Saved token-level predictions to:", jsonl_out)
    print("Saved metrics to:", out_dir / "metrics_summary.json")
    print("Done.")
