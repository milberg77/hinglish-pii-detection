#!/usr/bin/env python3
"""
Train token-classification model for Hinglish PII detection.

Expects preprocessed JSONL files (encoded) where each line is:
{"id": "...", "tokens": ["Tok", "##en", ...], "labels": ["O","B-CITY", ...]}

Usage (CPU test):
python src/models/train_anonymizer.py \
  --model_name_or_path bert-base-multilingual-cased \
  --train_file data/processed/encoded/train.jsonl \
  --val_file data/processed/encoded/val.jsonl \
  --output_dir models/final/hinglish_tokenclf_v1 \
  --per_device_train_batch_size 8 \
  --num_train_epochs 3 \
  --max_seq_length 128

Usage (Colab GPU): use larger batch sizes and set --model_name_or_path to
ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii
if it supports token-classification; otherwise use the BERT baseline above.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

# seqeval for entity metrics
try:
    from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
    HAS_SEQEVAL = True
except Exception:
    HAS_SEQEVAL = False


def read_jsonl_examples(path: str) -> List[Dict]:
    records = []
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
            records.append({"id": uid, "tokens": tokens, "labels": labels})
    return records


def build_label_list(label_seqs: List[List[str]]) -> List[str]:
    seen = set()
    for seq in label_seqs:
        for l in seq:
            seen.add(l)
    # ensure paired B-/I- for each entity
    additions = set()
    for l in list(seen):
        if l.startswith("B-"):
            ent = l.split("-", 1)[1]
            if f"I-{ent}" not in seen:
                additions.add(f"I-{ent}")
        if l.startswith("I-"):
            ent = l.split("-", 1)[1]
            if f"B-{ent}" not in seen:
                additions.add(f"B-{ent}")
    seen.update(additions)
    labels = ["O"] + sorted([x for x in seen if x != "O"])
    return labels


def tokenize_and_align_labels(examples, tokenizer, label2id, max_seq_length):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
    )
    aligned_labels = []
    for i, labs in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                lab = labs[word_idx]
                if word_idx != previous_word_idx:
                    # first subtoken -> original label
                    label_to_use = lab
                else:
                    # subsequent subtoken -> convert B- to I- (or keep I- / O)
                    if lab == "O":
                        label_to_use = "O"
                    elif lab.startswith("B-"):
                        ent = lab.split("-", 1)[1]
                        label_to_use = f"I-{ent}"
                    else:
                        label_to_use = lab
                label_ids.append(label2id.get(label_to_use, label2id["O"]))
            previous_word_idx = word_idx
        aligned_labels.append(label_ids)
    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs


def compute_metrics(pred):
    predictions, label_ids = pred
    preds = np.argmax(predictions, axis=2)

    true_seqs, pred_seqs = [], []
    for i in range(label_ids.shape[0]):
        true_seq = []
        pred_seq = []
        for j in range(label_ids.shape[1]):
            if label_ids[i, j] == -100:
                continue
            true_seq.append(id2label[label_ids[i, j]])
            pred_seq.append(id2label[preds[i, j]])
        true_seqs.append(true_seq)
        pred_seqs.append(pred_seq)

    if HAS_SEQEVAL:
        p = precision_score(true_seqs, pred_seqs)
        r = recall_score(true_seqs, pred_seqs)
        f = f1_score(true_seqs, pred_seqs)
        return {"precision": p, "recall": r, "f1": f}
    else:
        total = 0
        correct = 0
        for t, p in zip(true_seqs, pred_seqs):
            for a, b in zip(t, p):
                total += 1
                if a == b:
                    correct += 1
        acc = correct / total if total > 0 else 0.0
        return {"token_accuracy": acc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train token classification model for Hinglish PII")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # read data
    print("Loading examples...")
    train_examples = read_jsonl_examples(args.train_file)
    val_examples = read_jsonl_examples(args.val_file)

    all_label_seqs = [ex["labels"] for ex in train_examples] + [ex["labels"] for ex in val_examples]
    label_list = build_label_list(all_label_seqs)
    label2id = {lab: i for i, lab in enumerate(label_list)}
    id2label = {i: lab for lab, i in label2id.items()}

    print(f"Labels ({len(label_list)}): {label_list}")

    # attempt to load tokenizer and token-classification model
    print(f"Loading tokenizer and model from {args.model_name_or_path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name_or_path,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
        )
    except Exception as e:
        print("Warning: Could not load the provided model as AutoModelForTokenClassification.")
        print("Exception:", e)
        print("Falling back to 'bert-base-multilingual-cased' baseline.")
        args.model_name_or_path = "bert-base-multilingual-cased"
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name_or_path,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
        )

    # ensure pad token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

    # build HF datasets
    train_ds = Dataset.from_list(train_examples)
    val_ds = Dataset.from_list(val_examples)
    ds = DatasetDict({"train": train_ds, "validation": val_ds})

    # map tokenization/alignment
    print("Tokenizing and aligning labels...")
    def _map_fn(batch):
        return tokenize_and_align_labels(batch, tokenizer, label2id, args.max_seq_length)

    tokenized = ds.map(_map_fn, batched=True, remove_columns=["id", "tokens", "labels"])

    # data collator and training args
    data_collator = DataCollatorForTokenClassification(tokenizer)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=200,
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="f1" if HAS_SEQEVAL else "eval_loss",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # train
    print("Starting training...")
    trainer.train()

    # save artifacts
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "label2id.json"), "w", encoding="utf-8") as fh:
        json.dump(label2id, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "id2label.json"), "w", encoding="utf-8") as fh:
        json.dump(id2label, fh, indent=2, ensure_ascii=False)

    print("Training finished. Artifacts saved to:", args.output_dir)
    if HAS_SEQEVAL:
        print("You can inspect a detailed classification report using seqeval on your validation predictions.")
