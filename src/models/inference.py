#!/usr/bin/env python3
"""
Inference script for token-classification anonymizer.

Example:
python src/models/inference.py \
  --model_dir models/bert_tokenclf_v1/checkpoint-5000 \
  --text "Hi, mera naam Arun Sharma hai. Phone +91-9876543210." \
  --max_seq_length 128

Or for batch input (--input_file a JSONL where each line is plain text):
python src/models/inference.py --model_dir ... --input_file data/manual_texts.jsonl --out results/inference_batch.jsonl
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def load_label_maps(model_dir: str):
    p = Path(model_dir)
    id2label = None
    if (p / "id2label.json").exists():
        with open(p / "id2label.json", "r", encoding="utf-8") as fh:
            id2label = json.load(fh)
        id2label = {int(k): v for k, v in id2label.items()}
    elif (p / "label2id.json").exists():
        with open(p / "label2id.json", "r", encoding="utf-8") as fh:
            label2id = json.load(fh)
        id2label = {v: k for k, v in label2id.items()}
    else:
        # fallback to model config (may be present)
        id2label = None
    return id2label

def predict_spans(text: str, tokenizer, model, id2label, max_seq_length=128, device="cpu"):
    model.to(device)
    model.eval()
    with torch.no_grad():
        enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_seq_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        offsets = enc["offset_mapping"][0].tolist()  # list of (start,end)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.detach().cpu().numpy()[0]  # (seq_len, num_labels)
        preds = logits.argmax(axis=-1).tolist()  # length seq_len

    # Map predicted ids to labels
    label_seq = [id2label.get(int(pid), "O") for pid in preds]

    # Build spans by merging contiguous non-O tokens of same entity type
    spans = []
    cur = None
    for idx, lab in enumerate(label_seq):
        off = offsets[idx]
        if off[0] == off[1] == 0:
            # special token, skip
            if cur:
                spans.append(cur); cur = None
            continue
        if lab == "O":
            if cur:
                spans.append(cur); cur = None
            continue
        # lab is B- or I-
        ent = lab.split("-", 1)[1] if "-" in lab else lab
        start, end = off[0], off[1]
        if cur is None:
            cur = {"label": ent, "start": start, "end": end, "labels": [lab]}
        else:
            if ent == cur["label"] and start <= cur["end"]:
                # extend
                cur["end"] = end
                cur["labels"].append(lab)
            elif ent == cur["label"] and start > cur["end"]:
                # non-contiguous but same label: still extend end
                cur["end"] = end
                cur["labels"].append(lab)
            else:
                spans.append(cur)
                cur = {"label": ent, "start": start, "end": end, "labels": [lab]}
    if cur:
        spans.append(cur)

    # populate text fragments
    for s in spans:
        s["text"] = text[s["start"]:s["end"]]

    # masked text (replace spans in reverse order)
    masked = text
    # compute replacements counts per label to generate placeholders [LABEL_1], [LABEL_2], ...
    counts = {}
    for s in sorted(spans, key=lambda x: x["start"], reverse=True):
        lab = s["label"]
        counts.setdefault(lab, 0)
        counts[lab] += 1
        placeholder = f"[{lab}_{counts[lab]}]"
        masked = masked[:s["start"]] + placeholder + masked[s["end"]:]

    return {"text": text, "spans": spans, "masked_text": masked}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference & anonymization using a token-classification model")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--text", type=str, default=None, help="Single sentence to infer")
    parser.add_argument("--input_file", type=str, default=None, help="Optional: newline JSONL or plain text file for batch inference")
    parser.add_argument("--out", type=str, default="results/inference_out.jsonl")
    parser.add_argument("--max_seq_length", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    id2label = load_label_maps(args.model_dir) or {int(k): v for k, v in getattr(model.config, "id2label", {}).items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    if args.text:
        res = predict_spans(args.text, tokenizer, model, id2label, args.max_seq_length, device=device)
        print("Spans detected:", res["spans"])
        print("Masked:", res["masked_text"])
    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as inf, open(args.out, "w", encoding="utf-8") as outf:
            for line in inf:
                line = line.strip()
                if not line:
                    continue
                # support JSONL with {"text": "..."} or plain text lines
                try:
                    parsed = json.loads(line)
                    txt = parsed.get("text") or parsed.get("source_text") or list(parsed.values())[0]
                except Exception:
                    txt = line
                res = predict_spans(txt, tokenizer, model, id2label, args.max_seq_length, device=device)
                outf.write(json.dumps(res, ensure_ascii=False) + "\n")
        print("Batch inference done. Results at:", args.out)
    else:
        parser.print_help()
