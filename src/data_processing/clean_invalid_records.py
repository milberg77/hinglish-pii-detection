import json
import re

def is_valid_email(email: str) -> bool:
    return re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$", email) is not None

def is_valid_fullname(name: str) -> bool:
    return re.match(r"^[A-Z][a-zA-Z]+( [A-Z][a-zA-Z]+)+$", name) is not None

def is_valid_surname(surname: str) -> bool:
    return re.match(r"^[A-Z][a-z]{1,20}$", surname) is not None

def clean_invalid_records(input_file, output_file):
    cleaned = []
    removed = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)

            # Check for schema-violating fields
            if "EMAIL" in rec and not is_valid_email(rec["EMAIL"]):
                removed += 1
                continue
            if "FULLNAME" in rec and not is_valid_fullname(rec["FULLNAME"]):
                removed += 1
                continue
            if "SURNAME" in rec and not is_valid_surname(rec["SURNAME"]):
                removed += 1
                continue

            cleaned.append(rec)

    with open(output_file, "w", encoding="utf-8") as f:
        for rec in cleaned:
            f.write(json.dumps(rec) + "\n")

    print(f"Cleaned dataset saved: {output_file}")
    print(f"Removed invalids: {removed} | Final: {len(cleaned)}")

if __name__ == "__main__":
    clean_invalid_records(
        input_file="data/synthetic/hinglish_banking_20k.jsonl",
        output_file="data/processed/hinglish_banking_20k_clean.jsonl"
    )
