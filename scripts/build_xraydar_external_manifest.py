import csv
import json
from collections import Counter
from pathlib import Path

root = Path(r"data/raw/xraydar")
annotations = root / "annotations.jsonl"
output = Path(r"data/manifests/xraydar_external.csv")

mapping = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Effusion": "pleural_effusion",
    "Pneumothorax": "pneumothorax",
    "Consolidation": "consolidation",
    "Emphysema": "emphysema",
    "Hernia": "hernia",
}

rows = [
    json.loads(line)
    for line in annotations.open(encoding="utf-8")
    if line.strip()
]

ids = [row["xray_id"] for row in rows]
files = [row["image_file"] for row in rows]

missing = [
    image_file
    for image_file in files
    if not (root / image_file).is_file()
]

zero_size = [
    image_file
    for image_file in files
    if (root / image_file).is_file()
    and (root / image_file).stat().st_size == 0
]

image_files = sorted((root / "images").glob("*.png"))

output.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "dataset",
    "external_split",
    "xray_id",
    "image_file",
    "consensus_labels",
    *mapping.keys(),
]

with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for row in rows:
        labels = set(row["consensus_labels"])

        item = {
            "dataset": "xraydar",
            "external_split": "external_test",
            "xray_id": row["xray_id"],
            "image_file": row["image_file"],
            "consensus_labels": "|".join(sorted(labels)),
        }

        for nih_label, xraydar_label in mapping.items():
            item[nih_label] = int(xraydar_label in labels)

        writer.writerow(item)

counts = Counter()

for row in rows:
    labels = set(row["consensus_labels"])

    for nih_label, xraydar_label in mapping.items():
        counts[nih_label] += int(xraydar_label in labels)

print("Records:", len(rows))
print("Unique xray_id:", len(set(ids)))
print("PNG files:", len(image_files))
print("Missing images:", len(missing))
print("Zero-size images:", len(zero_size))

for label in mapping:
    print(f"{label}: {counts[label]}")

assert len(rows) == 979
assert len(set(ids)) == 979
assert len(set(files)) == 979
assert len(image_files) == 979
assert not missing
assert not zero_size

print("AUDIT PASSED")
