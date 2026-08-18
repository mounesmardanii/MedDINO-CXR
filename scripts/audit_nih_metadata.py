from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "nih"

METADATA_PATH = DATA_ROOT / "Data_Entry_2017_v2020.csv"
TRAIN_VAL_PATH = DATA_ROOT / "train_val_list.txt"
TEST_PATH = DATA_ROOT / "test_list.txt"
BBOX_PATH = DATA_ROOT / "BBox_List_2017.csv"

REQUIRED_COLUMNS = (
    "Image Index",
    "Finding Labels",
    "Patient ID",
    "Patient Age",
    "Patient Gender",
    "View Position",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_image_list(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def main():
    metadata = pd.read_csv(METADATA_PATH)
    train_val_images = load_image_list(TRAIN_VAL_PATH)
    test_images = load_image_list(TEST_PATH)
    bbox = pd.read_csv(BBOX_PATH)

    print("NIH ChestX-ray14 metadata audit")
    print()

    print(f"Metadata rows: {len(metadata)}")
    print(f"Metadata columns: {len(metadata.columns)}")
    print(f"Train/val images: {len(train_val_images)}")
    print(f"Test images: {len(test_images)}")
    print(f"Bounding-box rows: {len(bbox)}")
    print()

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in metadata.columns
    ]

    require(
        not missing_columns,
        f"Missing required columns: {missing_columns}",
    )

    require(
        metadata["Image Index"].is_unique,
        "Image Index is not unique",
    )

    metadata_images = set(
        metadata["Image Index"].astype(str)
    )

    train_val_set = set(train_val_images)
    test_set = set(test_images)

    require(
        len(train_val_images) == len(train_val_set),
        "Duplicate image found in train_val_list.txt",
    )

    require(
        len(test_images) == len(test_set),
        "Duplicate image found in test_list.txt",
    )

    split_overlap = train_val_set & test_set

    require(
        not split_overlap,
        f"Image overlap between train_val and test: {len(split_overlap)}",
    )

    missing_train_val = train_val_set - metadata_images
    missing_test = test_set - metadata_images

    require(
        not missing_train_val,
        f"Train/val images missing from metadata: {len(missing_train_val)}",
    )

    require(
        not missing_test,
        f"Test images missing from metadata: {len(missing_test)}",
    )

    listed_images = train_val_set | test_set
    unassigned_images = metadata_images - listed_images

    print(f"Images assigned to official split: {len(listed_images)}")
    print(f"Metadata images not in split files: {len(unassigned_images)}")
    print()

    metadata_indexed = metadata.set_index(
        "Image Index",
        drop=False,
    )

    train_val_metadata = metadata_indexed.loc[
        train_val_images
    ].copy()

    test_metadata = metadata_indexed.loc[
        test_images
    ].copy()

    train_val_patients = set(
        train_val_metadata["Patient ID"].astype(str)
    )

    test_patients = set(
        test_metadata["Patient ID"].astype(str)
    )

    patient_overlap = train_val_patients & test_patients

    require(
        not patient_overlap,
        f"Patient leakage between train_val and test: {len(patient_overlap)}",
    )

    print(f"Total patients: {metadata['Patient ID'].nunique()}")
    print(f"Train/val patients: {len(train_val_patients)}")
    print(f"Test patients: {len(test_patients)}")
    print("Patient overlap: 0")
    print()

    print("View positions:")
    print(
        metadata["View Position"]
        .value_counts(dropna=False)
        .to_string()
    )
    print()

    allowed_views = {
        "AP",
        "PA",
    }

    unexpected_views = (
        set(
            metadata["View Position"]
            .dropna()
            .astype(str)
            .unique()
        )
        - allowed_views
    )

    print(f"Unexpected view values: {sorted(unexpected_views)}")
    print()

    label_counter = Counter()

    for value in metadata["Finding Labels"].fillna(""):
        for label in str(value).split("|"):
            label = label.strip()

            if label:
                label_counter[label] += 1

    print(f"Unique label names: {len(label_counter)}")
    print()

    for label, count in sorted(label_counter.items()):
        print(f"{label}: {count}")

    print()

    bbox_image_column = None

    for candidate in (
        "Image Index",
        "Image Index ",
    ):
        if candidate in bbox.columns:
            bbox_image_column = candidate
            break

    require(
        bbox_image_column is not None,
        "Could not identify bounding-box image column",
    )

    bbox_images = set(
        bbox[bbox_image_column]
        .dropna()
        .astype(str)
    )

    missing_bbox_images = bbox_images - metadata_images

    require(
        not missing_bbox_images,
        f"Bounding-box images missing from metadata: {len(missing_bbox_images)}",
    )

    print(f"Unique images with bounding boxes: {len(bbox_images)}")
    print()

    print("Audit passed.")
    print("Official train_val/test image overlap: 0")
    print("Official train_val/test patient overlap: 0")


if __name__ == "__main__":
    main()