import argparse
import tarfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "nih"
SHARD_ROOT = DATA_ROOT / "hf_shards" / "data"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "manifests" / "nih_manifest.csv"

METADATA_PATH = DATA_ROOT / "Data_Entry_2017_v2020.csv"
TRAIN_VAL_PATH = DATA_ROOT / "train_val_list.txt"
TEST_PATH = DATA_ROOT / "test_list.txt"

EXPECTED_IMAGES = 112120
EXPECTED_SHARDS = 113

NIH_LABELS = (
    ("Atelectasis", "target_atelectasis"),
    ("Cardiomegaly", "target_cardiomegaly"),
    ("Effusion", "target_effusion"),
    ("Infiltration", "target_infiltration"),
    ("Mass", "target_mass"),
    ("Nodule", "target_nodule"),
    ("Pneumonia", "target_pneumonia"),
    ("Pneumothorax", "target_pneumothorax"),
    ("Consolidation", "target_consolidation"),
    ("Edema", "target_edema"),
    ("Emphysema", "target_emphysema"),
    ("Fibrosis", "target_fibrosis"),
    ("Pleural_Thickening", "target_pleural_thickening"),
    ("Hernia", "target_hernia"),
)

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


def build_shard_index():
    shards = sorted(SHARD_ROOT.glob("*.tar"))

    require(
        len(shards) == EXPECTED_SHARDS,
        f"Expected {EXPECTED_SHARDS} TAR shards, found {len(shards)}",
    )

    records = []
    seen = set()

    for index, shard in enumerate(shards, start=1):
        print(
            f"Indexing shard {index:03d}/{len(shards):03d}: {shard.name}"
        )

        relative_shard = shard.relative_to(PROJECT_ROOT).as_posix()

        with tarfile.open(shard, "r") as archive:
            for member in archive:
                if not member.isfile():
                    continue

                member_name = Path(member.name).name

                require(
                    member_name not in seen,
                    f"Duplicate image across TAR shards: {member_name}",
                )

                seen.add(member_name)

                records.append(
                    {
                        "image_id": member_name,
                        "storage_type": "tar",
                        "shard_path": relative_shard,
                        "member_name": member.name,
                    }
                )

    require(
        len(records) == EXPECTED_IMAGES,
        f"Expected {EXPECTED_IMAGES} archive images, found {len(records)}",
    )

    return pd.DataFrame.from_records(records)


def build_source_split():
    train_val_images = load_image_list(TRAIN_VAL_PATH)
    test_images = load_image_list(TEST_PATH)

    train_val_set = set(train_val_images)
    test_set = set(test_images)

    require(
        len(train_val_images) == len(train_val_set),
        "Duplicate image in train_val_list.txt",
    )

    require(
        len(test_images) == len(test_set),
        "Duplicate image in test_list.txt",
    )

    require(
        not train_val_set.intersection(test_set),
        "Image overlap between NIH train_val and test lists",
    )

    split_map = {
        image_id: "train_val"
        for image_id in train_val_images
    }

    split_map.update(
        {
            image_id: "test"
            for image_id in test_images
        }
    )

    require(
        len(split_map) == EXPECTED_IMAGES,
        f"Expected {EXPECTED_IMAGES} split assignments, found {len(split_map)}",
    )

    return split_map


def encode_labels(frame):
    label_sets = frame["finding_labels_raw"].fillna("").map(
        lambda value: {
            label.strip()
            for label in str(value).split("|")
            if label.strip()
        }
    )

    known_labels = {
        raw_label
        for raw_label, _ in NIH_LABELS
    } | {"No Finding"}

    observed_labels = set().union(*label_sets.tolist())

    unexpected = sorted(
        observed_labels - known_labels
    )

    require(
        not unexpected,
        f"Unexpected NIH labels: {unexpected}",
    )

    for raw_label, target_column in NIH_LABELS:
        frame[target_column] = label_sets.map(
            lambda labels: int(raw_label in labels)
        ).astype("int8")

    frame["target_no_finding"] = label_sets.map(
        lambda labels: int("No Finding" in labels)
    ).astype("int8")

    disease_columns = [
        target_column
        for _, target_column in NIH_LABELS
    ]

    conflicting_no_finding = (
        (frame["target_no_finding"] == 1)
        & (frame[disease_columns].sum(axis=1) > 0)
    )

    require(
        not conflicting_no_finding.any(),
        (
            "Found rows containing No Finding together with "
            f"disease labels: {int(conflicting_no_finding.sum())}"
        ),
    )

    return frame


def build_manifest():
    metadata = pd.read_csv(METADATA_PATH)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in metadata.columns
    ]

    require(
        not missing_columns,
        f"Missing required metadata columns: {missing_columns}",
    )

    require(
        len(metadata) == EXPECTED_IMAGES,
        f"Expected {EXPECTED_IMAGES} metadata rows, found {len(metadata)}",
    )

    require(
        metadata["Image Index"].is_unique,
        "Image Index is not unique",
    )

    metadata = metadata[
        list(REQUIRED_COLUMNS)
    ].copy()

    metadata["image_id"] = metadata[
        "Image Index"
    ].astype(str)

    metadata["patient_id"] = metadata[
        "Patient ID"
    ].astype(str)

    metadata["source_split"] = metadata[
        "image_id"
    ].map(build_source_split())

    require(
        metadata["source_split"].notna().all(),
        "At least one image has no official NIH split assignment",
    )

    test_mask = metadata["source_split"].eq("test")

    metadata["split"] = ""
    metadata.loc[test_mask, "split"] = "test"

    metadata["dataset"] = "nih_chestxray14"
    metadata["view_position"] = metadata[
        "View Position"
    ].astype(str)

    metadata["patient_age"] = pd.to_numeric(
        metadata["Patient Age"],
        errors="raise",
    )

    metadata["patient_gender"] = metadata[
        "Patient Gender"
    ].astype(str)

    metadata["finding_labels_raw"] = metadata[
        "Finding Labels"
    ].astype(str)

    metadata["included"] = True
    metadata["exclusion_reason"] = ""

    require(
        set(metadata["view_position"].unique()) <= {"AP", "PA"},
        "Unexpected NIH view position found",
    )

    manifest = metadata[
        [
            "dataset",
            "patient_id",
            "image_id",
            "source_split",
            "split",
            "view_position",
            "patient_age",
            "patient_gender",
            "finding_labels_raw",
            "included",
            "exclusion_reason",
        ]
    ].copy()

    manifest = encode_labels(manifest)

    shard_index = build_shard_index()

    manifest = manifest.merge(
        shard_index,
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    require(
        manifest["shard_path"].notna().all(),
        "At least one metadata image was not found in TAR storage",
    )

    require(
        len(manifest) == EXPECTED_IMAGES,
        f"Expected {EXPECTED_IMAGES} manifest rows, found {len(manifest)}",
    )

    require(
        manifest["image_id"].is_unique,
        "Manifest image_id is not unique",
    )

    train_val_patients = set(
        manifest.loc[
            manifest["source_split"].eq("train_val"),
            "patient_id",
        ]
    )

    test_patients = set(
        manifest.loc[
            manifest["source_split"].eq("test"),
            "patient_id",
        ]
    )

    require(
        not train_val_patients.intersection(test_patients),
        "Patient leakage between official NIH train_val and test splits",
    )

    ordered_columns = [
        "dataset",
        "patient_id",
        "image_id",
        "source_split",
        "split",
        "view_position",
        "patient_age",
        "patient_gender",
        "finding_labels_raw",
        "storage_type",
        "shard_path",
        "member_name",
        "included",
        "exclusion_reason",
        *[
            target_column
            for _, target_column in NIH_LABELS
        ],
        "target_no_finding",
    ]

    return manifest[ordered_columns]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output

    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    manifest = build_manifest()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    manifest.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(output_path)

    disease_columns = [
        target_column
        for _, target_column in NIH_LABELS
    ]

    print()
    print("NIH manifest built successfully.")
    print(f"Output: {output_path}")
    print(f"Rows: {len(manifest)}")
    print(f"Patients: {manifest['patient_id'].nunique()}")
    print(
        "Official train_val images: "
        f"{int(manifest['source_split'].eq('train_val').sum())}"
    )
    print(
        "Official test images: "
        f"{int(manifest['source_split'].eq('test').sum())}"
    )
    print(
        "Official train_val patients: "
        f"{manifest.loc[manifest['source_split'].eq('train_val'), 'patient_id'].nunique()}"
    )
    print(
        "Official test patients: "
        f"{manifest.loc[manifest['source_split'].eq('test'), 'patient_id'].nunique()}"
    )
    print(f"Disease target columns: {len(disease_columns)}")
    print(
        "Rows with No Finding: "
        f"{int(manifest['target_no_finding'].sum())}"
    )


if __name__ == "__main__":
    main()