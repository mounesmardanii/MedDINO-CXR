# CXR-ShiftBench Manifest Specification

## Purpose

This document defines the canonical manifest structure for the NIH-based CXR-ShiftBench source pipeline and the requirements for later external-domain harmonization.

All official experiments will consume frozen processed manifests rather than reading raw NIH metadata directly.

## Source Dataset

Dataset identifier:

nih_chestxray14

Expected source inventory:

- 112,120 images
- 30,805 patients
- 113 validated TAR shards
- AP and PA frontal views only

## Manifest Unit

The primary manifest unit is one chest radiograph.

Each row corresponds to one image.

NIH does not provide a study identifier in the source metadata used by this benchmark, so the source manifest will preserve patient and image identifiers without inventing a study identifier.

## Canonical Source Columns

Each NIH source manifest row will include at least:

- dataset
- patient_id
- image_id
- source_split
- split
- view_position
- patient_age
- patient_gender
- finding_labels_raw
- storage_type
- shard_path
- member_name
- included
- exclusion_reason

## Dataset Field

For NIH rows:

dataset = nih_chestxray14

External datasets will use separate dataset identifiers.

Patient identifiers will never be assumed to be globally unique across datasets.

## Patient Identifier

patient_id will be derived from the NIH Patient ID field.

Patient-level separation will be enforced across train, validation, and test partitions.

## Image Identifier

image_id will equal the NIH Image Index value.

Example:

00000001_000.png

image_id must be unique within the NIH manifest.

## TAR Storage

The NIH images are stored in validated TAR shards rather than extracted into an image directory.

Canonical storage fields:

storage_type = tar

shard_path = relative path to the TAR shard

member_name = filename of the PNG member inside that TAR

Example:

shard_path = data/raw/nih/hf_shards/data/train-00000.tar

member_name = 00000001_000.png

Absolute machine-specific paths will not be committed to Git.

## Shard Semantics

The Hugging Face filenames such as train-00000.tar are storage shard names only.

They do not define the official NIH training split.

Official source split membership must always come from:

- train_val_list.txt
- test_list.txt

No training decision may be inferred from the TAR shard filename.

## Source Split

Raw NIH source split values will be normalized to:

- train_val
- test

The processed experiment split will later use:

- train
- validate
- test

The official test membership will be preserved exactly.

The train and validate partitions will be derived only from the official train_val population.

## View Position

Allowed source values:

- AP
- PA

The metadata audit found:

- PA: 67,310 images
- AP: 44,810 images

No unexpected view values were observed.

## NIH Raw Metadata Mapping

From Data_Entry_2017_v2020.csv:

- Image Index
- Finding Labels
- Patient ID
- Patient Age
- Patient Gender
- View Position

Additional available metadata columns may be retained for auditing but will not automatically become model inputs.

## NIH Source Labels

The 14 source findings are:

1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural Thickening
14. Hernia

No Finding will be retained as an auxiliary label.

## Processed Label Column Names

Primary target columns:

- target_atelectasis
- target_cardiomegaly
- target_effusion
- target_infiltration
- target_mass
- target_nodule
- target_pneumonia
- target_pneumothorax
- target_consolidation
- target_edema
- target_emphysema
- target_fibrosis
- target_pleural_thickening
- target_hernia

Auxiliary column:

- target_no_finding

## NIH Label Encoding

For each of the 14 findings:

finding present in Finding Labels:
target = 1

finding absent from Finding Labels:
target = 0

No uncertainty category is used in the NIH source labels.

The original Finding Labels text will be preserved in finding_labels_raw.

## Label Ordering

The canonical NIH output order must remain identical across:

- manifests
- datasets
- model logits
- loss computation
- prediction exports
- metric computation
- result tables

The canonical order is the 14-label order defined in this document.

## Validation Construction

The official train_val population will be split at the patient level.

The validation builder must guarantee:

train patients ∩ validation patients = empty

test patients remain exactly the official NIH test patients.

The split builder will aim to preserve multilabel prevalence across train and validation.

The final algorithm and validation size will be frozen before official training.

## Label-Efficiency Sampling

Label-efficiency subsets will operate on unique patients in the frozen training partition.

Official fractions:

- 1%
- 5%
- 10%
- 25%
- 100%

Official seeds:

- 42
- 47
- 52

Within each seed:

1% ⊂ 5% ⊂ 10% ⊂ 25% ⊂ 100%

All images belonging to a selected training patient may enter the corresponding subset.

## Source Reproducibility Files

Planned generated files:

data/manifests/nih_train.csv

data/manifests/nih_validate.csv

data/manifests/nih_test.csv

data/subsets/seed_42/fraction_001_patients.csv

data/subsets/seed_42/fraction_005_patients.csv

data/subsets/seed_42/fraction_010_patients.csv

data/subsets/seed_42/fraction_025_patients.csv

data/subsets/seed_42/fraction_100_patients.csv

Equivalent subset files will be generated for seeds 47 and 52.

## Image Availability Audit

For TAR-backed storage, availability means:

- referenced shard exists
- TAR can be opened
- member_name exists in the referenced shard

Training code must not silently skip missing images.

Any missing required image will stop the official pipeline.

## Duplicate Handling

image_id must be unique.

The full shard integrity audit already confirmed:

- 112,120 total images
- 112,120 unique images
- 0 duplicates
- 0 missing images
- 0 extra images

Future manifest builders must preserve these invariants.

## Integrity Checks

The source manifest pipeline must verify:

- required metadata columns exist
- Image Index is unique
- all official split images exist in metadata
- official test membership is preserved
- train_val/test image overlap is zero
- train_val/test patient overlap is zero
- view_position is AP or PA
- target columns are binary
- target ordering is fixed
- each included image resolves to exactly one TAR shard and member
- no image is assigned to more than one experiment split

## Source Manifest Statistics

For each frozen source split, summary statistics will include:

- number of patients
- number of images
- AP count
- PA count
- age summary
- gender counts
- positive count per finding
- prevalence per finding
- No Finding count

Statistics will be generated separately for:

- train
- validate
- test

## External Manifest

The external dataset will be selected and audited in T15.

The external manifest will use a compatible canonical structure where possible, including:

- dataset
- patient_id
- image_id
- split
- view_position when available
- storage locator
- source labels
- harmonized common-label targets
- validity masks where required by the external annotation semantics

No external field mapping will be assumed before the selected dataset is inspected.

## Common Cross-Domain Label Space

The common NIH/external label set will be defined only after the external dataset is selected.

A mapping table will explicitly document:

- NIH label
- external label
- canonical common label
- mapping status
- rationale
- inclusion in the primary cross-domain metric

Only clinically and semantically defensible mappings will enter the primary cross-domain benchmark.

## External Label Validity

The canonical external schema may include per-label validity masks if the external dataset contains:

- uncertain labels
- missing labels
- partially annotated classes
- annotation policies that prevent treating every unlisted label as negative

NIH source rows do not require uncertainty masks for their native 14-label benchmark.

## Data Leakage Prevention

Source validation construction will operate only inside the official NIH train_val population.

The official NIH test population will remain untouched.

External data will not enter:

- source train manifests
- source validation manifests
- label-efficiency patient subsets
- source checkpoint selection

## Versioning

Version-controlled artifacts will include:

- manifest specification
- manifest builder
- split builder
- subset builder
- aggregate manifest statistics
- audit scripts

Raw images and TAR shards will not be committed.

Generated files containing dataset identifiers will be committed only when permitted and appropriate under the applicable dataset terms.

## Protocol Freeze Requirement

Before official NIH training, the following must be frozen:

- source manifest schema
- TAR member resolution
- label ordering
- source target encoding
- patient-level validation construction
- subset construction
- official fractions
- official seeds

Before official external evaluation, the following must additionally be frozen:

- external dataset
- external manifest schema
- common label mapping
- external label validity policy
- cross-domain primary metric
