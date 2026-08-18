# NIH ChestX-ray14 Split Freeze

## Status

This document freezes the official NIH ChestX-ray14 source-domain train, validation, and test partition used by CXR-ShiftBench.

Freeze status: FINAL

## Source Population

Dataset: NIH ChestX-ray14

Total images: 112,120

Total patients: 30,805

Official NIH train_val pool:

- 86,524 images
- 28,008 patients

Official NIH test split:

- 25,596 images
- 2,797 patients

The official NIH test split is preserved exactly and is not involved in validation construction.

## Validation Construction

Validation fraction: 10%

Split level: patient

Patient target aggregation: maximum across all images belonging to the patient

Stratification method: MultilabelStratifiedShuffleSplit

Split seed: 2026

The 14 NIH disease targets are used for stratification.

All images from a patient are assigned to exactly one final partition.

## Frozen Final Split

| Split | Patients | Images |
|---|---:|---:|
| Train | 25,207 | 77,911 |
| Validate | 2,801 | 8,613 |
| Test | 2,797 | 25,596 |
| Total | 30,805 | 112,120 |

## Leakage Audit

Patient overlap between train and validate: 0

Patient overlap between train and test: 0

Patient overlap between validate and test: 0

The official NIH test image membership remains unchanged.

## Image-Level Prevalence Audit

| Label | Train prevalence | Validation prevalence | Absolute gap | Train positives | Validation positives |
|---|---:|---:|---:|---:|---:|
| Atelectasis | 0.09585 | 0.09428 | 0.00158 | 7,468 | 812 |
| Cardiomegaly | 0.01983 | 0.01881 | 0.00102 | 1,545 | 162 |
| Effusion | 0.09946 | 0.10565 | 0.00619 | 7,749 | 910 |
| Infiltration | 0.15815 | 0.16951 | 0.01136 | 12,322 | 1,460 |
| Mass | 0.04662 | 0.04667 | 0.00006 | 3,632 | 402 |
| Nodule | 0.05481 | 0.05085 | 0.00395 | 4,270 | 438 |
| Pneumonia | 0.01018 | 0.00964 | 0.00054 | 793 | 83 |
| Pneumothorax | 0.03091 | 0.02659 | 0.00432 | 2,408 | 229 |
| Consolidation | 0.03283 | 0.03413 | 0.00130 | 2,558 | 294 |
| Edema | 0.01574 | 0.01765 | 0.00191 | 1,226 | 152 |
| Emphysema | 0.01661 | 0.01498 | 0.00163 | 1,294 | 129 |
| Fibrosis | 0.01461 | 0.01312 | 0.00149 | 1,138 | 113 |
| Pleural Thickening | 0.02584 | 0.02659 | 0.00075 | 2,013 | 229 |
| Hernia | 0.00164 | 0.00151 | 0.00013 | 128 | 13 |

Maximum image-level prevalence gap: 0.01136

## Patient-Level Prevalence Audit

| Label | Train patient prevalence | Validation patient prevalence | Absolute gap | Train positive patients | Validation positive patients |
|---|---:|---:|---:|---:|---:|
| Atelectasis | 0.14932 | 0.14923 | 0.00009 | 3,764 | 418 |
| Cardiomegaly | 0.04384 | 0.04391 | 0.00008 | 1,105 | 123 |
| Effusion | 0.12504 | 0.12496 | 0.00009 | 3,152 | 350 |
| Infiltration | 0.25390 | 0.25384 | 0.00006 | 6,400 | 711 |
| Mass | 0.07549 | 0.07569 | 0.00019 | 1,903 | 212 |
| Nodule | 0.10192 | 0.10211 | 0.00019 | 2,569 | 286 |
| Pneumonia | 0.02487 | 0.02499 | 0.00012 | 627 | 70 |
| Pneumothorax | 0.03856 | 0.03856 | 0.00000 | 972 | 108 |
| Consolidation | 0.05772 | 0.05784 | 0.00011 | 1,455 | 162 |
| Edema | 0.02666 | 0.02678 | 0.00012 | 672 | 75 |
| Emphysema | 0.02721 | 0.02713 | 0.00008 | 686 | 76 |
| Fibrosis | 0.03582 | 0.03570 | 0.00012 | 903 | 100 |
| Pleural Thickening | 0.05566 | 0.05569 | 0.00004 | 1,403 | 156 |
| Hernia | 0.00365 | 0.00357 | 0.00008 | 92 | 10 |

Maximum patient-level prevalence gap: 0.00019

## Reproducibility Artifacts

The patient-level split assignment is stored in:

data/manifests/nih_split_patients.csv

The split-generation implementation is stored in:

scripts/build_nih_patient_split.py

The canonical source manifest builder is stored in:

scripts/build_nih_manifest.py

Large generated manifests are reproducible and remain excluded from Git:

- data/manifests/nih_manifest.csv
- data/manifests/nih_manifest_split.csv
- data/manifests/nih_train.csv
- data/manifests/nih_validate.csv
- data/manifests/nih_test.csv

## Freeze Rule

The following values are frozen for official NIH source experiments:

- validation fraction: 10%
- split level: patient
- split seed: 2026
- patient target aggregation: maximum across patient images
- stratification method: multilabel stratification
- train patient membership
- validation patient membership
- NIH official test membership

The split seed must not be changed to improve downstream model performance.

Any future change to this split must be documented as a protocol amendment and must not silently replace the frozen benchmark split.
