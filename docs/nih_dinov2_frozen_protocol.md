# NIH Frozen DINOv2 Protocol

## Purpose

This stage evaluates the label efficiency of frozen DINOv2 representations on NIH ChestX-ray14 before any end-to-end DINOv2 fine-tuning or external evaluation.

The official NIH test split remains locked for later final evaluation.

## Source Dataset

Dataset: NIH ChestX-ray14

Total images: 112120

Official source split:

| Split | Images |
|---|---:|
| Official train/validation pool | 86524 |
| Official test | 25596 |

A patient-level split was created only within the official NIH train/validation pool.

Frozen development split:

| Split | Patients | Images |
|---|---:|---:|
| Train | 25207 | 77911 |
| Validation | 2801 | 8613 |
| Test | 2797 | 25596 |

There is no patient overlap between train, validation, and test.

## NIH Storage Freeze

Hugging Face repository:

`yeigen/nih-chest-xray`

Frozen revision:

`c0b558ec72f1ce434f7355f0f5cf914e2d62c60a`

The NIH mirror contains 113 TAR shards covering all 112120 images.

The storage shard names are not used as experimental train/test assignments. Experimental splits are defined exclusively by the frozen NIH manifest.

## DINOv2 Freeze

Model:

`dinov2_vits14`

Feature dimension:

`384`

DINOv2 repository revision:

`facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8`

The DINOv2 encoder is fully frozen during this stage.

No DINOv2 backbone parameters are updated.

## Image Preprocessing

Images are converted to RGB and processed using the deterministic evaluation transform:

- Resize to 224 x 224
- Convert to tensor
- ImageNet normalization

ImageNet mean:

`0.485, 0.456, 0.406`

ImageNet standard deviation:

`0.229, 0.224, 0.225`

No stochastic training augmentation is used during embedding extraction.

## Embedding Extraction

Frozen DINOv2 embeddings were extracted once for all 112120 NIH images.

Each image produces a 384-dimensional float32 representation.

Outputs are stored as 113 shard-level NPZ files.

Each output contains:

- embeddings
- 14 NIH target labels
- image ID
- patient ID
- frozen split
- view position

The extraction artifacts are excluded from Git because they are derived data.

Embedding extraction itself is label-independent.

Although embeddings were extracted for the official NIH test images for computational efficiency, no NIH test metrics, test predictions, model selection, hyperparameter selection, or threshold selection are performed during this stage.

The official NIH test set remains locked until the final source evaluation stage.

## NIH Targets

The 14 primary findings are:

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

`No Finding` is retained in the NIH manifest as auxiliary metadata but is not a primary classification target.

NIH ChestX-ray14 labels are report-derived weak labels and should not be interpreted as expert-adjudicated diagnostic ground truth.

## Label-Efficiency Design

Frozen source label fractions:

| Fraction | Patients |
|---|---:|
| 1% | 252 |
| 5% | 1260 |
| 10% | 2521 |
| 25% | 6302 |
| 100% | 25207 |

Model experiment seeds:

`42, 47, 52`

Subsets are patient-level, multilabel-stratified, and nested within each seed.

When a patient is selected, all training images belonging to that patient are included.

The patient-level targets are used only for subset stratification.

Linear-probe training uses the original image-level NIH labels.

The 1% experiment is retained as an extreme low-label stress test and must be interpreted cautiously because very rare findings may contain only one positive training patient.

## Linear Probe

Classifier:

`Linear(384, 14)`

Loss:

`BCEWithLogitsLoss`

Class imbalance is handled using positive-class weights calculated independently from each labeled training subset.

Positive weights are calculated only from the labels available within that subset.

Maximum positive-class weight:

`20`

This prevents information from unlabeled training patients from entering the loss construction.

Optimizer:

`AdamW`

Frozen training protocol:

| Parameter | Value |
|---|---:|
| Maximum epochs | 100 |
| Batch size | 2048 |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Early-stopping patience | 5 |
| Minimum improvement | 1e-4 |

Model selection uses NIH validation Macro ROC-AUC only.

The NIH test set and X-Raydar external dataset are not used for model selection.

## Validation Metrics

Primary validation metric:

`Macro ROC-AUC`

Secondary metrics:

- Macro Average Precision
- Micro ROC-AUC
- Micro Average Precision
- Per-class ROC-AUC
- Per-class Average Precision

Reported label-efficiency results aggregate the best validation checkpoint across three seeds.

## Frozen DINOv2 Validation Results

| Labels | Macro ROC-AUC mean +/- SD | Macro AP mean +/- SD |
|---:|---:|---:|
| 1% | 0.620452 +/- 0.014776 | 0.073434 +/- 0.001570 |
| 5% | 0.680287 +/- 0.007135 | 0.090529 +/- 0.001162 |
| 10% | 0.709254 +/- 0.009338 | 0.100963 +/- 0.000919 |
| 25% | 0.730674 +/- 0.002273 | 0.114450 +/- 0.001529 |
| 100% | 0.754646 +/- 0.000296 | 0.126244 +/- 0.000601 |

These are source-domain validation results only.

They demonstrate increasing predictive performance as labeled NIH training data increase, while the 25% setting retains much of the 100% frozen-linear-probe performance.

These results do not establish test performance, external generalization, clinical utility, or superiority over other adaptation strategies.

## Reproducibility Scripts

Embedding extraction:

`scripts/extract_nih_dinov2_embeddings.py`

Linear-probe data preparation:

`scripts/prepare_nih_dinov2_linear_probe_data.py`

Linear-probe training:

`scripts/train_nih_dinov2_linear_probe.py`

Systematic experiment runner:

`scripts/run_nih_dinov2_label_efficiency.py`

Validation summarization:

`scripts/summarize_nih_dinov2_label_efficiency.py`

Patient-level label subset construction:

`scripts/build_nih_label_efficiency_subsets.py`

## Evaluation Boundary

This stage ends after source-domain NIH validation.

The following remain intentionally deferred:

- end-to-end ResNet18 source baseline
- DINOv2 fine-tuning
- frozen NIH test evaluation
- X-Raydar cross-institutional external evaluation
- calibration and reliability analysis
- subgroup analysis
- domain-shift and OOD analysis

No claims about final model superiority or clinical performance should be made from this stage alone.