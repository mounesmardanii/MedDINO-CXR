# CXR-ShiftBench Protocol

## Title

CXR-ShiftBench: Label-Efficient Medical Foundation Models Under Cross-Institutional Domain Shift

## Objective

The primary objective is to evaluate how label-efficient chest X-ray adaptation strategies behave within a large real-world source dataset and how well source-trained models retain performance under cross-institutional domain shift.

The benchmark will compare frozen foundation-model representations, foundation-model fine-tuning, and supervised CNN fine-tuning under progressively increasing labeled-data availability.

## Source Domain

NIH ChestX-ray14 will be the official source-domain dataset.

The source dataset contains 112,120 frontal chest radiographs from 30,805 patients.

The official NIH split files will be preserved:

- train_val_list.txt
- test_list.txt

The official train/val pool contains 86,524 images from 28,008 patients.

The official test split contains 25,596 images from 2,797 patients.

The source-data audit confirmed zero image overlap and zero patient overlap between the official train/val pool and official test split.

## Source Image Scope

All NIH ChestX-ray14 images in the audited release are frontal radiographs.

Observed view positions:

- AP
- PA

No unexpected view positions were found during metadata audit.

Images are stored as PNG files and will be read directly from validated TAR shards without extracting the full dataset to disk.

## Source Storage

The local source dataset consists of 113 TAR shards containing all 112,120 NIH images.

The complete shard set occupies approximately 42.15 GB.

The integrity audit confirmed:

- 113 expected shards
- 112,120 total archive images
- 112,120 unique archive images
- 112,120 metadata images
- 0 duplicate images
- 0 missing images
- 0 extra images

Raw images and large generated caches will remain excluded from Git.

## Models

### DINOv2 Frozen Linear Probe

DINOv2 ViT-S/14 will be used as a frozen feature extractor.

Only the multilabel classification head will be trained.

This arm will represent parameter-efficient adaptation using fixed foundation-model representations.

### DINOv2 Fine-Tuning

DINOv2 ViT-S/14 will be adapted to NIH chest X-ray classification through fine-tuning.

The exact trainable layers, optimizer, learning-rate schedule, regularization, batch size, stopping rule, and augmentation strategy will be frozen before official fine-tuning experiments begin.

### ResNet18 Baseline

An ImageNet-pretrained ResNet18 will be fine-tuned end-to-end as the supervised CNN baseline.

The model will use the same source-domain train, validation, test, label-fraction, and seed protocol used by the DINOv2 experiments.

## NIH Label Space

The source-domain benchmark will use the 14 NIH ChestX-ray14 findings:

- Atelectasis
- Cardiomegaly
- Effusion
- Infiltration
- Mass
- Nodule
- Pneumonia
- Pneumothorax
- Consolidation
- Edema
- Emphysema
- Fibrosis
- Pleural Thickening
- Hernia

No Finding will be retained as an auxiliary source label but excluded from the 14-finding macro-average disease metric.

## NIH Label Encoding

NIH source labels are provided as pipe-separated positive findings.

For the 14 source findings:

- listed finding = 1
- unlisted finding = 0

No uncertainty category is present in the NIH source labels used by this benchmark.

The original Finding Labels string will be retained in the source manifest for auditing.

## Source Validation Split

The official NIH train_val population will be divided into source training and source validation partitions at the patient level.

The validation-construction algorithm will be frozen before model training.

The split builder will preserve patient separation and will aim to retain multilabel prevalence across the training and validation populations.

The official NIH test split will not participate in validation construction.

## Label-Efficiency Regimes

Official source-domain labeled-data fractions:

- 1%
- 5%
- 10%
- 25%
- 100%

Label-efficiency subsets will be selected at the patient level from the frozen source training partition.

Official seeds:

- 42
- 47
- 52

Subsets within each seed will be nested:

1% ⊂ 5% ⊂ 10% ⊂ 25% ⊂ 100%

All eligible images belonging to a selected training patient may enter that patient's subset.

The 100% condition will use the complete frozen source training population.

Multiple training seeds will still be used at 100% to quantify training stochasticity.

## Source Model Selection

Model selection will use only the frozen NIH validation partition.

The primary checkpoint-selection metric will be:

Validation Macro ROC-AUC

Checkpoint selection will never use the NIH test split or any external dataset.

## Internal Evaluation

Selected checkpoints will be evaluated on the untouched official NIH test split.

The source-domain internal benchmark will report performance across all 14 NIH findings.

Primary internal metric:

Macro ROC-AUC

Secondary internal metrics:

- Macro Average Precision
- Micro ROC-AUC
- Micro Average Precision
- Per-class ROC-AUC
- Per-class Average Precision

## External Domain

The external cross-institutional dataset will be finalized in T15 after access, metadata, annotation semantics, and label compatibility are verified.

The external dataset must satisfy the following requirements:

- practically accessible under its applicable license
- institutionally distinct from NIH
- chest radiographs suitable for image classification
- patient-level identifiers or defensible evaluation units when available
- sufficient overlap with NIH findings for a predefined common label space
- label provenance documented well enough for valid interpretation

The external dataset will not be used for source training, source checkpoint selection, or source hyperparameter optimization.

## Cross-Dataset Label Harmonization

The final common cross-domain label set will be frozen only after the external dataset is selected and audited.

The mapping will be conservative.

Labels will be treated as equivalent only when their clinical meaning and annotation definitions are sufficiently compatible.

The NIH 14-label source benchmark will remain available independently of the common cross-domain subset.

Cross-domain comparisons will use only the frozen common label set.

## External Evaluation

Frozen NIH-trained checkpoints will be evaluated on the external institutional dataset without external-domain fine-tuning in the primary domain-shift benchmark.

Internal NIH and external performance will always be reported separately.

External performance will not influence source-model selection.

## Domain-Shift Definition

For a metric where larger values are better:

Delta Shift = Internal Metric - External Metric

A larger positive Delta Shift indicates greater measured degradation under the evaluated institutional shift.

A smaller positive Delta Shift indicates greater retention of source-domain performance.

Negative values indicate higher measured external performance than internal performance for the corresponding metric.

Domain-shift values will always be interpreted alongside the absolute internal and external metrics.

## Cross-Domain Primary Metric

After the common label space is frozen, the primary cross-domain predictive metric will be:

Macro ROC-AUC across the common harmonized findings.

Secondary cross-domain metrics will include:

- Macro Average Precision
- Micro ROC-AUC
- Micro Average Precision
- Per-class ROC-AUC
- Per-class Average Precision

## Statistical Reporting

Official model experiments will use three training seeds.

Training variability will be summarized using:

- mean
- sample standard deviation

Seed-level results will remain available.

Evaluation uncertainty will additionally be estimated using 95% bootstrap confidence intervals.

Bootstrap resampling will use the patient as the preferred resampling unit when patient identifiers are available.

## Calibration

Calibration will be evaluated separately on the internal and external domains.

Planned analyses include:

- Expected Calibration Error
- Brier Score
- reliability diagrams
- per-class calibration where meaningful
- calibration degradation under domain shift

Any calibration transformation will be fitted only on permitted source validation data.

External evaluation data will not be used to fit calibration parameters.

## Selective Prediction

Selective prediction will evaluate whether model uncertainty can support abstention or case deferral.

Planned analyses include:

- risk-coverage curves
- performance at predefined coverage levels
- internal versus external selective-prediction degradation

## OOD and Domain-Shift Detection

NIH ChestX-ray14 will represent the source institutional distribution.

The selected external dataset will represent the external institutional distribution.

Planned signals may include:

- predictive confidence
- predictive entropy
- representation-space distance
- embedding-based scores
- other predefined model-derived uncertainty measures

These analyses will be interpreted as distribution-shift detection rather than clinical abnormality detection.

## Subgroup Analysis

NIH metadata currently supports candidate subgroup variables including:

- patient age
- patient gender
- AP versus PA view position

Cross-domain subgroup analyses will be performed only when the external dataset provides compatible metadata definitions.

Subgroup sample sizes will always be reported.

## Reproducibility

The following will be version-controlled:

- protocol documents
- manifest specifications
- acquisition scripts
- metadata audit scripts
- shard integrity audit scripts
- split-generation scripts
- patient-level subset indices when shareable
- random seeds
- model configurations
- training configurations
- checkpoint-selection criteria
- evaluation scripts
- aggregate result tables
- figures

Raw medical images, large TAR shards, model checkpoints, embedding caches, and other large derived artifacts will not be committed to Git.

## Data Leakage Prevention

Patient-level separation will be enforced between source training, source validation, and source test partitions.

Label-efficiency sampling will occur only inside the frozen source training population.

The NIH official test split will remain untouched until source training and model selection are complete.

The external dataset will never influence:

- source training
- source validation construction
- checkpoint selection
- source hyperparameter selection
- source augmentation tuning
- common-label decisions made after external performance inspection

## Compute Strategy

Metadata processing, split construction, statistics, aggregation, calibration analysis, bootstrap analysis, and most representation-level analyses may be performed locally.

Large-scale DINOv2 embedding extraction and neural-network fine-tuning may be executed using GPU infrastructure such as Google Colab when local execution becomes inefficient.

The same frozen manifests and configurations will be used regardless of execution environment.

## Interpretation Constraints

Frozen DINOv2 linear probing and end-to-end ResNet18 fine-tuning represent different adaptation regimes.

Their comparison will not be described as a strictly controlled architectural comparison.

The DINOv2 fine-tuning arm is included to better separate foundation-model representation quality from adaptation capacity.

Performance on retrospective public datasets will not be interpreted as evidence of clinical deployment readiness.

Evaluation on one external dataset will not be described as general clinical robustness.

## Protocol Freeze

Before official NIH model training begins, the following must be frozen:

- source manifest schema
- patient-level train/validation construction
- source label ordering
- source target encoding
- label-efficiency subset algorithm
- official fractions
- official seeds
- preprocessing
- model configurations
- checkpoint-selection criteria

Before official external evaluation begins, the following must additionally be frozen:

- external dataset
- external manifest schema
- common cross-dataset label mapping
- external label handling
- cross-domain primary metric

Any change after the corresponding freeze point will be documented explicitly as an amendment or post hoc analysis.
