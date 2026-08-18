# NIH Label-Efficiency Subset Freeze

Source split: NIH training patients only

Total training patients: 25207

Fractions:

| Fraction | Patients |
|---|---:|
| 1% | 252 |
| 5% | 1260 |
| 10% | 2521 |
| 25% | 6302 |
| 100% | 25207 |

Model experiment seeds:

42
47
52

Subsets are patient-level, multilabel-stratified, and nested within each seed.

The official NIH validation and test sets are not used to construct these subsets.

Minimum positive-patient counts across the 14 NIH targets:

| Fraction | Minimum positives |
|---|---:|
| 1% | 1 |
| 5% | 4 |
| 10% | 9 |
| 25% | 23 |
| 100% | 92 |

The 1% setting is retained as an extreme low-label stress test. Results at this fraction must be interpreted with caution because rare classes may contain very few positive training patients.

Subset assignments are stored in:

data/manifests/nih_label_efficiency_patients.csv

Generation script:

scripts/build_nih_label_efficiency_subsets.py
