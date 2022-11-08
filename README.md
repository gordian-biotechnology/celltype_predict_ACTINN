# Celltype prediction from reference data for single cell data: celltype_predict_ACTINN

For single cell celltype annotation. Input reference dataset and dataset to label, output labeled dataset.
This package is entirely based on ACTINN written by Feiyang Ma [original code](https://github.com/mafeiyang/ACTINN.git) and the [publication](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btz592/5540320) should be cited if used. 
# Installation

```
git clone https://github.com/gordian-biotechnology/celltype_predict_ACTINN.git
cd celltype_predict_ACTINN
pip install .
```
## Requirements

1. [tensorflow](https://www.tensorflow.org/install)
2. [scanpy](https://icb-scanpy.readthedocs-hosted.com/en/latest/index.html)