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

## Usage

Celltype annotation from a reference dataset. There is a example human liver dataset in Training_data.zip. adata is a raw (not logged or normalized) adata object containing single cell data. The gene names in adata.var must have some overlap.
```
import celltype_predict_ACTINN as ctp

train_adata = 'Training_data/GSE185477_human_liver_all_good.h5ad'
output_path = '/path/to/output_dir'

 adata, params = ctp.actinn_predict.celltype_predict_actinn(adata, train_adata,output_path ,train_label_name= 'Manual_Annotation',output_label_name='celltype_pred', output_h5ad=False)
 ```