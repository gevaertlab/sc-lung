## Single-cell multi-modal analysis reveals tumor microenvironment predictive of treatment response in non-small cell lung cancer

## Before started

- Download the `sample_data` folder from: https://drive.google.com/drive/folders/1Uaql_imN9OqwGlAtYdn9R71E5OapOEkI?usp=sharingg
- Unzip and place it within this repository

## Repository structure

- sample_data: example datasets for testing and debugging the codes
- cal_spatial.py: perform spatial statistical analyses of the mIF data
- spatial_util.py: utility functions for spatial statistical analyses, visualization and group comparisons
- deseq_rank_genes.R: perform differential gene expression analysis
- geneset_rna.py: gene set enrichment analysis
- hover_net: training and evaluating scripts for nuclear segmentation and classification using histology images. The codes were adapted from Graham et al. (https://github.com/vqdang/hover_net), with my own customizations 
- sc_MTOP: codes for extracting single-cell morphological, textural and topological features from histology images. The codes were adapted from Zhao et al. Nat Comm (2023), with my own customizations for incorporating new features and additional cell types

## Environment setup 

1. To run spatial statistical analyses of the mIF data, install the scimap package: https://scimap.xyz/Getting%20Started/
2. To train a nuclear segmentation and classification model on histology images, install the system requirements for hovernet: https://github.com/vqdang/hover_net
3. To perform single-cell microenvironment analysis on histology images, install the system requirements for sc-MTOP: https://github.com/fuscc-deep-path/sc_MTOP
4. To run differential gene expression analysis, install the DESeq2 library: https://bioconductor.org/packages/release/bioc/html/DESeq2.html
5. To perform gene set enrichment analysis, install the GSEApy library: https://gseapy.readthedocs.io/en/latest/introduction.html#id1

## Running the analyses

1. Perform spatial statistical analyses of the mIF data<br>
    `python cal_spatial.py`

2. Perform gene set enrichment analysis<br>
    `Rscript deseq_rank_genes.R`<br>
    `python gene_set_rna.py`

3. Train nuclear segmentation and classification model on histology images<br>
    `python extract_patches.py`<br>
    `python hover_net/run_train.py`<br>

4. Run inference using pre-trained model: https://github.com/gevaertlab/NucSegAI 

5. Extract nuclear morphology, textural, and topological features<br>
    `python sc_MTOP/extract_features.py`











