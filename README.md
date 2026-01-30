# ADMET properties prediction
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

- [ADMET properties prediction](#admet-properties-prediction)
  - [About](#about)
  - [Customized Usage](#customized-usage)
  - [Citation](#citation)

`predictADMET` is a project to predict 5 different properties (membrane permeability, solubility, protein plasma binding, genotoxicity, cardiotoxicity) of molecules and get explanations (why a molecule is predicted to have a property).

## About 
Prediction of ADMET properties was done with the use of three types of ML regression models: 6 shallow models, 3 ensembles of shallow models, and 2 deep learning models, more precisely:
* Ridge Regression (`RR`),
* Random Forest Regressor (`RF`), 
* Histogram-based Gradient Boosting Regression Tree (`HistGrad`),
* Support Vector Regression (`SVR`),
* Extreme Gradient Boosting (`XGBoost`),
* Light Gradient Boosting Machine (`LGBM`),
* Ensemble models,
* Fully Connected Neural Networks (`FCNNs`),
* Graph Convolutional Neural Networks (`GCNNs`), 

and representations: `MACCSFp`, `PubchemFp`, `KRFp`, `molecular graphs`.

More details and results available at:

## Customized Usage

Create conda environment and install packages:
```bash
conda env create -f env.yml
conda activate admet_print
```

Download data used in experiments from [link](https://drive.google.com/drive/u/0/folders/1NYHdDnOjMdqqBhDmRRRQT4mok3xtXUH2) or add your own data to the `data` directory.

Run training:
```bash
bash scripts/run.sh
```

## Citation

Please cite [Jamrozik et al.](https://pubs.acs.org/doi/10.1021/acs.jcim.3c02038)

```bibtex
@article{jamrozik2024admet,
  title={ADMET-PrInt: Evaluation of ADMET properties: Prediction and interpretation},
  author={Jamrozik, Ewelina and Smieja, Marek and Podlewska, Sabina},
  journal={Journal of Chemical Information and Modeling},
  volume={64},
  number={5},
  pages={1425--1432},
  year={2024},
  publisher={ACS Publications}
}
```


