# An Ensemble Learning Approach to Binary Classification

### Overview

Binary classification using a s soft-voting ensemble meta-classifier.
The workflow trains multiple iterations of XGBoost and LightBGM on the data and optimally tunes their hyperparameters.
Data sampling is also implemented for the imbalanced dataset scenario.
The main workflow can be found in `main.py`.
All functions used in the main flow can be found in `utils.py`.
After running multiple epochs of the flow use `EpochsAnalysis.ipynb` to compare and select the best epoch.

![Image of Workflow](https://github.com/arezaz/meta-binary-classification/blob/master/Data/pipeline.PNG)

### Citation
If you find this repository useful in your research, please consider citing the paper.

```
@article{rezazadeh2020generalized,
  title={A Generalized Flow for B2B Sales Predictive Modeling: An Azure Machine-Learning Approach},
  author={Rezazadeh, Alireza},
  journal={Forecasting},
  volume={2},
  number={3},
  pages={267--283},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
Alireza Rezazadeh  
Summer 2020  
alr.rezazadeh@gmail.com
