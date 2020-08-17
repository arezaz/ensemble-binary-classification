# An Ensemble Learning Approach to Binary Classification

Dependencies:
- [sklearn '0.23.2']
- [xgboost '1.1.1']
- [lightgbm '2.3.1']
- [imblearn '0.7.0']
- [hyperopt '0.2.4']

### Overview

Binary classification using a s soft-voting ensemble meta-classifier.
The workflow trains multiple iterations of XGBoost and LightBGM on the data and optimally tunes their hyperparameters.
Data sampling is also implemented for the imbalanced dataset scenario.
The main workflow can be found in `main.py`.
All functions used in the main flow can be found in `utils.py`.
After running multiple epochs of the flow use `EpochsAnalysis.ipynb` to compare and select the best epoch.

![Workflow](https://github.com/arezaz/meta-binary-classification/blob/master/Data/pipeline.PNG)


Alireza Rezazadeh  
Spring 2020  
alr.rezazadeh@gmail.com
