# meta-binary-classification

Dependencies
- [sklearn]
- [xgboost]
- [lightgbm]
- [hyperopt]
- [pandas]
- [tqdm]
- [pickle]


### Overview

Binary classification using an ensemble meta-classifier.
The workflow trains multiple iterations of XGBoost and LightBGM on the data and optimally tunes their hyperparameters.
Data sampling is also implemented for the imbalanced dataset scenario.
The main workflow can be found in `main.py`.
All functions used in the main flow can be found in `utils.py`.
The function `gen_data()` needs to be modified according to dataset to encode all categorical features.

### Reference

A Generalized Flow for B2B Sales Predictive Modeling: An Azure Machine-Learning Approach [paper](https://www.mdpi.com/2571-9394/2/3/15/htm).
```
@article{rezazadeh_2020,
         title={A Generalized Flow for B2B Sales Predictive Modeling: An Azure Machine-Learning Approach},
         volume={2}, DOI={10.3390/forecast2030015}, number={3},
         journal={Forecasting},
         author={Rezazadeh, Alireza},
         year={2020},
         pages={267–283}}
```

Alireza Rezazadeh  
Spring 2020  
rezaz003@umn.edu
