import os
from os.path import join as pjoin
import pandas as pd
import pickle
import time
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from utils import gen_data, iterModel, MetaClassifier, Sampling


EPOCHS = 10 # number times that the flow will work through the entire dataset.

for i in tqdm(range(0,EPOCHS)):

    # ------------------------------ I) Load & Process Data ----------------------------------- #

    #    A) Train Set: generate train dataset fro challenge data
    PATH_TRAIN = pjoin("Data", "dataset-challenge.xlsx")
    data_dict = gen_data(PATH_TRAIN, 'train')
    X, y = [data_dict['X'], data_dict['y']]

    #    B) Test Set: generate test dataset from the challenge data (optional)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    #train_dict = {'X_train': X_train, 'y_train':y_train}
    #test_dict = {'X_test': X_test, 'y_test':y_test}

    #    C) Evaluation Set: generate evaluation dataset
    PATH_EVAL = pjoin("Data", "dataset-evaluation.xlsx")
    eval_dict = gen_data(PATH_EVAL, 'eval')
    X_eval = eval_dict['X']

    train_dict = {'X_train': X, 'y_train':y}
    test_dict = {}

    # ------------------------------ II) Sampling Imbalanced Data ----------------------------- #

    method = 'SMOTEENN' # samlpling method
    X_res, y_res = Sampling(train_dict['X_train'], train_dict['y_train'], method=method)
    train_dict = {'X_train': X_res, 'y_train':y_res}

    # --------------------------------- III) Machine Learning --------------------------------- #

    # ---- iterate classification algorithms: XGBoost, LightGBM
    max_evals = 15 # max iters for tunings hyperparameters
    XGBoost_iters = iterModel(name='XGBoost', max_evals=max_evals, train_dict=train_dict, test_dict=test_dict)
    LightGBM_iters = iterModel(name='LightGBM', max_evals=max_evals, train_dict=train_dict, test_dict=test_dict)

    # ---- Build an soft-voting ensemble meta-classification model
    Results = MetaClassifier([XGBoost_iters, LightGBM_iters], train_dict, test_dict)

    # ---------------------------------- IV) Make Prediction ---------------------------------- #

    # ---- make predictions on evaluation set
    y_pred = Results['BestModel'].predict(X_eval.drop(columns='scenario'))
    y_proba = Results['BestModel'].predict_proba(X_eval.drop(columns='scenario'))
    X_eval['prediction_score'] = y_pred
    X_eval['prediction_score_proba'] = y_proba[:,1]

    prediction_df = X_eval[['scenario', 'prediction_score','prediction_score_proba']]

    # ------------------------------------ V) Save Outputs ----------------------------------- #

    timestr = method+time.strftime("-%Y%m%d-%H%M%S") # create timestamp for saving epoch results

    PATH_SAVE = pjoin("Sandbox", "Output-"+timestr)
    if not os.path.exists(PATH_SAVE):
        os.makedirs(PATH_SAVE)

    # ---- A) save evaluation set predictions
    # --------- 1 - evaluation set prediction with submission format 
    filename = 'prediction-'+timestr+'.csv'
    prediction_df[['scenario', 'prediction_score']].to_csv(pjoin(PATH_SAVE ,filename), index=False)
    # --------- 2 -evaluation set prediction also outputing predicted probabilities
    filename = 'prediction_proba-'+timestr+'.csv'
    prediction_df[['scenario', 'prediction_score','prediction_score_proba']].to_csv(pjoin(PATH_SAVE ,filename), index=False) 

    #  ---- B) train set performance metrics summary
    filename = 'metrics-'+timestr+'.csv'
    Results['Metrics'].to_csv(pjoin(PATH_SAVE ,filename))

    # ---- C) metaclassifier performance on trainset
    filename = 'train_df_pred-'+timestr+'.csv'
    Results['train_df_pred'].to_csv(pjoin(PATH_SAVE ,filename))

    # ---- D) meta-classification model
    filename = 'MetaClassifier-'+timestr+'.pkl'
    with open(pjoin(PATH_SAVE ,filename), 'wb') as file:
        pickle.dump(Results['BestModel'], file)

