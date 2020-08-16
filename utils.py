import pandas as pd
from os.path import join as pjoin
from tqdm import tqdm

import xgboost as xgb
import lightgbm as lgbm

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.ensemble import VotingClassifier

from hyperopt import Trials, STATUS_OK, tpe, hp, fmin


def gen_data(dataPath, set):
    """
    function to generates train, and test dataset:
    reads .xlsx file, extracts features and labels, encode the categorical feature

    Arguments:
    dataPath -- a string of dataset directory
    set -- "train", or "test"

    Return:
    data_dict -- a dictionary of features ('X') and labels ('y') (has labels if it's a train set)
    """

    # read xlsx file and extract data frame
    xl_file = pd.ExcelFile(dataPath) 
    dfs = {sheet_name: xl_file.parse(sheet_name) 
            for sheet_name in xl_file.sheet_names}
    Data_raw = dfs['out1']

    # one-hote encoding the categorical variable, XC
    XC_OneHot = pd.get_dummies(Data_raw.XC, prefix='XC')
    Data = pd.concat([Data_raw, XC_OneHot], axis=1)
    Data.drop(columns=['XC'], inplace = True)

    if set == 'train':
        X = Data.drop(columns = 'y')
        y = Data.y
        data_dict = {'X':X, 'y':y}
    else:
        X = Data
        data_dict = {'X':X}

    return data_dict

def Sampling(X,y, method):
    """
    function to sample imbalanced dataset:

    Arguments:
    X -- trainset features
    y -- trainset labels
    method -- sampling method

    Return:
    X_res -- sampled trainset features
    y_res -- sampled trainset labels
    """
    
    #Under-sampling:
    if method == 'RandomUnderSampler':
        from imblearn.under_sampling import RandomUnderSampler
        us = RandomUnderSampler()
        X_res, y_res = us.fit_resample(X, y)

    elif method == 'TomekLinks':
        from imblearn.under_sampling import TomekLinks 
        us = TomekLinks()
        X_res, y_res = us.fit_resample(X, y)

    elif method == 'OneSidedSelection':
        from imblearn.under_sampling import OneSidedSelection
        us = OneSidedSelection()
        X_res, y_res = us.fit_resample(X, y)
    
    elif method == 'NeighbourhoodCleaningRule':
        from imblearn.under_sampling import NeighbourhoodCleaningRule 
        us = NeighbourhoodCleaningRule()
        X_res, y_res = us.fit_resample(X, y)
        
    elif method == 'NearMiss':
        from imblearn.under_sampling import NearMiss 
        us = NearMiss()
        X_res, y_res = us.fit_resample(X, y)

    elif method == 'InstanceHardnessThreshold':
        from imblearn.under_sampling import InstanceHardnessThreshold 
        us = InstanceHardnessThreshold()
        X_res, y_res = us.fit_resample(X, y)

    elif method == 'AllKNN':
        from imblearn.under_sampling import AllKNN 
        us = AllKNN()
        X_res, y_res = us.fit_resample(X, y)

    elif method == 'RepeatedEditedNearestNeighbours':
        from imblearn.under_sampling import RepeatedEditedNearestNeighbours 
        us = RepeatedEditedNearestNeighbours()
        X_res, y_res = us.fit_resample(X, y)

    elif method == 'EditedNearestNeighbours':
        from imblearn.under_sampling import EditedNearestNeighbours 
        us = EditedNearestNeighbours()
        X_res, y_res = us.fit_resample(X, y)

    elif method == 'CondensedNearestNeighbour':
        from imblearn.under_sampling import CondensedNearestNeighbour 
        us = CondensedNearestNeighbour()
        X_res, y_res = us.fit_resample(X, y)
    
    # Combination of over- and under-sampling:
    elif method == 'SMOTEENN':
        from imblearn.combine import SMOTEENN 
        us = SMOTEENN()
        X_res, y_res = us.fit_resample(X, y)

    elif method == 'SMOTETomek':
        from imblearn.combine import SMOTETomek 
        us = SMOTETomek()
        X_res, y_res = us.fit_resample(X, y)

    return X_res, y_res


def classifier_XGBoost(scoring, max_evals, train_dict, test_dict={}):
    """
    function to train iterations of XGBoost Classifier and optimize hyperparameters

    Arguments:
    train_dict -- a dictionary of train set features and labels
    test_dict (optional) -- a dictionary that contains test set features and labels
    scoring -- a performance metric to use as the objective of tunning hyperparams
    max_evals -- number of optimization evaluations to tune hyperparams

    Return:
    out_dict -- a dictionary of the optimal model found ('model') and its optimal objective metric ('score')
    """

    print("-----XGBoostClassifier-----")
    X_train, y_train = [train_dict['X_train'], train_dict['y_train']]

    print('Objective Metric: '+ scoring)
    def objective(space):
        classifier = xgb.XGBClassifier(n_estimators = space['n_estimators'],
                                    max_depth = int(space['max_depth']),
                                    learning_rate = space['learning_rate'],
                                    gamma = space['gamma'],
                                    min_child_weight = space['min_child_weight'],
                                    subsample = space['subsample'],
                                    colsample_bytree = space['colsample_bytree'],
                                    )
        
        classifier.fit(X_train, y_train)
        
        #k-Fold Cross Validation
        Scores = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, scoring=scoring)
        score = Scores.mean()
        loss = 1-score
        return {'loss': loss, 'status': STATUS_OK}

    # Tune Hyperparams
    space = {
        'max_depth' : hp.choice('max_depth', range(5, 30, 1)),
        'learning_rate' : hp.quniform('learning_rate', 0.01, 0.5, 0.01),
        'n_estimators' : hp.choice('n_estimators', range(20, 205, 5)),
        'gamma' : hp.quniform('gamma', 0, 0.50, 0.01),
        'min_child_weight' : hp.quniform('min_child_weight', 1, 10, 1),
        'subsample' : hp.quniform('subsample', 0.1, 1, 0.01),
        'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1.0, 0.01)}

    trials = Trials()
    print("Tuning Hyperparameters ...")
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)

    print("Best Hyperparameters: ", best)

    # Fit the best model
    BestModel = xgb.XGBClassifier(n_estimators = best['n_estimators'],
                                max_depth = best['max_depth'],
                                learning_rate = best['learning_rate'],
                                gamma = best['gamma'],
                                min_child_weight = best['min_child_weight'],
                                subsample = best['subsample'],
                                colsample_bytree = best['colsample_bytree'],
                                )

    BestModel.fit(X_train, y_train)

    print('XGBoostClassifier Performance:')
    # Applying k-Fold Cross Validation - Train set
    Scores = cross_val_score(estimator = BestModel, X = X_train, y = y_train, cv = 10, scoring='f1')
    score_train = Scores.mean()
    print("Train Set 10-Fold F1-Score: ", score_train)

    if bool(test_dict):
        # F1 score - Test set
        X_test, y_test = [test_dict['X_test'], test_dict['y_test']]
        y_pred = BestModel.predict(X_test)
        score_test = f1_score(y_test, y_pred)
        print("Test Set F1-Score: ", score_test)

    out_dict = {'model': BestModel, 'score': score_train}

    return out_dict

def classifier_LightGBM(scoring, max_evals, train_dict, test_dict={}):
    """
    function to train iterations of LightGBM Classifier and optimize hyperparameters

    Arguments:
    train_dict -- a dictionary of train set features and labels
    test_dict (optional) -- a dictionary that contains test set features and labels
    scoring -- a performance metric to use as the objective of tunning hyperparams
    max_evals -- number of optimization evaluations to tune hyperparams

    Return:
    out_dict -- a dictionary of the optimal model found ('model') and its optimal objective metric ('score')
    """

    print("-----LightGBM Classifier-----")

    X_train, y_train = [train_dict['X_train'], train_dict['y_train']]

    print('Objective Metric: '+ scoring)
    def objective(space):
        classifier = lgbm.LGBMClassifier(n_estimators = space['n_estimators'],
                                    max_depth = int(space['max_depth']),
                                    learning_rate = space['learning_rate'],
                                    min_child_weight = space['min_child_weight'],
                                    subsample = space['subsample'],
                                    colsample_bytree = space['colsample_bytree']
                                    )
        
        classifier.fit(X_train, y_train)
        
        #k-Fold Cross Validation
        Scores = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, scoring=scoring)
        score = Scores.mean()
        loss = 1-score
        return {'loss': loss, 'status': STATUS_OK}

    # Tune Hyperparams
    space = {
        'max_depth' : hp.choice('max_depth', range(5, 30, 1)),
        'learning_rate' : hp.quniform('learning_rate', 0.01, 0.5, 0.01),
        'n_estimators' : hp.choice('n_estimators', range(20, 205, 5)),
        'min_child_weight' : hp.quniform('min_child_weight', 1, 10, 1),
        'subsample' : hp.quniform('subsample', 0.1, 1, 0.01),
        'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1.0, 0.01)}

    trials = Trials()
    print("Tuning Hyperparameters ...")
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)

    print("Best Hyperparameters: ", best)

    # Fit the best model
    BestModel = lgbm.LGBMClassifier(n_estimators = best['n_estimators'],
                                max_depth = best['max_depth'],
                                learning_rate = best['learning_rate'],
                                min_child_weight = best['min_child_weight'],
                                subsample = best['subsample'],
                                colsample_bytree = best['colsample_bytree']
                                )

    BestModel.fit(X_train, y_train)

    print('LightGBMClassifier Performance:')
    # Applying k-Fold Cross Validation - Train set
    Scores = cross_val_score(estimator = BestModel, X = X_train, y = y_train, cv = 10, scoring='f1')
    score_train = Scores.mean()
    print("Train Set 10-Fold F1-Score: ", score_train)

    if bool(test_dict):
        # F1 score - Test set
        X_test, y_test = [test_dict['X_test'], test_dict['y_test']]
        y_pred = BestModel.predict(X_test)
        score_test = f1_score(y_test, y_pred)
        print("Test Set F1-Score: ", score_test)

    out_dict = {'model': BestModel, 'score': score_train}

    return out_dict

def eval_model(model, train_dict, test_dict={}):
    """
    function to evaluate a binary classification model

    Arguments:
    train_dict -- a dictionary of train set features and labels
    test_dict (optional) -- a dictionary that contains test set features and labels
    model -- a model object to evaluate

    Return:
    results -- a dictionary of performance metrics on train and test set
    """

    #X_train, y_train = [train_dict['X_train'], train_dict['y_train']]
    # reload original data in case of undersampling to calculate CV on unsampled data
    PATH_TRAIN = pjoin("Data", "dataset-challenge.xlsx")
    data_dict = gen_data(PATH_TRAIN, 'train')
    X, y = [data_dict['X'], data_dict['y']]
    train_dict = {'X_train': X, 'y_train':y}
    X_train, y_train = [train_dict['X_train'], train_dict['y_train']]

    results = {}
    # Applying k-Fold Cross Validation - Train set
    results['Algorithm'] = model.__class__.__name__ 
    results['train_CV_accuracy'] = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10, scoring='accuracy').mean()
    results['train_CV_accuracy_balanced'] = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10, scoring='balanced_accuracy').mean()
    results['train_CV_f1'] = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10, scoring='f1').mean()
    results['train_CV_f1_micro'] = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10, scoring='f1_micro').mean()
    results['train_CV_f1_macro'] = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10, scoring='f1_macro').mean()
    results['train_CV_f1_weighted'] = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10, scoring='f1_weighted').mean()
    results['train_CV_roc_auc'] = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10, scoring='roc_auc').mean()

    # test set evaluation
    if bool(test_dict):
        X_test, y_test = [test_dict['X_test'], test_dict['y_test']]
        y_pred = model.predict(X_test)
        results['test_accuracy'] = accuracy_score(y_test, y_pred)
        results['test_accuracy_balanced'] = balanced_accuracy_score(y_test, y_pred)
        results['test_f1'] = f1_score(y_test, y_pred)
        results['test_f1_micro'] = f1_score(y_test, y_pred, average='micro')
        results['test_f1_macro'] = f1_score(y_test, y_pred, average='macro')
        results['test_f1_weighted'] = f1_score(y_test, y_pred, average='weighted')
        results['test_roc_auc'] = roc_auc_score(y_test, y_pred)

    return results


def iterModel(name, max_evals, train_dict, test_dict={}):
    """
    function to iterate classification algorithms by optimizing various objective metrics

    Arguments:
    name -- a string of the classification algorithm's name.
    train_dict -- a dictionary of train set features and labels
    test_dict (optional) -- a dictionary that contains test set features and labels
    max_evals -- number of optimization evaluations to tune hyperparams

    Return:
    out -- a list of dictionaries of optimized models and their optimal objective metric
    """


    Scorings = ['accuracy', 'balanced_accuracy', 'f1', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision', 'recall', 'roc_auc']
    out = []
    if name == 'XGBoost':
        out = {}
        for scoring in tqdm(Scorings):
            out[scoring] = classifier_XGBoost(scoring=scoring, max_evals=max_evals, train_dict=train_dict, test_dict=test_dict)
    elif name == 'LightGBM':
        out = {}
        for scoring in tqdm(Scorings):
            out[scoring] = classifier_LightGBM(scoring=scoring, max_evals=max_evals, train_dict=train_dict, test_dict=test_dict)
    else:
        print('Algorithm not found!')

    return out
  

def MetaClassifier(Algorithms, train_dict, test_dict={}):
    """
    function to build a soft-voting ensemble meta-classifier

    Arguments:
    name -- a string of the classification algorithm's name.
    train_dict -- a dictionary of train set features and labels
    test_dict (optional) -- a dictionary that contains test set features and labels
    Algorithms -- list of model iterations outputs of different algorithms

    Return:
    out -- a dictionary of meta-classifier model ('BestModel'), a dataframe of performance metrics of all models (Metrics), 
           and a dataframe of metaclassifier predicted labels and probabilities on trainset instances (train_df_pred)
    """

    estimators=[]
    weights = []
    results = []

    print('Genearting a Soft-Voting Ensemble Classifier...')

    for alg in tqdm(Algorithms):
        for metric in tqdm(list(alg.keys())):
            
            # for voting ensemble
            update = (str(str(alg)+'_'+metric), alg[metric]['model'])
            estimators.append(update)
            weights.append(alg[metric]['score'])

            # for results
            results.append(eval_model(model=alg[metric]['model'], train_dict=train_dict, test_dict=test_dict))

    results_df = pd.DataFrame(results)

    results_df.sort_values(by='train_CV_f1', ascending=False)

    # keeping top five iterations for the ensemble
    results_df.nlargest(5, 'train_CV_f1')
    KeepIdx = results_df.nlargest(5, 'train_CV_f1').index

    BestEstimators= [estimators[i] for i in KeepIdx.tolist()]
    BestWeights = [weights[i] for i in KeepIdx.tolist()]

    X_train, y_train = [train_dict['X_train'], train_dict['y_train']]

    MetaClassifier = VotingClassifier(estimators=BestEstimators, voting='soft')
    MetaClassifier = MetaClassifier.fit(X_train, y_train)

    results_df = results_df.append(eval_model(MetaClassifier, train_dict, test_dict), ignore_index=True).sort_values(by='train_CV_f1', ascending=False) # add metaclassification model metrics

    # storing metaclassifier performance on trainset
    y_pred = MetaClassifier.predict(X_train)
    y_proba =  MetaClassifier.predict_proba(X_train)
    train_df_pred = X_train
    train_df_pred['y_true'] = y_train
    train_df_pred['y_hat'] = y_pred
    train_df_pred['y_hat_proba'] = y_proba[:,1] # probability of belonging to class 1

    out = {'BestModel': BestEstimators[0][1], 'Metrics':results_df, 'train_df_pred':train_df_pred}
    return out
