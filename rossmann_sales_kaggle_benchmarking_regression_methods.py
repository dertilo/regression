#!/usr/bin/python

'''
Based on https://www.kaggle.com/justdoit/rossmann-store-sales/xgboost-in-python-with-rmspe/code
Public Score :  0.11389
Private Validation Score :  0.096959
'''

import matplotlib
import numpy as np

matplotlib.use("Agg") #Needed to save figures


class DummyTargetPreprocessor(object):
    def __init__(self,target_names):
        self.classes_ = target_names
    def fit(self,X):
        return self
    def transform(self,X):
        return X
    def inverse_transform(self,X):
        return X

def rmspe(ys, yhats):
    return np.sqrt(np.mean([(yhat/y-1) ** 2 for yhat,y in zip(yhats,ys)]))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return rmspe(y,yhat)

def score_trained_regressor(regressor,train_data_dicts,y_train_list,test_data_dicts,y_test_list):
    yhat = regressor.predict(test_data_dicts)
    yhat_train = regressor.predict(train_data_dicts)
    train_score = rmspe(np.expm1(y_train_list), np.expm1(yhat_train))
    test_score = rmspe(np.expm1(y_test_list), np.expm1(yhat))
    return train_score,test_score