import os

import numpy
from sklearn.model_selection import train_test_split

from feature_extraction import NumericFeatureGetter
from rossmann_data import read_or_process_data
from rossmann_sales_kaggle_benchmarking_regression_methods import score_trained_regressor

from sklearn.neural_network import MLPRegressor

from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from time import time

if __name__ == '__main__':

    path = os.getcwd()+"/data"
    train_data_dicts,categorical_features, numerical_features = read_or_process_data(path)
    train_data_dicts, test_data_dicts = train_test_split(train_data_dicts, test_size=0.2, random_state=10)

    y_train_list = [d['target'] for d in train_data_dicts]
    y_test_list = [d['target'] for d in test_data_dicts]
    print('got %d train-samples' % len(train_data_dicts))
    print('got %d test-samples' % len(test_data_dicts))
    ###################################################################################################################
    #       just printing mix-max
    for feat_name in numerical_features:
        # print(pandas.Series(data=[d[feat_name] for d in train_data_dicts]).describe())
        features = [d[feat_name] for d in train_data_dicts]
        print(feat_name + '# min: %0.2f; max: %0.2f' % (min(features), max(features)))
    ###################################################################################################################
    #      defining Feature-Extractor objects

    num_getter = NumericFeatureGetter(numerical_features)
    preprocessed_numeric_features = Pipeline([
        ('numeric_features', num_getter),
        ('preprocessed_numeric_feats', PCA(whiten=True))
        # ('preprocessed_numeric_feats', preprocessing.Normalizer())
    ])
    # preprocessed_numeric_features.fit(train_data_dicts)
    for name, regressor in [
        ('lin_reg',LinearRegression()),
        ('elastic_net',ElasticNet(alpha=0.001, l1_ratio=0.7)),
        ('mlp',MLPRegressor(hidden_layer_sizes=(10,),batch_size=128,alpha=0.000001,early_stopping=True)),
    ]:
        start = time()
        regression_pipeline = Pipeline([('feature_extractor', preprocessed_numeric_features), ('regressor', regressor)])
        target_array = numpy.array([d['target'] for d in train_data_dicts])
        regression_pipeline.fit(train_data_dicts, target_array)

        train_err,test_err = score_trained_regressor(regression_pipeline, train_data_dicts, y_train_list, test_data_dicts, y_test_list)
        duration = time()-start
        assert isinstance(duration,float)
        print(name+'     train: %0.3f; test: %0.3f'%(train_err,test_err)+' took: %0.2f secs'%duration)

