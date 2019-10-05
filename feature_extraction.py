from sklearn.impute import SimpleImputer
from typing import List, Tuple, Dict

import numpy as np
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

# fmt: off
feature_names = ['other','numeric', 'mainly_numeric', 'allLower', 'allUpper', 'initialUpper', 'contains_digit',
                 'not_identifier']
# fmt: on


def getCasing(word:str)->str:
    casing = 'other'

    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1

    digitFraction = numDigits / float(len(word))

    if word.isdigit():  # Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower():  # All lower case
        casing = 'allLower'
    elif word.isupper():  # All upper case
        casing = 'allUpper'
    elif word.istitle():  # is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'
    elif not word.isidentifier():
        casing = 'not_identifier'

    return casing

def getCasing_numeric(word:str)->float:
    return (feature_names.index(getCasing(word))-len(feature_names)/2)/len(feature_names)

class NumericFeatureGetter(BaseEstimator, TransformerMixin):
    def __init__(self,feature_names):
        self.feature_names = feature_names
    def get_feature_names(self):
        return self.feature_names
    def fit(self, data, y=None):
        # dt = TimeDiff()
        self.imps:Dict[str,SimpleImputer] = {}
        for feat in self.feature_names:
            self.imps[feat] = SimpleImputer(strategy='median')
            x = np.array([d[feat] for d in data]).reshape(-1, 1)
            self.imps[feat].fit(x)
        # dt.print('done fitting NumericFeatureGetter')
        return self

    def transform(self, data):
        if len(data)==1:
            out = np.transpose(np.array( #TODO this is ugly shit!
                [np.squeeze(self.imps[feat].transform(np.array([d[feat] for d in data]).reshape(-1, 1)))
                 for feat in self.feature_names]).reshape(-1,1))
        else:
            out = np.transpose(np.array(
                [np.squeeze(self.imps[feat].transform(np.array([d[feat] for d in data]).reshape(-1, 1))) for feat in
                 self.feature_names]))
        # assert out.shape[0]==len(data)
        # assert out.shape[1]==len(self.feature_names)
        return out
        # return [self.imps[feat].transform(np.array([d[feat] for d in data]).reshape(-1, 1)) for feat in self.feature_names]


class FeatureBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_binarize = []):
        self.features_to_binarize = features_to_binarize
        self.feature_names = []
    def _wrap_in_list(self,x):
        return [x] if not isinstance(x,list) and not isinstance(x,set) else x
    def _prefix_features(self,feats,prefix):
        return  [prefix+'_'+f for f in feats]
    def fit(self,data,y=None):
        self.binarizers = {}
        dt = TimeDiff()
        for feat_name in self.features_to_binarize:
            features = [self._prefix_features(self._wrap_in_list(d[feat_name]),feat_name) for d in data]
            binarizer = MultiLabelBinarizer(sparse_output=True)
            binarizer.fit(features)
            self.feature_names.extend(binarizer.classes_)
            self.binarizers[feat_name] = binarizer
            print(feat_name+": "+str(len(binarizer.classes_)))
        dt.print('done fitting FeatureBinarizer')
        return self

    def transform(self,data):
        sparse_feats = []
        def filter_out_the_unknown(bin_feat_names):
            return [f for f in bin_feat_names if f in self.feature_names]
        for feat_name in self.features_to_binarize:
            features = [self._wrap_in_list(filter_out_the_unknown(self._prefix_features(d[feat_name], feat_name))) for d in data]
            sparse_feats.append(self.binarizers[feat_name].transform(features))
        X_bin = hstack(sparse_feats,format='csr')
        return X_bin
    def get_feature_names(self):
        return self.feature_names
