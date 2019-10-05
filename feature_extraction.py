from typing import List, Tuple, Dict

import numpy as np
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Imputer, MultiLabelBinarizer

from text_processing.text_classification_util import multifeature_tokenize
from .util_methods import TimeDiff

feature_names = ['other','numeric', 'mainly_numeric', 'allLower', 'allUpper', 'initialUpper', 'contains_digit',
                 'not_identifier']


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
        self.imps:Dict[str,Imputer] = {}
        for feat in self.feature_names:
            self.imps[feat] = Imputer(missing_values='NaN', strategy='median', axis=0)
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


class BagOfWordsFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, text_fields, vectorizer=TfidfVectorizer()):
        self.vectorizer = vectorizer
        self.vectorizer.preprocessor = lambda x:x
        self.vectorizer.tokenizer = lambda x:x
        self.text_fields = text_fields

    def datum_to_bow(self,datum:Dict):
        return multifeature_tokenize(datum,self.text_fields,prefix_or_tuples=True)

    def fit(self, data:List[Dict],dummy=None):
        self.vectorizer.fit((self.datum_to_bow(d) for d in data))
        self.feature_names = self.vectorizer.get_feature_names()
        print('vocabulary size of bag-of-words: ' + str(len(self.vectorizer.vocabulary_)))
        return self

    def transform(self,data):
        return self.vectorizer.transform((self.datum_to_bow(d) for d in data))

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()


def calc_entity_distance_sequences(token_spans:List[Tuple[int, int, str]],
                                   entity_start_ends:List[Tuple[int, int]],
                                   min_max_offset:Tuple[int,int],
                                   normalize_by=1.0,
                                   )->List[List[float]]:
    def calc_closest_token_start_or_end(char_pos,start_flag = True):
        return min([i for i in range(len(token_spans))], key=lambda pos: np.abs(char_pos - token_spans[pos][0 if start_flag else 1]))

    entity_token_start_ends = [(calc_closest_token_start_or_end(start,start_flag=True),
                           calc_closest_token_start_or_end(start,start_flag=False))
                          for start,end in entity_start_ends]
    minimum,maximum = min_max_offset[0], min_max_offset[1]
    def calc_distance(k,ent_start,ent_end):
        if np.abs(k-ent_start)<np.abs(k-ent_end):
            return max(minimum, min(maximum, k - ent_start)) / normalize_by
        else:
            return max(minimum, min(maximum, k - ent_end)) / normalize_by

    return [ [calc_distance(k,ent_start,ent_end) for ent_start,ent_end in entity_token_start_ends]
             for k in range(len(token_spans))]