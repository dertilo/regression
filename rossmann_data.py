import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from util import data_io


def build_features(data):
    features = []
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    # Use some properties directly
    features.extend(
        ['Store', 'CompetitionDistance', 'Promo', 'Promo2', 'SchoolHoliday'])

    # Label encode some features
    features.extend(['StoreType', 'Assortment', 'StateHoliday'])
    mappings = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
    data.StoreType.replace(mappings, inplace=True)
    data.Assortment.replace(mappings, inplace=True)
    data.StateHoliday.replace(mappings, inplace=True)

    features.extend(['DayOfWeek', 'Month', 'Day', 'Year', 'WeekOfYear'])
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek
    data['WeekOfYear'] = data.Date.dt.weekofyear

    # CompetionOpen en PromoOpen from https://www.kaggle.com/ananya77041/rossmann-store-sales/randomforestpython/code
    # Calculate time competition open time in months
    features.append('CompetitionOpen')
    data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + \
                              (data.Month - data.CompetitionOpenSinceMonth)
    # Promo open time in months
    features.append('PromoOpen')
    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) + \
                        (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)
    data.loc[data.Promo2SinceYear == 0, 'PromoOpen'] = 0

    # Indicate that sales on that day are in promo interval
    features.append('IsPromoMonth')
    month2str = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', \
                 7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    data['monthStr'] = data.Month.map(month2str)
    data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
    data['IsPromoMonth'] = 0
    for interval in data.PromoInterval.unique():
        if interval != '':
            for month in interval.split(','):
                data.loc[(data.monthStr == month) & (
                        data.PromoInterval == interval), 'IsPromoMonth'] = 1

    return features

def read_csvs_build_features(path):
    types = {'CompetitionOpenSinceYear': np.dtype(int),
             'CompetitionOpenSinceMonth': np.dtype(int),
             'StateHoliday': np.dtype(str),
             'Promo2SinceWeek': np.dtype(int),
             'SchoolHoliday': np.dtype(float),
             'PromoInterval': np.dtype(str)}
    train_df = pd.read_csv(path + "/train.csv", parse_dates=[2], dtype=types)
    store = pd.read_csv(path + "/store.csv")
    # print("Assume store open, if not provided")
    train_df.fillna(1, inplace=True)
    train_df = train_df[train_df["Open"] != 0]
    train_df = train_df[train_df["Sales"] > 0]
    train_df = pd.merge(train_df, store, on='Store')
    features = build_features(train_df)
    print("features: %s"%str(features))
    train_df['target'] = np.log1p(train_df.Sales)
    return train_df, features

def read_or_process_data(path):
    tmp_train_data_file = '/tmp/train_data.jsonl.gz'
    tmp_meta_file = '/tmp/meta_data.jsonl'
    if not os.path.exists(tmp_train_data_file):

        train_df, features = read_csvs_build_features(path)
        types = {name: typ for name, typ in train_df.dtypes.to_dict().items() if
                 name in features}
        numerical_features = [name for name, typ in types.items() if
                              typ == float or typ == int]
        # print(numerical_features)
        categorical_features = [name for name, typ in types.items() if typ == str]
        # print(categorical_features)
        input_dim = len(numerical_features) + len(categorical_features)

        def dataframe_to_dicts(df):
            data = [row[1].to_dict() for row in df.iterrows()]
            [d.__delitem__('Date') for d in data]
            return data

        train_data_dicts = dataframe_to_dicts(train_df)
        y_train_list = train_df['target'].tolist()
        [d.__setitem__('target', t) for d, t in zip(train_data_dicts, y_train_list)]
        data_io.write_jsonl(tmp_train_data_file, train_data_dicts)
        data_io.write_jsonl(tmp_meta_file, [
            {'numerical_features': numerical_features,
             'categorical_features': categorical_features}])
    else:
        print('loading already processed data')
        train_data_dicts = list(data_io.read_jsonl(tmp_train_data_file, limit=limit))
        y_train_list = [d['target'] for d in train_data_dicts]
        meta = list(data_io.read_jsonl(tmp_meta_file))[0]
        numerical_features = meta['numerical_features']
        categorical_features = meta['categorical_features']
    # features = categorical_features + numerical_features
    return train_data_dicts, categorical_features, numerical_features




if __name__ == '__main__':
    path = "/home/tilo/code/ML/regression/data"
    train_df, features = read_csvs_build_features(path)

    # train_data_dicts,categorical_features, numerical_features = read_or_process_data(path)
    # X_train, X_valid = train_test_split(train_data_dicts, test_size=0.2, random_state=10)
    # return X_train,X_valid,categorical_features, numerical_features
