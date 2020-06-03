from pprint import pprint

import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.utils import resample

if __name__ == '__main__':
    df = pd.read_csv('online_shoppers_intention_V0.2.csv')

    # Print out the head name of all features
    print(df.columns)

    #Print out the first and last rows of the dataset
    print(df.iloc[0])
    print(df.iloc[-1])

    # Check if there are any missing values in the data. Just provide the code, it is not necessary to print something
    df.isna()
    df.isnull()
    #Please interchange third and fifth features and print the head of the dataframe.
    cols = df.columns.tolist()
    new_cols = cols
    cols[2],cols[4]=cols[4],cols[2]
    print(df[cols].head()) # head
    print(df[cols].columns) # or header ?
    #Print out the row index of the cell with the highest “ProductRelated_Duration” value
    print(np.argmax(np.array(df['ProductRelated_Duration'].fillna(0.0))))
    # Please remove the “Informational” feature from the dataset and print the head of the dataframe.
    df.pop('Informational')
    print(df.columns)
    #Print out the first and third quantile of “PageValues” feature TODO(tilo): quartiles ??
    print(df['PageValues'].describe([0.25, 0.75]))
    #Sort the columns in reverse alphabetical order of head names and print the head of the dataframe.
    print(list(reversed(sorted(df.columns.tolist()))))
    # Count and print the number of missing values in each column
    print(df.isna().sum(axis=0))
    # From the dataset, filter the “Month”, “ExitRates” and “SpecialDay” for every 250th row starting from row 0 and print the result.
    print(df[['Month','ExitRates','SpecialDay']].iloc[list(range(0,len(df),250))])
    #Scales and translates all of the numerical features in the range of 0 to 5, using min-max normalization method and print the first 10 rows of the result.
    dfnum = df.select_dtypes(include=[np.number])
    df_norm = (dfnum - dfnum.min()) / (dfnum.max() - dfnum.min())*5
    print(df.head(10))
    #Print out the head name and variance of the numerical feature with highest variance
    print(df.select_dtypes(include=[np.number]).var().argmax())
    # Use the “apply” function to replace missing values in the “Informational_Duration” and “ProductRelated_Duration” columns with the mean of the attribute
    m = df_norm['Informational_Duration'].mean()
    df_norm['Informational_Duration'].apply(lambda x:m if np.isnan(x) else x)
    m = df_norm['ProductRelated_Duration'].mean()
    df_norm['ProductRelated_Duration'].apply(lambda x:m if np.isnan(x) else x)
    #Print out the index of rows where values of “OperatingSystems”, “Browser” and “TrafficType” columns match
    print(df.iloc[np.equal(df['Browser'].eq(df['TrafficType']).values, df['Browser'].eq(df['OperatingSystems']).values)])

    #For all of the features, please print out the maximum possible correlation value of each column against other columns # TODO(tilo):numeric features?
    print(df.corr())
    #In the “Browser” feature, keep only top 4 most frequent values as it is and replace the other values with the “Other” label TODO(tilo):what is the Other label?
    most_frequent = df.Browser.value_counts().iloc[:4].index.tolist()
    OTHER = -1
    df.Browser.apply(lambda x:x if x in most_frequent else OTHER)
    # Compute and print out the mean of “BounceRates” of each “Month”
    print(df.groupby('Month').BounceRates.agg('mean'))
    #Print out all of the instances with “BounceRates” values greater than “ProductRelated_Duration”
    print(df.loc[df.BounceRates.gt(df.ProductRelated_Duration)])
    #Please plot the boxplot graph of all numerical features # TODO(tilo): of normalized features ?
    from matplotlib import pyplot as plt
    df_norm.boxplot()
    # plt.show()
    #Bin all of the numeric series to 5 groups of equal size and print out the first 10 rows of the resulting data
    print(pd.DataFrame([pd.qcut(dfnum[col], 5, duplicates='drop') for col in dfnum]).head(10))
    #Since the provided data is imbalanced, use upsampling approach to make it balanced. Please print the distribution of target labels in the resulted data.
    df_norm['Revenue'] = df.Revenue
    upsampled_df = resample(df_norm[df.Revenue == True], replace=True, n_samples=sum(df.Revenue == False), random_state=123)
    balanced_df = pd.concat([df_norm[df.Revenue == False], upsampled_df])
    print(balanced_df.Revenue.value_counts())
    # Please apply linear regression, support vector machine and naive bayes classifiers  on the preprocessed dataset (5-fold cross validation approach).
    # TODO(tilo): linear regression classifier?
    y = balanced_df.pop('Revenue')
    X = balanced_df.fillna(0.0).to_numpy() # TODO(tilo): data was not properly preprocessed
    regression = LinearRegression()

    classifiers = [('gaussNB', GaussianNB()),('sgdclf', SGDClassifier()), ('svc', SVC())]
    scores = {name: cross_val_score(model, X, y, scoring=make_scorer(sklearn.metrics.f1_score), cv=5).mean() for name, model in classifiers}
    pprint(scores)
    # Please print out the confusion matrix of obtained results and the F1 measure of the 3 classifiers
    for name, clf in classifiers:
        y_pred = cross_val_predict(clf,X,y,cv=5)
        print(confusion_matrix(y, y_pred))
    #Ensemble top two classifiers with better results and print out the F1 score of the results
    eclf = VotingClassifier(estimators = classifiers[1:],voting = 'hard')
    print(cross_val_score(eclf, X, y, scoring=make_scorer(sklearn.metrics.f1_score), cv=5).mean())
