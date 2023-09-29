import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def csv_reader(file_name):
    directory_name = file_name.split('/')[0]
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    data = pd.read_csv(file_name)
    return data

def process(data):
    df = data.drop_duplicates()
    df.dropna(inplace=True)

    df.drop(['YEAR'], axis=1, inplace=True)

    min_count=df['ILLICIT'].value_counts().min()

    df = df.groupby('ILLICIT').apply(lambda x: x.sample(min_count)).reset_index(drop=True)

    Y = df['ILLICIT']
    X = df.drop('ILLICIT', axis=1)
    return X, Y

def feature_engineering(X):
    continuous_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    df_continuous = X[continuous_features]

    df_categorical = X[categorical_features]

    X_ = X.drop(categorical_features, axis=1)
    X = X.drop(continuous_features, axis=1)
    return X, X_, categorical_features, continuous_features

def normalisation(X, X_, categorical_features, continuous_features)
    label_encoder = LabelEncoder()

    for column in categorical_features:
        X[column] = label_encoder.fit_transform(X[column])
    
    X_ = StandardScaler().fit_transform(X_)

    X_ = pd.DataFrame(X_, columns=continuous_features)
    X = pd.concat([X_,X], axis=1, ignore_index=True)
    return X