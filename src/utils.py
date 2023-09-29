import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SequentialFeatureSelector

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

def normalisation(X, X_, categorical_features, continuous_features):
    label_encoder = LabelEncoder()

    for column in categorical_features:
        X[column] = label_encoder.fit_transform(X[column])
    
    X_ = StandardScaler().fit_transform(X_)

    X_ = pd.DataFrame(X_, columns=continuous_features)
    X = pd.concat([X_,X], axis=1, ignore_index=True)
    return X

def gridsearch_dit(models, params,X_train, y_train):
  model_best=[]
  accuracy_best=[]
  for i in range(len(models)):
    print(f'tour numero: {i}')
    sfs = SequentialFeatureSelector(estimator=models[i], n_features_to_select = 'auto', scoring='accuracy', direction='backward')
    sfs.fit(X_train, y_train)
    selected_feature_indices = sfs.get_support(indices=True)
    X_train_selected = X_train[:, selected_feature_indices]
    model=GridSearchCV(models[i], params[i], cv=5)
    model.fit(X_train_selected, y_train)
    print(model.best_estimator_)
    print(model.best_score_)
    model_best.append(model.best_estimator_)
    accuracy_best.append(model.best_score_)
  return model_best, accuracy_best