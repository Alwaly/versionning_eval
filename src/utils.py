import pandas as pd
import os

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