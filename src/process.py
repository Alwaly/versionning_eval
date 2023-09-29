from utils import *
from pandas_profiling import ProfileReport

def processor(file_name):
    data = csv_reader(file_name)
    profile = ProfileReport(data)
    profile.to_file(output_file='../docs/rapport.html')
    X, Y = process(data)
    X, X_, categorical_features, continuous_features = feature_engineering(X)
    X = normalisation(X, X_, categorical_features, continuous_features)
    
    return X, Y