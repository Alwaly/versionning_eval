from utils import *

def processor(file_name):
    data = csv_reader(file_name)
    X, Y = process(data)
    X, X_, categorical_features, continuous_features = feature_engineering(X)
    X = normalisation(X, X_, categorical_features, continuous_features)
    
    return X, Y