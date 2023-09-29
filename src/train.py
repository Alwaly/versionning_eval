from process import *
import pickle as pkl
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

X, Y = processor('data/douanesDataset.csv')

param_grid_rf = [
    {'n_estimators':[10,30]},
    {'max_depth':[2,5,7]}, 
    {'max_features':[2,4,6]}, 
    {'min_samples_leaf':[2,5,9,3]}
]
param_grid_knn= [
    {'n_neighbors':[3,5,7,10,15,12,13,16,19,17,11]}
]
param_grid_log=[
    {'max_iter':[10,100,120,5,90,80,85]}
]

classifiers =[]
classifiers.append(RandomForestClassifier())
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression())

parametres=[]
parametres.append(param_grid_rf)
parametres.append(param_grid_knn)
parametres.append(param_grid_log)

best_model= gridsearch_dit(classifiers, parametres, X, Y)

mod, accu = best_model
for i in range(len(mod)):
  if accu[i]== max(accu):
    with open("model.pkl", 'wb') as model_file:
      pkl.dump(mod[i], model_file)