from tacoma.utils import weigthed_KNN
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
rng = np.random.RandomState(0)

X_train = rng.random_sample(size = (100,3))
y_train = rng.random_sample(size = (100,111_000))
X_test = rng.random_sample(size = (10,3))
y_test = rng.random_sample(size = (10,111_000))

def test_weighted_knn():
    model = KNeighborsRegressor(weights=weigthed_KNN)
    model.fit(X_train,y_train)
    y_pred  = model.predict(X_test)
    
