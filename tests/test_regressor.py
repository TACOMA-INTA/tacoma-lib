import pytest
from tacoma.interpolator import RBFIRegressor
import numpy as np
rng = np.random.RandomState(0)

X_train = rng.random_sample(size = (100,3))
y_train = rng.random_sample(size = (100,111_000))
X_test = rng.random_sample(size = (10,3))
y_test = rng.random_sample(size = (10,111_000))

def test_RBFIRegressor():
    model = RBFIRegressor()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)