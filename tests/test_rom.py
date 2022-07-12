#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import pytest
from tacoma.rom import IsomapRegressor, PODRegressor
from tacoma.interpolator import KNeighborsBackmap
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
rng = np.random.RandomState(42)

X_train = rng.random_sample(size = (100,3))
y_train = rng.random_sample(size = (100,111_000))
X_test = rng.random_sample(size = (10,3))
y_test = rng.random_sample(size = (10,111_000))


@pytest.mark.parametrize("regression",[RandomForestRegressor,DecisionTreeRegressor])
def test_pod_rom(regression):
    reg_model = regression()
    model = PODRegressor(regression_model=reg_model)
    model.fit_transform(X_train,y_train,X_test,y_test)


@pytest.mark.parametrize("backmapping",[KNeighborsBackmap])
@pytest.mark.parametrize("regression",[RandomForestRegressor,DecisionTreeRegressor,KNeighborsRegressor])
def test_isomap_rom(regression,backmapping):
    reg_model = regression() 
    map_model = backmapping() 
    model = IsomapRegressor(regression_model=reg_model,backmapping_model = map_model)
    model.fit_transform(X_train,y_train,X_test,y_test)


