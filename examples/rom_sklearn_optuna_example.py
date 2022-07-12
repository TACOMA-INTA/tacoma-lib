from tacoma.rom import IsomapRegressor, PODRegressor
from tacoma.metrics import GetScores
from tacoma.interpolator import KNeighborsBackmap
import numpy as np
import optuna # framework for Bayesian optimization
from sklearn.ensemble import RandomForestRegressor
rng = np.random.RandomState(42)

X_train = rng.random_sample(size = (100,3))
y_train = rng.random_sample(size = (100,111_000))
X_test = rng.random_sample(size = (10,3))
y_test = rng.random_sample(size = (10,111_000))

def objective(trial):
    """
    Function to be optimized using Bayesina optimization using Optuna
    In this case, the number of components and neigbors of the Isomap are going 
    to be optimized
    """
    n_components_iso = trial.suggest_int("n_components",2,5,1) # fist hyperparameter to be optimized
    n_neighbors_iso = trial.suggest_int("n_neighbors",5,15,1) # second hyperparameter to be optimized
    reg_model = RandomForestRegressor()
    map_model = KNeighborsBackmap()
    iso_model = IsomapRegressor(
        n_components=n_components_iso,
        n_neighbors=n_neighbors_iso,
        regression_model=reg_model,backmapping_model = map_model)
    iso_model.fit_transform(X_train,y_train,X_test,y_test)
    iso_scores = GetScores(y_test,iso_model.predict())
    return iso_scores.r2_score().mean() # returns the mean of the r2 of the regression, acts as the loss function

study = optuna.create_study(direction="maximize")
study.optimize(objective,n_trials = 10)
