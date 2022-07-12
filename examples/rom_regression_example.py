from tacoma.rom import IsomapRegressor, PODRegressor
from tacoma.metrics import GetScores
from tacoma.interpolator import KNeighborsBackmap
import numpy as np
from sklearn.ensemble import RandomForestRegressor
rng = np.random.RandomState(42)

X_train = rng.random_sample(size = (100,3))
y_train = rng.random_sample(size = (100,111_000))
X_test = rng.random_sample(size = (10,3))
y_test = rng.random_sample(size = (10,111_000))



# Example of an POD ROM model using 100% of the reconstruction energy 
reg_model = RandomForestRegressor() # regression algorithm
pod_model = PODRegressor(regression_model=reg_model)
pod_model.fit_transform(X_train,y_train,X_test,y_test)
pod_scores = GetScores(y_test,pod_model.predict())
print(f"R2:{pod_scores.r2_score().mean()} MSE: {pod_scores.mse_error().mean()}") # Regression metrics
# Example of an ISOMAP ROM using 2 componentes and 10 neighbors 
# based on KNNeighbors backmapping 
map_model = KNeighborsBackmap()
iso_model = IsomapRegressor(
    n_components=2,
    n_neighbors=10,
    regression_model=reg_model,backmapping_model = map_model)
iso_model.fit_transform(X_train,y_train,X_test,y_test)
iso_scores = GetScores(y_test,iso_model.predict())
print(f"R2:{iso_scores.r2_score().mean()} MSE: {iso_scores.mse_error().mean()}") # Regression metrics