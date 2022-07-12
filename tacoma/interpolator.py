import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor

class RBFIRegressor:
    """ RBFInterpolator from Scipy rewritten to have the same API as in Sklearn.
    The full documentation of the interpolator can be found at https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html

    """
    def __init__(self,degree:int = 2,**kwargs):
        self.params = {
            "degree":degree,
            **kwargs} 

    def fit(self,X,y):
        """
        Fits the regression model to the features and labels

        :param X: Array of features for training the model
        :param y: array of label for training the model
        """
        self.model = RBFInterpolator(X,y,**self.params)

    def predict(self,X):
        """
        Returns the input from a prediction

        :param X: array of feature for predicting labels
        """
        return self.model(X)

    def get_params(self):
        """
        Returns a dict with the hyperparameters of the model
        """
        return self.params

    def set_params(self,new_params:dict):
        """
        Set new hyperparameters for the model

        :param new_params: new set of hyperparameters for the model
        """
        self.params = new_params

class KNeighborsBackmap:
    """
    KNeighborsRegressor implmentation as a backmapping method.
    """
    def __init__(self,n_neighbors = 5,**kwargs):
        self.map_model = KNeighborsRegressor(n_neighbors = n_neighbors,**kwargs)
    def fit(self,y_train,y_iso_train,y_iso_pred):
        """
        Fit
        """
        self.map_model.fit(y_iso_train,y_train)
        self.y_pred = self.map_model.predict(y_iso_pred)

    def predict(self):
        return self.y_pred

