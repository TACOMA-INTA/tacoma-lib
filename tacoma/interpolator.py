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



class TaylorKNNBackmap:
    """
    Backmapping method

    TO-DO:
        - Add docstrings
        - Add citation

    """
    def __init__(self,n_neighbors = 5,**kwargs):
        self.map_model = NearestNeighbors(n_neighbors = n_neighbors,**kwargs)

    def fit(self,y_train,y_train_iso):
        """
        Fit method
        """
        self.y_train = y_train
        self.y_iso_train = y_train_iso
        self.map_model.fit(self.y_iso_train)

    def predict(self,y_iso_pred):
        """
        Transform method
        """
        self.y_pred = np.zeros([y_iso_pred.shape[0],self.y_train.shape[1]])
        for i in range(y_iso_pred.shape[0]):
            idxk = self.map_model.kneighbors(y_iso_pred[i:i+1,:],return_distance = False)
            self.y_pred[i,:] = self.__taylor_expansion__(X = self.y_train,y = self.y_iso_train,idx = idxk,y_pred = y_iso_pred[i:i+1,:])
        return self.y_pred
    def fit_transform(self,y_train,y_train_iso,y_iso_pred):
        """
        Fit Transform method
        """
        self.fit(y_train,y_train_iso)
        self.y_pred = self.predict(y_iso_pred)
        return self.y_pred

    @staticmethod
    def __taylor_expansion__(X,y,y_pred,idx):
        """
        Back-mapping with a Taylor norder approximation of the k neighbours.
        """
        xx = X[idx[0,1:],:] - X[idx[0,0],:]
        yy = y[idx[0,1:],:] - y[idx[0,0],:]
        grad = np.linalg.multi_dot(
        [np.linalg.inv(np.dot(yy.T,yy))
        ,yy.T
        ,xx])
        # grad = np.dot(np.linalg.inv(np.dot(yy.T,yy)),np.dot(yy.T,xx))
        xpred = np.dot((y_pred-y[idx[0,0],:]),grad) + X[idx[0,0],:]
        return xpred
