import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import pickle
from sklearn.manifold import Isomap 
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics

class PODRegressor:
    """POD Regression ROM. is a numerical method that enables a reduction in the complexity of computer intensive simulations such as computational fluid dynamics.
    It is based on the usage of Singular Value Descomposition to obtain the spatial and temporal modes 
    """
    def __init__(self,n_PODmodes = None,regression_model=None,**kwargs):
        self.__init_regression_model__(regression_model)
        self.n_PODmodes = n_PODmodes 

    def __init_regression_model__(self,regression_model):
        """Private method to initilaize the regression model.
        """
        self.regression_model = regression_model
        if self.regression_model == None: # weird bug, see https://github.com/scikit-learn/scikit-learn/issues/23769 
            raise ValueError("There is no regression_model assigned")

    def fit_SVD(self,y:np.array, subtract_mean:bool = True):
        """
        SVD performs a Singular Value Desompositon of the snapshot matrix X.
        
        ----------
        Parameters
        ----------
        Attributes:
        - y -> snapshot matrix (each column is an snapshot)
        Output:
        - U,S,Vh -> SVD decomposition.
        """
        if subtract_mean:
            y = y-[np.mean(y,axis=0)]
        self.U,self.S,self.Vh = np.linalg.svd(y.T, full_matrices=False)
        if not self.n_PODmodes:
            self.n_PODmodes = self.U.shape[1]
        self.Vhr = self.Vh[:self.n_PODmodes,:]
        

    def fit_transform(self,X_train,y_train,X_test,y_test):
        """
        Perfoms POD -> train Regressor model -> fit data

        """
        self.fit_SVD(y_train, subtract_mean = False)
        self.y_pod_train = self.Vhr.T
        self.regression_model.fit(X_train, self.y_pod_train)
        self.y_pod_pred = self.regression_model.predict(X_test)
        # Backmapping
        self.y_pred = self.backmapping(self.y_pod_pred)
    
    def predict(self):
        """
        Reconstruct POD

        """
        return self.y_pred


    def backmapping(self,Vr_pred:np.array):
        """
        Backmapping from POD modes to high-dimensional space
        """
        #if not self.n_PODmodes: # se realiza esta operacion en el metodo de SVD
        #    self.n_PODmodes = self.U.shape[1]
        
        Ur = self.U[:,:self.n_PODmodes]
        Sr = self.S[:self.n_PODmodes]

        y_pred = np.dot(Ur, np.dot(np.diag(Sr), Vr_pred.T))
        return y_pred.T


class IsomapRegressor:
    """
    Isomap Regression ROM. For more infomation of the avilable parameters visits http://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html
    

    -----------
    Parameters
    -----------
    n_components: int: Number of coordinates for the manifold.
    n_neighbors: int:  Number of coordinates for the manifold.
    regression_model: Regression model to be used for the ROM interpolation
    backmapping_model: Regression model to go from the low dimensionality prediction to high dimensionaly representation
    """
    def __init__(self,n_components=2, n_neighbors= 10,regression_model=None, backmapping_model=None,**kwargs):
        self.__init_embedding__(n_components,n_neighbors,**kwargs)
        self.__init_regression_model__(regression_model)
        self.__init_backmapping_model__(backmapping_model)


    def __repr__(self):
        return self.embedding.get_params()


    def __init_embedding__(self,n_components:int,n_neighbors:int,**kwargs):
        """Private method to initilaize the embedding in the Isomap.
        To see more information go to http://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html
        """
        self.embedding = Isomap(n_components=n_components,n_neighbors=n_neighbors,**kwargs)

    def __init_regression_model__(self,regression_model):
        """Private method to initilaize the regression model.
        """
        self.regression_model = regression_model
        if self.regression_model == None: # weird bug, see https://github.com/scikit-learn/scikit-learn/issues/23769 
            raise ValueError("There is no regression_model assigned")


    def __init_backmapping_model__(self,backmapping_model):
        """Private method to initilaize the regression model.
        """
        if backmapping_model == None:
            raise ValueError("There is no backmappign model")

        self.backmapping_model = backmapping_model  

    def predict(self):
        """
        Return the predicted output from the X_test input
        """
        return self.y_pred

    def fit_transform(self,X_train,y_train,X_test,y_test):
        """
        Performs the training loop for the ISOMAP ROM

        ----------
        Parameters
        ----------
        regression_model:str -> regression_model for the ISOMAP ROM
        X_train: training features
        y_train: training labels
        X_test: testing features
        y_test: testing labels

        """
        self.y_iso_train = self.embedding.fit_transform(y_train)
        self.regression_model.fit(X_train,self.y_iso_train)
        self.y_iso_pred = self.regression_model.predict(X_test)
        self.backmapping_model.fit(y_train,self.y_iso_train,self.y_iso_pred)               
        self.y_pred = self.backmapping_model.predict()

    def get_params(self):
        """
        Returns the global parameters of the model. 
        The first element is the regression model parameters. The second model is the Isomap embedding parameters
        """
        return self.regression_model.get_params(), self.embedding.get_params()
    
