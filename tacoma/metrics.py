from sklearn import metrics
import pandas as pd
import numpy as np
from scipy.io import savemat


class GetScores:
    """Class for obtainint the results of the regression for multiouput regressions

    # TODO: add more erroe metrics from the sklearn library
    # TODO: add custom error metrics helper functions

    """
    def __init__(self, y_true:np.array, y_pred:np.array,y_rom:np.array= None)->None:
        """
    :params y_true: array of true labels
    :params y_pred: array of predicted labels
    :params X: array of features of the labels. The columns must follow the order [Mach, Alpha]
    :params y_rom: array of ROM labels
        """
        if not y_true.shape == y_pred.shape:
            raise ValueError("Shapes of inputs array do not coincide")
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_rom = y_rom
        self.arr = np.append(y_true,y_pred,axis = 1)
        self.union_index = self.y_true.shape[1]
    

    def __max_error__(self,arr,**kwargs):
        """
        Private method. Implement the max_error metric from Sklearn for np.apply_along_axis func
        """
        
        return metrics.max_error(arr[0:self.union_index],arr[self.union_index:],**kwargs)
    

    def __mse_error__(self,arr,**kwargs):
        """
        Private method. Implement the mean_squared_error metric from Sklearn for np.apply_along_axis func
        """
        return metrics.mean_squared_error(arr[0:self.union_index],arr[self.union_index:],**kwargs)
    

    def __r2_score__(self,arr,**kwargs):
        """
        Private method. Implement the r2_score metric from Sklearn for np.apply_along_axis func
        """
        return metrics.r2_score(arr[0:self.union_index],arr[self.union_index:],**kwargs)
        
    
    def r2_score(self):
        """
        Returns the r2 score of all the cases
        """
        
        return np.apply_along_axis(self.__r2_score__,1,self.arr)
        
    def mse_error(self):
        """
        Returns the Mean Squared Error of all the cases
        """
        return np.apply_along_axis(self.__mse_error__,1,self.arr)


    def max_error(self):
        """
        Returns the Max Squared Error of all the cases
        """
        return np.apply_along_axis(self.__max_error__,1,self.arr)
    

    def to_dataframe(self):
        """
        Return a pd.DataFrame with all the errors per case
        """
        results = {
            "r2": self.r2_score(),
            "mse":self.mse_error(),
            "me":self.max_error()
        }
        results = pd.DataFrame(results,index = list(range(len(self.r2_score()))))
        return results
        

    def to_mat(self,X=None,fname = "training_results.mat"):
        """
        Export the inputs to .mat format. It should contain X and y_rom.

        :params fname: name of the file
        """
        if type(X) == None:
            raise ValueError("Use to_npz method if you dont want to add X to your data file")
        if type(self.y_rom) == None:
            savemat(fname,
                {
                    "X":X,
                    "y_true":self.y_true,
                    "y_pred":self.y_pred,
                }
                    )
        else:
            savemat(fname,
                    {
                        "X":X,
                        "y_true":self.y_true,
                        "y_pred":self.y_pred,
                        "y_rom":self.y_rom
                    }
                    )
    def to_npz(self,X=None,fname = "training_results.npz"):
        """
        Export the inputs to .npz format. It should contain X and y_rom.

        :params fname: name of the file
        """
        np.savez(fname,
                 X = X,
                 y_true = self.y_true,
                 y_pred = self.y_pred,
                 y_rom = self.y_rom)

        
