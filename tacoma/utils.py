import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import Isomap
from sklearn.metrics import r2_score
from sklearn.neighbors import kneighbors_graph


def weigthed_KNN(dist_neigh,epsilon=0.01,k=4):
    """
    Function to return KNN weigths using the test point's distances to it's k neighbours in Low-dimensional space
    to be used in sklearn.neighbors.KNeighborsRegressor funcion

    N: Number of used nearest neighbors
    Weigths Computation (To decrease the effect of further away neighbors) using:
      1. The scaled distances (Devided by max distance) (0, 1]
      2. Two penalty parameters:
          2.1. k: power to scaled distances (1 < k, integer) (Best practice: 4)
          2.2. epsilon: multipily to final weigths (0 < epsilon << 1) (Best practice: 0.01)

    author: EHSAN FARZAMNIK <efarzamn@pa.uc3m.es>
    """
    n_neighbors = dist_neigh.shape[1]
    n_predictions = dist_neigh.shape[0]
    
    max_dist = np.amax(dist_neigh, axis=1)
    max_dist = max_dist.reshape(-1, 1)
    dist_neigh_scaled = dist_neigh/max_dist[:,]
    C = epsilon*dist_neigh_scaled**k
    G =  np.zeros([n_neighbors, n_neighbors])
    G_bar = np.zeros([n_neighbors, n_neighbors])
    w = np.zeros([n_predictions, n_neighbors])

    for pred in range(n_predictions):
        for i in range(n_neighbors):
            for j in range(n_neighbors):
                G[i,j] = (dist_neigh[pred,i])*(dist_neigh[pred,j])
        G_bar = G/np.amax(G) + np.diag(C[pred,])

        I_ones = np.ones([n_neighbors, n_neighbors], dtype = int)
        I_zeros = np.zeros([n_neighbors, n_neighbors], dtype=int)
        I_ones_vec = np.ones([n_neighbors, 1], dtype = int)
        I_zeros_vec = np.zeros([n_neighbors, 1], dtype=int)

        A = np.block([[2*G_bar, I_ones_vec], [I_ones_vec.T, 0]])
        B = np.block([[I_zeros_vec], [1]])
        optimal_solutions = np.linalg.solve(A,B) # be careful! It may cause problems with Singular Matrix
        w[pred,:] = optimal_solutions[0:n_neighbors,:].ravel()
    return w

def Isomap_max_kneighbors(X,**kwargs):
    """
    Determination of maximum k-neighbours for isomap computarion. Based on Samko et al. (2006)
    2E/N <= k +2
    being E the number of edges and N the number of nodes in the neighbour graph G.
    
    ----------
    Parameters
    ----------
    Attributes:
    - X -> snapshot matrix (each column is an snapshot)
    Output:
    - kmax -> maximum number of neighbors to perform isomap
    """
    criteria = True
    k = 2
    while criteria:
        k += 1
        A = kneighbors_graph(X, k, mode='connectivity', include_self=True,**kwargs)
        nodes = A.shape[0]
        edges = np.sum(np.triu(A.toarray(),k=1))
        criteria = ((k +2 )>= 2*edges/nodes) & (k>X.shape[1]) # Force stop in case K bigger than the number of samples

    kmax = k-1
    return kmax

class ResidualVariance:
    """
    Class for computing the residual variance of the embedding of X array in an Isomap with different
    parameters. More info at https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html
    
    Parameters
    ----------
    X: array 
        Array of values to be embedded in the Isomap
    n_components: list or range, default = range(1,10)
        Number of coordianates for the manifold
    n_neighbors: list or range, default = range(3,10,2)
        Number of neighbors to consider for each point. If `n_neighbors` is an int,
        then `radius` must be `None`.
    
    """
    def __init__(self,X,n_components = range(1,10), n_neighbors = range(3,10,2),**kwargs):
        self.X = X
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.kwargs = kwargs
        self.__run__()

    def __run__(self):
        RV = np.zeros([len(self.n_components),len(self.n_neighbors)]) # Initialize RV matrix
        for i, component in enumerate(self.n_components):
            for j, neighbor in enumerate(self.n_neighbors):
                embedding = Isomap(n_components=component,n_neighbors=neighbor,**self.kwargs)
                X_iso = embedding.fit_transform(self.X)
                G = embedding.dist_matrix_
                G_iso = np.linalg.norm(X_iso-X_iso[:,None], axis = -1) # Eucledian distance in manifold
                rv = r2_score(G.ravel(),G_iso.ravel())
                RV[i,j] = 1 -rv
        self.RV = RV

    def get_results(self):
        """
        Retunrs a pd.DataFrame with the results of computing the residual variance for all diferent cases of neighbords and componentes of the Isomap
        """
        df = pd.DataFrame(
            self.RV,
            columns =  [f"kneigh = {k}" for k in self.n_neighbors],
            index = [f"Niso = {n}" for n in self.n_components]
        )
        return df

    def plot(self):
        """
        Plots the residual variance in the 2D space
        """
        fig, ax = plt.subplots()
        ax.plot(self.n_components,self.RV,)
        ax.plot(self.n_components,self.RV, label=[f"k = {k}" for k in self.n_neighbors])
        ax.legend()
        return fig

class PODTruncation:
    """
    Computes the number of n_PODmodes necesarry to reconstruc the X% percentage of the energy

    :params X:
    """
    def __init__(self,X,energy_percentage = 0.99):
        self. X = X
        self.energy_percentage = energy_percentage
        self.__run__()
    def __repr__(self):
        return f"X shape: {self.X.shape}\n Percentage to reconstruc:{self.energy_percentage}\nNumber of nodes:{self.node_trunc}"
    def __run__(self):
        """
        Runs the SVD and obtains the number of modes to get the 99% of the total energy
        """
        self.U,self.S,self.Vh = np.linalg.svd(self.X.T, full_matrices=False)
        nmodes = range(self.X.shape[1]) 
        self.error_recon = np.cumsum(self.S)/np.sum(self.S)
        idx = int(np.argmin((self.error_recon-self.energy_percentage)**2))
        self.node_trunc = nmodes[idx]

    def plot(self):
        """
        Plots the reconstruction error of adding a different number of modes
        """
        fig,ax = plt.subplots() 
        ax.plot(self.error_recon)
        ax.plot(self.node_trunc-1,self.error_recon[self.node_trunc-1],"ro")
        return fig
