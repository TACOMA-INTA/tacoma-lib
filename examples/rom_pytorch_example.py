import torch
from torch import nn, optim
from tacoma.rom import IsomapRegressor
from tacoma.metrics import GetScores
from tacoma.interpolator import KNeighborsBackmap
import numpy as np
rng = np.random.RandomState(42)

X_train = rng.random_sample(size = (100,2))
y_train = rng.random_sample(size = (100,111_000))
X_test = rng.random_sample(size = (10,2))
y_test = rng.random_sample(size = (10,111_000))

class RomDNN:
    """
    Helper class to run the model inside the tacoma.rom classes. In future version, this will be added inside the library
    """
    def __init__(self,torch_model = None,epochs = 5000,criterion = None,optimizer = None,verbose = False):
        self.torch_model = torch_model # must be initialized beforehand
        self.epochs = epochs
        self.__init_dnn_params__(criterion,optimizer)
        self.verbose = verbose
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def __repr__(self):
      return self.device
    def __init_dnn_params__(self,criterion,optimizer):
        self.criterion = criterion 
        if criterion == None:
            self.criterion = nn.MSELoss
        self.optimizer = optimizer
        if optimizer == None:
            self.optimizer = optim.Adam
    def __to_tensor__(self,arr):
        """
        """
        arr = torch.from_numpy(arr).float()
        arr = arr.to(self.device)
        return arr
    @staticmethod
    def __to_numpy__(arr):
        arr = arr.cpu().numpy()
        return arr
    def __run__(self,X_train,y_train):
        """
        Train the model and run the model to obtain the results
        """
        model = self.torch_model
        model.to(self.device)
        optimizer = self.optimizer(model.parameters(),lr = 1e-4,weight_decay = 1e-5)
        criterion = self.criterion()
        for i in range(self.epochs):
            y_pred = model(self.__to_tensor__(X_train))
            loss = criterion(y_pred,self.__to_tensor__(y_train))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.detach().cpu().numpy()
            if self.verbose:
              print(f"Epoch: {i}, Loss:{loss_val}")
        print(f"Final Loss: {loss_val}")
        return model

    def fit(self,X_train,y_train):
        self.trained_model = self.__run__(X_train,y_train)
    def predict(self,X_test):
        with torch.no_grad():
            y_pred = self.trained_model(self.__to_tensor__(X_test))
            return self.__to_numpy__(y_pred)

class MultiLayerPerceptron(nn.Module):
    """
    Multi Layer Perceptron based on the usage of ReLU activation functions
    """
    def __init__(self, input_size, output_size,n_layers,size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.input_layer = nn.Linear(input_size,size)
        self.layers.append(self.input_layer)
        self.layers.append(nn.ReLU())
        # hidden layers
        for i in range(n_layers):
            self.layers.append(nn.Linear(size, size))
            self.layers.append(nn.ReLU())

        self.output_layer = nn.Linear(size,output_size)
        self.layers.append(self.output_layer)

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data

dnn_model = MultiLayerPerceptron(2,2,10,10) 
reg_model = RomDNN(dnn_model,epochs = 5_000,verbose = True) # helper class to encapsulate the training loop 
map_model = KNeighborsBackmap(5)
model = IsomapRegressor(n_components=2,
                        n_neighbors=15,
                        regression_model=reg_model,
                        backmapping_model=map_model)
model.fit_transform(X_train,y_train,X_test,y_test)
scores = GetScores(y_test,model.predict())
print(f"r2:{scores.r2_score().mean()}\t mse: {scores.mse_error().mean()}")