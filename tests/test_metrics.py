import numpy as np
from tacoma.metrics import GetScores
import os
import pytest


rng = np.random.RandomState(0)
y_true = rng.random_sample(size = (10,111_000))
y_pred = rng.random_sample(size = (10,111_000))
X = rng.random_sample(size = (10,2))
y_rom = rng.random_sample(size = (2,111_000))

def test_GetScores():
    scores = GetScores(y_true,y_pred,y_rom)
    scores.to_dataframe()
    scores.to_npz()
    scores.to_mat(X) # X cannot be None

    for file in ["training_results.npz","training_results.mat"]:
        os.remove(file)
