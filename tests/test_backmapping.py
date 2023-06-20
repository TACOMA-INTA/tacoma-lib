import numpy as np
from tacoma.interpolator import TaylorKNNBackmap
from sklearn.manifold import Isomap


rng = np.random.RandomState(0)
X_train = rng.random_sample(size = (10,2))
X_test = rng.random_sample(size = (10,2))
y_train= rng.random_sample(size = (10,111_000))
y_test = rng.random_sample(size = (10,111_000))
iso = Isomap(n_components=2)

y_train_iso = iso.fit_transform(y_train)
y_test_iso = iso.transform(y_test)


def test_taylorknn():
    model = TaylorKNNBackmap()
    model.fit(X_train, y_train_iso)
    y_pred = model.predict(X_test)



