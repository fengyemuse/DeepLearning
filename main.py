from sklearn import datasets
import numpy as np
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

dataset = datasets.load_boston()
x = dataset.data
y = dataset.target
seed = 7
np.random.seed(seed)


def create_model(units_list=None, opitimizer='adam'):
    if units_list is None:
        units_list = [13]
    model = tf.keras.Sequential()
    for unit in units_list:
        model.add(tf.keras.layers.Dense(units=unit, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer=opitimizer)
    return model


Model = KerasRegressor(build_fn=create_model, epochs=200, batch_size=5, verbose=0)
steps = [('standardize', StandardScaler()), ('mlp', Model)]
pipline = Pipeline(steps)

k10fold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipline, x, y, cv=k10fold)
print('Baseline: %f (%f) MSE' % (results.mean(), results.std()))
