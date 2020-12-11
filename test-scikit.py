import numpy as np
import pandas as pd
import matplotlib.pylab as plt

#from sklearn.datasets import load_boston
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import LinearRegession
#from sklearn.metrics import mean_squared_error
#
#from sklego.metrics import p_percent_score
#from sklearn.linear_model import LogisticRegression
#
#
#print('p_percent_score:', p_percent_score(sensitive_column="x2")(mod_unfair, X))
## Loading dataset
#X, y = load_boston(return_X_y=True)
#pipe = Pipeline([
#    ('scale', StandardScaler()),
#    ('model', LinearRegession())
#    ])
#
## Measuring fariness for Regression is different from measuring fairnes for classification
## Prediction has a distribution independent if group A and group B
#sensitive_classification_dataset = pd.DataFrame({
#    "x1": [1, 0, 1, 0, 1, 0, 1, 1],
#    "x2": [0, 0, 0, 0, 0, 1, 1, 1],
#    "y": [1, 1, 1, 0, 1, 0, 0, 0]}
#)
#
#X, y = sensitive_classification_dataset.drop(columns='y'), sensitive_classification_dataset['y']
#mod_unfair = LogisticRegression(solver='lbfgs').fit(X, y)

from sklego.metrics import equal_opportunity_score
from sklearn.linear_model import LogisticRegression
import types

sensitive_classification_dataset = pd.DataFrame({
    "x1": [1, 0, 1, 0, 1, 0, 1, 1],
    "x2": [0, 0, 0, 0, 0, 1, 1, 1],
    "y": [1, 1, 1, 0, 1, 0, 0, 1]}
)

X, y = sensitive_classification_dataset.drop(columns='y'), sensitive_classification_dataset['y']

mod_1 = types.SimpleNamespace()

mod_1.predict = lambda X: np.array([1, 0, 1, 0, 1, 0, 1, 1])
print(mod_1)
print('equal_opportunity_score:', equal_opportunity_score(sensitive_column="x2")(mod_1, X, y))

mod_1.predict = lambda X: np.array([1, 0, 1, 0, 1, 0, 0, 1])
print('equal_opportunity_score:', equal_opportunity_score(sensitive_column="x2")(mod_1, X, y))

mod_1.predict = lambda X: np.array([1, 0, 1, 0, 1, 0, 0, 0])

print('equal_opportunity_score:', equal_opportunity_score(sensitive_column="x2")(mod_1, X, y))
