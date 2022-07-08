import pandas as pd

from calendar import month_abbr
from math import pi
import math
import numpy as np


from sklearn.base import BaseEstimator, TransformerMixin

from lightgbm import LGBMClassifier

import warnings





class CustomTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for feat in self.feature_names:
            X[feat + '_quadratic'] = self.quadratic_transformation(X[feat])
            X[feat + '_cubic'] = self.cubic_transformation(X[feat])
            X[feat + '_log'] = self.log_transformation(X[feat])
            X[feat + '_root'] = self.root_transformation(X[feat])
        self.new_df = X
        return X

    def quadratic_transformation(self, x_col):
        return (x_col) ** 2

    def cubic_transformation(self, x_col):
        return (x_col) ** 3

    def log_transformation(self, x_col):
        return np.log(x_col + 0.0001)

    def root_transformation(self, x_col):
        return 2 * np.sqrt(x_col)

    def get_feature_names(self):
        return self.new_df.columns.tolist()


