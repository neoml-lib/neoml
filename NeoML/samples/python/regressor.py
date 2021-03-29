"""This sample finds optimal parameters and performs
regression over Boston house-prices dataset
"""

import neoml
import numpy as np
import itertools
from sklearn.datasets import load_boston


# Get data
X, y = load_boston(return_X_y=True)

# Convert data type
X = X.astype(np.float32)
y = y.astype(np.float32)

# Split into train/test
test_size = 50
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]


def mse(a, b):
    """Mean squared error of 2 numpy arrays"""
    return ((a - b) ** 2).mean()


def cv_iterator(X, y, n_folds):
    """Returns X_train, y_train, X_test, y_test for each of the folds"""
    data_size = len(y)
    test_size = data_size // n_folds
    for i in range(n_folds):
        train = list(itertools.chain(range(i*test_size),
                                     range((i+1)*test_size, data_size)))
        test = range(i*test_size, (i+1)*test_size)
        yield X[train], y[train], X[test], y[test]


def grid_search(X, y, param_grid, n_folds=3):
    """Searches for the most optimal parameters in the grid
    Returns trained model and optimal parameters
    """
    best_params = {}

    if param_grid:  # Avoid corner case when param_grid is empty
        param_names, param_values_lists = zip(*param_grid.items())
        best_mse = 2. ** 32
        for param_values in itertools.product(*param_values_lists):
            kwargs = dict(zip(param_names, param_values))
            linear = neoml.Linear.LinearRegressor(**kwargs)
            folds = 20  # dataset is very tiny
            avg_mse = 0.
            # Calculate average MSE for K-folds
            for X_train, y_train, X_test, y_test in cv_iterator(X, y, folds):
                model = linear.train(X_train, y_train)
                avg_mse += mse(y_test, model.predict(X_test))
            # Update params if MSE is less
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_params = kwargs

    best_linear = neoml.Linear.LinearRegressor(**best_params)
    return best_linear.train(X, y), best_params


# Param grid
param_grid = {
    'error_weight': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 1e1, 1e2],
    'l1_reg': [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'thread_count': [6],
}

# Search for optimal parameters
model, params = grid_search(X_train, y_train, param_grid)

# Print result
print('Best params: ', params)
print(f'Test MSE: {mse(y_test, model.predict(X_test)):.3f}')
