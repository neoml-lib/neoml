"""This sample finds optimal parameters and performs
classification of the 20newsgroups dataset
"""

__copyright__ = """

Copyright © 2017-2021 ABBYY Production LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

__license__ = 'Apache 2.0'

import neoml
import numpy as np
import itertools

# Get data
from sklearn.datasets import fetch_20newsgroups_vectorized

train_data = fetch_20newsgroups_vectorized(subset='train')
test_data = fetch_20newsgroups_vectorized(subset='test')


def accuracy(model, X, y):
    """Returns the accuracy of model on the given data"""
    correct = sum(1 for label, probs in zip(y, model.classify(X))
                  if label == np.argmax(probs))
    return float(correct)/len(y)


def grid_search(init_classifier, X, y, param_grid, n_folds=3):
    """Searches for the most optimal parameters in the grid
    Returns trained model and optimal parameters
    """
    best_params = {}

    if param_grid:  # Avoid corner case when param_grid is empty
        param_names, param_values_lists = zip(*param_grid.items())
        best_acc = -1.
        for param_values in itertools.product(*param_values_lists):
            params = dict(zip(param_names, param_values))
            classifier = init_classifier(**params)
            acc = neoml.CrossValidation.cross_validation_score(classifier, X, y, parts=5).mean()
            if acc > best_acc:
                best_acc = acc
                best_params = params

    best_classifier = init_classifier(**best_params)
    return best_classifier.train(X, y), best_params


param_grid = {
    'loss': ['binomial', 'squared_hinge', 'smoothed_hinge'],
    'l1_reg': [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
}

# It will take some time...
model, params = grid_search(neoml.Linear.LinearClassifier, train_data.data,
                            train_data.target, param_grid)

print('Best params: ', params)
print(f'Accuracy: {accuracy(model, test_data.data, test_data.target):.4f}')
