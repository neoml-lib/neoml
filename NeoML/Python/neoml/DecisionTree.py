""" Copyright (c) 2017-2024 ABBYY

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--------------------------------------------------------------------------------------------------------------
"""

import numpy as np
from .Utils import convert_data, get_data
import neoml.PythonWrapper as PythonWrapper


class DecisionTreeClassificationModel:
    """Decision tree classification model.
    """

    def __init__(self, internal):
        self.internal = internal

    def classify(self, X):
        """Gets the classification results for the input sample.

        :param X: the input vectors, put into a matrix. The values will be 
            converted to ``dtype=np.float32``. If a sparse matrix is
            passed in, it will be converted to a sparse ``csr_matrix``.
        :type X: {array-like, sparse matrix} of shape (n_samples, n_features)

        :return: the predictions of class probability for each input vector.
        :rtype: *generator of ndarray of shape (n_samples, n_classes)*
        """
        x = convert_data(X)
        return self.internal.classify(*get_data(x))


class DecisionTreeClassifier(PythonWrapper.DecisionTree):
    """Decision tree classifier.

    :param criterion: the type of criterion to be used for subtree splitting.
    :type criterion: str, {'gini', 'information_gain'}, default='gini'

    :param min_subset_size: the minimum number of vectors corresponding to a node subtree.
    :type min_subset_size: int, default=1

    :param min_subset_part: the minimum weight of the vectors in a subtree relative to the parent node weight.
    :type min_subset_part: float, [0..1], default=0.0

    :param min_split_size: the minimum number of vectors in a node subtree when it may be divided further.
    :type min_split_size: int, default=1

    :param max_tree_depth: the maximum depth of the tree.
    :type max_tree_depth: int, default=32

    :param max_node_count: the maximum number of nodes in the tree.
    :type max_node_count: int, default=4096

    :param const_threshold: if the ratio of same class elements in the subset is greater than this value,
        a constant node will be created.
    :type const_threshold: float, [0..1], default=0.99

    :param random_selected_feature_count: no more than this number of randomly selected features will be used for each node.
        -1 means use all features every time.
    :type random_selected_feature_count: int, default=-1

    :param available_memory: the memory limit for the algorithm (default is 1 Gigabyte)
    :type available_memory: int, default=1024*1024*1024

    :param multiclass_mode: determines how to handle multi-class classification
    :type multiclass_mode: str, ['single_tree', 'one_vs_all', 'one_vs_one'], default='single_tree'
    """

    def __init__(self, criterion='gini', min_subset_size=1, min_subset_part=0.0, min_split_size=1, max_tree_depth=32,
                 max_node_count=4096, const_threshold=0.99, random_selected_feature_count=-1, available_memory=1024*1024*1024,
                 multiclass_mode='single_tree'):

        if criterion != 'gini' and criterion != 'information_gain':
            raise ValueError('The `criterion` must be one of: `gini`, `information_gain`.')
        if min_subset_size < 1:
            raise ValueError('The `min_subset_size` must be > 0.')
        if min_subset_part > 1 or min_subset_part < 0:
            raise ValueError('The `min_subset_part` must be in [0, 1].')
        if min_split_size < 1:
            raise ValueError('The `min_split_size` must be > 0.')
        if max_tree_depth <= 0:
            raise ValueError('The `max_tree_depth` must be > 0.')
        if max_node_count <= 1:
            raise ValueError('The `max_node_count` must be > 1.')
        if const_threshold > 1 or const_threshold < 0:
            raise ValueError('The `const_threshold` must be in [0, 1].')
        if random_selected_feature_count <= 0 and random_selected_feature_count != -1:
            raise ValueError('The `random_selected_feature_count` must be > 0 or -1.')
        if available_memory < 0:
            raise ValueError('The `available_memory` must be non-negative.')
        if multiclass_mode != 'single_tree' and multiclass_mode != 'one_vs_all' and multiclass_mode != 'one_vs_one':
            raise ValueError('The `multiclass_mode` must be one of: `single_tree`, `one_vs_all`, `one_vs_one`.')

        super().__init__(int(min_subset_size), float(min_subset_part), int(min_split_size), int(max_tree_depth),
                         int(max_node_count), criterion, float(const_threshold), int(random_selected_feature_count),
                         int(available_memory), multiclass_mode)

    def train(self, X, Y, weight=None):
        """Trains the decision tree.

        :param X: the training sample. The values will be converted 
            to ``dtype=np.float32``. If a sparse matrix is
            passed in, it will be converted to a sparse ``csr_matrix``.
        :type X: {array-like, sparse matrix} of shape (n_samples, n_features)

        :param Y: correct class labels (``int``) for the training set vectors.
        :type Y: array-like of shape (n_samples,)

        :param weight: sample weights. If None, then samples are equally weighted.
        :type weight: array-like of shape (n_samples,), default=None

        :return: the trained classification model.
        :rtype: neoml.DecisionTree.DecisionTreeClassificationModel
        """

        x = convert_data(X)
        y = np.asarray(Y, dtype=np.int32, order='C')

        if x.shape[0] != y.size:
            raise ValueError('The `X` and `Y` inputs must be the same length.')

        if weight is None:
            weight = np.ones(y.size, np.float32, order='C')
        else:
            weight = np.asarray(weight, dtype=np.float32, order='C')

        if np.any(y < 0):
            raise ValueError('All `Y` elements must be >= 0.')

        if np.any(weight < 0):
            raise ValueError('All `weight` elements must be >= 0.')

        return DecisionTreeClassificationModel(super().train_classifier(*get_data(x), int(x.shape[1]), y, weight))
