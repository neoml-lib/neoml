""" Copyright (c) 2017-2021 ABBYY Production LLC

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

import numpy
from scipy.sparse import csr_matrix
import neoml.PythonWrapper as PythonWrapper


class DecisionTreeClassificationModel:
    """Decision tree classification model.
    """

    def __init__(self, internal):
        self.internal = internal

    def classify(self, X):
        """Gets the classification results for the input sample.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input vectors, put into a matrix. The values will be 
            converted to ``dtype=np.float32``. If a sparse matrix is
            passed in, it will be converted to a sparse ``csr_matrix``.

        Return values
        -------
        predictions : generator of ndarray of shape (n_samples, n_classes)
            The predictions of class probability for each input vector.
        """
        x = csr_matrix(X, dtype=numpy.float32)
        return self.internal.classify(x.indices, x.data, x.indptr)


class DecisionTreeClassifier(PythonWrapper.DecisionTree):
    """Decision tree classifier.
    Parameters
    ----------
    criterion : {'gini', 'information_gain'}, default='gini'
        The type of criterion to be used for subtree splitting.

    min_subset_size : int, default=1
        The minimum number of vectors corresponding to a node subtree. 

    min_subset_part : float, [0..1], default=0.0
        The minimum weight of the vectors in a subtree relative to the parent node weight.

    min_split_size : int, default=1
        The minimum number of vectors in a node subtree when it may be divided further.

    max_tree_depth : int, default=32
        The maximum depth of the tree.

    max_node_count : int, default=4096
        The maximum number of nodes in the tree.

    const_threshold : float, [0..1], default=0.99
        If the ratio of same class elements in the subset is greater than this value,
        a constant node will be created.

    random_selected_feature_count : int, default=-1
        No more than this number of randomly selected features will be used for each node.
        -1 means use all features every time.
    """

    def __init__(self, criterion='gini', min_subset_size=1, min_subset_part=0.0, min_split_size=1, max_tree_depth=32,
                 max_node_count=4096, const_threshold=0.99, random_selected_feature_count=-1):

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

        super().__init__(int(min_subset_size), float(min_subset_part), int(min_split_size), int(max_tree_depth),
                         int(max_node_count), criterion, float(const_threshold), int(random_selected_feature_count))

    def train(self, X, Y, weight=None):
        """Trains the decision tree.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training sample. The values will be converted 
            to ``dtype=np.float32``. If a sparse matrix is
            passed in, it will be converted to a sparse ``csr_matrix``.

        Y : array-like of shape (n_samples,)
            Correct class labels (``int``) for the training set vectors.

        weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Return values
        -------
        model : object
            The trained ``DecisionTreeClassificationModel``.
        """

        x = csr_matrix(X, dtype=numpy.float32)
        y = numpy.array(Y, dtype=numpy.int32, copy=False)

        if x.shape[0] != y.size:
            raise ValueError('The `X` and `Y` inputs must be the same length.')

        if weight is None:
            weight = numpy.ones(y.size, numpy.float32)
        else:
            weight = numpy.array(weight, dtype=numpy.float32, copy=False)

        if numpy.any(y < 0):
            raise ValueError('All `Y` elements must be >= 0.')

        if numpy.any(weight < 0):
            raise ValueError('All `weight` elements must be >= 0.')

        return DecisionTreeClassificationModel(super().train_classifier(x.indices, x.data, x.indptr, int(x.shape[1]),
                                                                         y, weight))
