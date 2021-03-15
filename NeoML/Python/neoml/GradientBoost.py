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
from .Utils import convert_data, get_data
import neoml.PythonWrapper as PythonWrapper

class GradientBoostClassificationModel:
    """Gradient boosting classification model.
    """
    def __init__(self, value):
        if type(value) == type('str'):
            self.internal = PythonWrapper.Model(value)
        else:
            self.internal = value


    def store(self, path):
        self.internal.store(path)       

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
        x = convert_data(X)
        return self.internal.classify(*get_data(x))

class GradientBoostClassifier(PythonWrapper.GradientBoost):
    """Gradient boosting for classification.
    
    Gradient boosting method creates an ensemble of decision trees
    using random subsets of features and input data.

    Parameters
    ----------  
    loss : {'exponential', 'binomial', 'squared_hinge', 'l2'}, default='binomial'
        The loss function to be optimized. 
        'binomial' refers to deviance (= logistic regression) for classification
        with probabilistic outputs. 
        'exponential' is similar to the AdaBoost algorithm.

    iteration_count : int, default=100
        The maximum number of iterations (that is, the number of trees in the ensemble).

    learning_rate : float, default=0.1
        The multiplier for each classifier tree.
        There is a trade-off between ``learning_rate`` and ``iteration_count``.

    subsample : float, [0..1], default=1.0
        The fraction of input data that is used for building one tree.

    subfeature : float, [0..1], default=1.0
        The fraction of features that is used for building one tree.

    random_seed : int, default=0
        The random generator seed number.

    max_depth : int, default=10
        The maximum depth of a tree in ensemble.

    max_node_count : int, default=-1
        The maximum number of nodes in a tree. -1 means no limitation.

    l1_reg : float, default=0.0
        The L1 regularization factor.

    l2_reg : float, default=1.0
        The L2 regularization factor.

    prune : float, default=0.0
        The value of criterion difference when the nodes should be merged.
        The 0 default value means never merge nodes.

    thread_count : int, default=1
        The number of processing threads to be used while training the model.

    builder_type : {'full', 'hist', 'multi_full'}, default='full'
        The type of tree builder used. 
        ``full`` means all feature values are used for splitting nodes.
        ``hist`` means the steps of a histogram created from feature values
        ``multi_full`` means 'full' with multiclass trees
        will be used for splitting nodes.

    max_bins : int, default=32
        The largest possible histogram size to be used in ``hist`` mode.

    min_subtree_weight : float, default=0.0
        The minimum subtree weight. The 0 default value means no lower limit.
    """

    def __init__(self, loss='binomial', iteration_count=100, learning_rate=0.1,
        subsample=1.0, subfeature=1.0, random_seed=0, max_depth=10,
        max_node_count=-1, l1_reg=0.0, l2_reg=1.0, prune=0.0, thread_count=1,
        builder_type='full', max_bins=32, min_subtree_weight=0.0):

        if loss != 'binomial' and loss != 'exponential' and loss != 'squared_hinge' and loss != 'l2':
            raise ValueError('The `loss` must be one of: `exponential`, `binomial`, `squared_hinge`, `l2`.')
        if builder_type not in ('full', 'hist', 'multi_full'):
            raise ValueError('The `builder_type` must be one of: `full`, `hist`, `multi_full`.')
        if iteration_count <= 0:
            raise ValueError('The `iteration_count` must be > 0.')
        if subsample < 0 or subsample > 1:
            raise ValueError('The `subsample` must be in [0..1].')
        if subfeature < 0 or subfeature > 1:
            raise ValueError('The `subfeature` must be in [0..1].')
        if max_depth < 0:
            raise ValueError('The `max_depth` must be >= 0.')
        if max_node_count < 0 and max_node_count != -1:
            raise ValueError('The `max_node_count` must be >= 0 or equal to -1.')
        if prune < 0:
            raise ValueError('The `prune` must be >= 0.')
        if thread_count <= 0:
            raise ValueError('The `thread_count` must be > 0.')
        if min_subtree_weight < 0:
            raise ValueError('The `min_subtree_weight` must be >= 0.')

        super().__init__(loss, int(iteration_count), float(learning_rate), float(subsample), float(subfeature), int(random_seed), int(max_depth),
            int(max_node_count), float(l1_reg), float(l2_reg), float(prune), int(thread_count), builder_type, int(max_bins), float(min_subtree_weight))

    def train(self, X, Y, weight=None):
        """Trains the gradient boosting model for classification.
        
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
            The trained ``GradientBoostClassificationModel``.
        """
        x = convert_data( X )
        y = numpy.array( Y, dtype=numpy.int32, copy=False )

        if x.shape[0] != y.size:
            raise ValueError('The `X` and `Y` inputs must be the same length.')

        if weight is None:
            weight = numpy.ones(y.size, numpy.float32)
        else:
            weight = numpy.array( weight, dtype=numpy.float32, copy=False )

        if numpy.any(y < 0):
            raise ValueError('All `Y` elements must be >= 0.')

        if numpy.any(weight < 0):
            raise ValueError('All `weight` elements must be >= 0.')

        return GradientBoostClassificationModel(super().train_classifier(*get_data(x), int(x.shape[1]), y, weight)) 

class GradientBoostRegressionModel:
    """Gradient boosting regression model.
    """
    def __init__(self, value):
        if type(value) == type('str'):
            self.internal = PythonWrapper.Model(value)
        else:
            self.internal = value


    def store(self, path):
        self.internal.store(path)       

    def predict(self, X):
        """Predicts the value of the function.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input vectors. The values will be converted 
            to ``dtype=np.float32``. If a sparse matrix is
            passed in, it will be converted to a sparse ``csr_matrix``.

        Return values
        -------
        predictions : generator of ndarray of shape (n_samples)
            The predictions of the function value on each input vector.
        """
        x = convert_data(X)
        return self.internal.predict(*get_data(x))

class GradientBoostRegressor(PythonWrapper.GradientBoost):
    """Gradient boosting for regression.
    Gradient boosting method creates an ensemble of decision trees
    using random subsets of features and input data.

    Parameters
    ----------
    loss : {'l2'}, default='l2'
        The loss function to be optimized. 
        The quadratic loss L2 is the only one supported.

    iteration_count : int, default=100
        The maximum number of iterations (that is, the number of trees in the ensemble).

    learning_rate : float, default=0.1
        The multiplier for each tree.
        There is a trade-off between ``learning_rate`` and ``iteration_count``.

    subsample : float, [0..1], default=1.0
        The fraction of input data that is used for building one tree.

    subfeature : float, [0..1], default=1.0
        The fraction of features that is used for building one tree.

    random_seed : int, default=0
        The random generator seed number.

    max_depth : int, default=10
        The maximum depth of a tree in ensemble.

    max_node_count : int, default=-1
        The maximum number of nodes in a tree. -1 means no limitation.

    l1_reg : float, default=0.0
        The L1 regularization factor.

    l2_reg : float, default=1.0
        The L2 regularization factor.

    prune : float, default=0.0
        The value of criterion difference when the nodes should be merged.
        The 0 default value means never merge nodes.

    thread_count : int, default=1
        The number of processing threads to be used while training the model.

    builder_type : {'full', 'hist'}, default='full'
        The type of tree builder used. 
        ``full`` means all feature values are used for splitting nodes.
        ``hist`` means the steps of a histogram created from feature values
        will be used for splitting nodes.

    max_bins : int, default=32
        The largest possible histogram size to be used in ``hist`` mode.

    min_subtree_weight : float, default=0.0
        The minimum subtree weight. The 0 default value means no lower limit.
    """

    def __init__(self, loss='l2', iteration_count=100, learning_rate=0.1,
        subsample=1.0, subfeature=1.0, random_seed=0, max_depth=10,
        max_node_count=-1, l1_reg=0.0, l2_reg=1.0, prune=0.0, thread_count=1,
        builder_type='full', max_bins=32, min_subtree_weight=0.0):

        if loss != 'l2':
            raise ValueError('The `loss` must be `l2` for regression.')
        if builder_type not in ('full', 'hist'):
            raise ValueError('The `builder_type` must be one of: `full`, `hist`.')
        if iteration_count <= 0:
            raise ValueError('The `iteration_count` must be > 0.')
        if subsample < 0 or subsample > 1:
            raise ValueError('The `subsample` must be in [0..1].')
        if subfeature < 0 or subfeature > 1:
            raise ValueError('The `subfeature` must be in [0..1].')
        if max_depth < 0:
            raise ValueError('The `max_depth` must be >= 0.')
        if max_node_count < 0 and max_node_count != -1:
            raise ValueError('The `max_node_count` must be >= 0 or equal to -1.')
        if prune < 0:
            raise ValueError('The `prune` must be >= 0.')
        if thread_count <= 0:
            raise ValueError('The `thread_count` must be > 0.')
        if min_subtree_weight < 0:
            raise ValueError('The `min_subtree_weight` must be >= 0.')

        super().__init__(loss, int(iteration_count), float(learning_rate), float(subsample), float(subfeature), int(random_seed), int(max_depth),
            int(max_node_count), float(l1_reg), float(l2_reg), float(prune), int(thread_count), builder_type, int(max_bins), float(min_subtree_weight))

    def train(self, X, Y, weight=None):
        """Trains the gradient boosting model for regression.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training sample. The values will be converted 
            to ``dtype=np.float32``. If a sparse matrix is
            passed in, it will be converted to a sparse ``csr_matrix``.

        Y : array-like of shape (n_samples,)
            Correct function values (``float``) for the training set vectors.

        weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
        
        Return values
        -------
        model : object
            The trained ``GradientBoostRegressionModel``.
        """
        x = convert_data( X )
        y = numpy.array( Y, dtype=numpy.float32, copy=False )

        if x.shape[0] != y.size:
            raise ValueError('The `X` and `Y` inputs must be the same length.')

        if weight is None:
            weight = numpy.ones(y.size, numpy.float32)
        else:
            weight = numpy.array( weight, dtype=numpy.float32, copy=False )

        if numpy.any(weight < 0):
            raise ValueError('All `weight` elements must be >= 0.')

        return GradientBoostRegressionModel(super().train_regressor(*get_data(x), int(x.shape[1]), y, weight)) 
