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
from scipy.sparse import csr_matrix, issparse
import neoml.PythonWrapper as PythonWrapper

class LinearClassificationModel :
    """Linear binary classification model.
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
        x = convert_data( X )
        return self.internal.classify(*get_data(x))

class LinearClassifier(PythonWrapper.Linear) :
    """Linear binary classifier.

    Parameters
    ----------
    loss : {'binomial', 'squared_hinge', 'smoothed_hinge'}, default='binomial'
        The loss function to be optimized. 'binomial' refers to
        deviance (= logistic regression) for classification
        with probabilistic outputs.

    max_iteration_count : int, default=1000
        The maximum number of iterations.
    
    error_weight : float, default=1.0
        The error weight relative to the regularization coefficient.
    
    sigmoid : array of 2 float, default=(0.0, 0.0)
        The predefined sigmoid function coefficients.
    
    tolerance : float, default=-1.0
        The stop criterion.
        -1 means calculate stop criterion automatically, 
        from the amount of vectors in each class in the training sample.
        
    normalizeError : bool, default=False
        Specifies if the error should be normalized.
        
    l1_reg : float, default=0.0
        The L1 regularization coefficient.
        If 0, L2 regularization will be used instead.
    
    thread_count: int, default=1
        The number of threads to be used while training the model.
    """

    def __init__(self, loss='binomial', max_iteration_count=1000, error_weight=1.0,
        sigmoid=(0.0, 0.0), tolerance=-1.0, normalizeError=False, l1_reg=0.0, thread_count=1):

        if loss != 'binomial' and loss != 'squared_hinge' and loss != 'smoothed_hinge':
            raise ValueError('The `loss` must be one of: `binomial`, `squared_hinge`, `smoothed_hinge`.')
        if max_iteration_count <= 0:
            raise ValueError('The `max_iteration_count` must be > 0.')
        if error_weight <= 0:
            raise ValueError('The `error_weight` must be >= 0.')

        if thread_count <= 0:
            raise ValueError('The `thread_count` must be > 0.')

        super().__init__(loss, int(max_iteration_count), float(error_weight), float(sigmoid[0]), float(sigmoid[1]), float(tolerance), bool(normalizeError),
            float(l1_reg), int(thread_count))

    def train(self, X, Y, weight=None):
        """Trains the linear classification model.
        
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
            The trained ``LinearClassificationModel``.
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

        return LinearClassificationModel(super().train_classifier(*get_data(x), int(x.shape[1]), y, weight))


class LinearRegressionModel :
    """Linear regression model.
    """
    def __init__(self, internal):
        self.internal = internal

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
        x = convert_data( X )
        return self.internal.predict(*get_data(x))

class LinearRegressor(PythonWrapper.Linear) :
    """Linear regressor.

    Parameters
    ----------
    loss : {'l2'}, default='l2'
        The loss function to be optimized. 
        The quadratic loss L2 is the only one supported.

    max_iteration_count : int, default=1000
        The maximum number of iterations.
    
    error_weight : float, default=1.0
        The error weight relative to the regularization coefficient.

    sigmoid : array of 2 float, default=(0.0, 0.0)
        The predefined sigmoid function coefficients.
    
    tolerance : float, default=-1.0
        The stop criterion.
        -1 means calculate stop criterion automatically, 
        from the amount of vectors in each class in the training sample.
        
    normalizeError : bool, default=False
        Specifies if the error should be normalized.
        
    l1_reg : float, default=0.0
        The L1 regularization coefficient.
        If 0, L2 regularization will be used instead.
    
    thread_count: int, default=1
        The number of threads to be used while training the model.
    """

    def __init__(self, loss='l2', max_iteration_count=1000, error_weight=1.0,
        sigmoid=(0.0, 0.0), tolerance=-1.0, normalizeError=False, l1_reg=0.0, thread_count=1):

        if loss != 'l2':
            raise ValueError('The `loss` must be `l2 for regression.')
        if max_iteration_count <= 0:
            raise ValueError('The `max_iteration_count` must be > 0.')
        if error_weight <= 0:
            raise ValueError('The `error_weight` must be >= 0.')

        if thread_count <= 0:
            raise ValueError('The `thread_count` must be > 0.')

        super().__init__(loss, int(max_iteration_count), float(error_weight), float(sigmoid[0]), float(sigmoid[1]), float(tolerance), bool(normalizeError),
            float(l1_reg), int(thread_count))

    def train(self, X, Y, weight=None):
        """Trains the linear regression model.

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
            The trained ``LinearRegressionModel``.
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

        return LinearRegressionModel(super().train_regressor(*get_data(x), int(x.shape[1]), y, weight))
