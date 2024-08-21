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


class LinearClassificationModel :
    """Linear binary classification model.
    """
    def __init__(self, internal):
        self.internal = internal

    def classify(self, X):
        """Gets the classification results for the input sample.

        :param X: the input sample. Internally, it will be converted to
            ``dtype=np.float32``, and if a sparse matrix is provided -
            to a sparse ``csr_matrix``.
        :type X:  {array-like, sparse matrix} of shape (n_samples, n_features)

        :return: predictions of the input samples.
        :rtype: *generator of ndarray of shape (n_samples, n_classes)*
        """
        x = convert_data( X )
        return self.internal.classify(*get_data(x))

class LinearClassifier(PythonWrapper.Linear) :
    """Linear binary classifier.

    :param loss: the loss function to be optimized. `binomial` refers to
        deviance (= logistic regression) for classification
        with probabilistic outputs.
    :type loss: str, {'binomial', 'squared_hinge', 'smoothed_hinge'}, default='binomial'

    :param max_iteration_count: the maximum number of iterations.
    :type max_iteration_count: int, default=1000
    
    :param error_weight: the error weight relative to the regularization coefficient.
    :type error_weight: float, default=1.0
    
    :param sigmoid: the predefined sigmoid function coefficients.
    :type sigmoid: array of 2 float, default=(0.0, 0.0)
    
    :param tolerance: the stop criterion.
        -1 means calculate stop criterion automatically, 
        from the amount of vectors in each class in the training sample.
    :type tolerance: float, default=-1.0
        
    :param normalizeError: specifies if the error should be normalized.
    :type normalizeError: bool, default=False
        
    :param l1_reg: the L1 regularization coefficient.
        If 0, L2 regularization will be used instead.
    :type l1_reg: float, default=0.0
    
    :param thread_count: the number of threads to be used while training the model.
    :type thread_count: int, default=1

    :param multiclass_mode: determines how to handle multi-class classification
    :type multiclass_mode: str, ['one_vs_all', 'one_vs_one'], default='one_vs_all'
    """

    def __init__(self, loss='binomial', max_iteration_count=1000, error_weight=1.0,
        sigmoid=(0.0, 0.0), tolerance=-1.0, normalizeError=False, l1_reg=0.0, thread_count=1,
        multiclass_mode='one_vs_all'):

        if loss != 'binomial' and loss != 'squared_hinge' and loss != 'smoothed_hinge':
            raise ValueError('The `loss` must be one of: `binomial`, `squared_hinge`, `smoothed_hinge`.')
        if max_iteration_count <= 0:
            raise ValueError('The `max_iteration_count` must be > 0.')
        if error_weight <= 0:
            raise ValueError('The `error_weight` must be >= 0.')

        if thread_count <= 0:
            raise ValueError('The `thread_count` must be > 0.')
        if multiclass_mode != 'one_vs_all' and multiclass_mode != 'one_vs_one':
            raise ValueError('The `multiclass_mode` must be one of: `one_vs_all`, `one_vs_one`.')

        super().__init__(loss, int(max_iteration_count), float(error_weight), float(sigmoid[0]), float(sigmoid[1]), float(tolerance), bool(normalizeError),
            float(l1_reg), int(thread_count), multiclass_mode)

    def train(self, X, Y, weight=None):
        """Trains the linear classification model.

        :param X: the training sample. The values will be converted 
            to ``dtype=np.float32``. If a sparse matrix is
            passed in, it will be converted to a sparse ``csr_matrix``.
        :type X: {array-like, sparse matrix} of shape (n_samples, n_features)

        :param Y: correct class labels (``int``) for the training set vectors.
        :type Y: array-like of shape (n_samples,)

        :param weight: sample weights. If None, then samples are equally weighted.
        :type weight: array-like of shape (n_samples,), default=None

        :return: the trained classification model.
        :rtype: neoml.Linear.LinearClassificationModel
        """
        x = convert_data( X )
        y = np.asarray( Y, dtype=np.int32, order='C' )

        if x.shape[0] != y.size:
            raise ValueError('The `X` and `Y` inputs must be the same length.')

        if weight is None:
            weight = np.ones(y.size, np.float32, order='C')
        else:
            weight = np.asarray( weight, dtype=np.float32, order='C' )

        if np.any(y < 0):
            raise ValueError('All `Y` elements must be >= 0.')

        if np.any(weight < 0):
            raise ValueError('All `weight` elements must be >= 0.')

        return LinearClassificationModel(super().train_classifier(*get_data(x), int(x.shape[1]), y, weight))


class LinearRegressionModel :
    """Linear regression model.
    """
    def __init__(self, internal):
        self.internal = internal

    def predict(self, X):
        """Predicts the value of the function.

        :param X: the input vectors. The values will be converted 
            to ``dtype=np.float32``. If a sparse matrix is
            passed in, it will be converted to a sparse ``csr_matrix``.
        :type X: {array-like, sparse matrix} of shape (n_samples, n_features)

        :return: the predictions of the function value on each input vector.
        :rtype: *generator of ndarray of shape (n_samples)*
        """
        x = convert_data( X )
        return self.internal.predict(*get_data(x))

class LinearRegressor(PythonWrapper.Linear) :
    """Linear regressor.

    :param loss: the loss function to be optimized. 
        The quadratic loss L2 is the only one supported.
    :type loss: str, {'l2'}, default='l2'

    :param max_iteration_count: the maximum number of iterations.
    :type max_iteration_count: int, default=1000

    :param error_weight: the error weight relative to the regularization coefficient.
    :type error_weight: float, default=1.0

    :param sigmoid: the predefined sigmoid function coefficients.
    :type sigmoid: array of 2 float, default=(0.0, 0.0)

    :param tolerance: the stop criterion.
        -1 means calculate stop criterion automatically, 
        from the amount of vectors in each class in the training sample.
    :type tolerance: float, default=-1.0

    :param normalizeError: specifies if the error should be normalized.
    :type normalizeError: bool, default=False

    :param l1_reg: the L1 regularization coefficient.
        If 0, L2 regularization will be used instead.
    :type l1_reg: float, default=0.0

    :param thread_count: the number of threads to be used while training the model.
    :type thread_count: int, default=1
    """

    def __init__(self, loss='l2', max_iteration_count=1000, error_weight=1.0,
        sigmoid=(0.0, 0.0), tolerance=-1.0, normalizeError=False, l1_reg=0.0, thread_count=1):

        if loss != 'l2':
            raise ValueError('The `loss` must be `l2` for regression.')
        if max_iteration_count <= 0:
            raise ValueError('The `max_iteration_count` must be > 0.')
        if error_weight <= 0:
            raise ValueError('The `error_weight` must be >= 0.')

        if thread_count <= 0:
            raise ValueError('The `thread_count` must be > 0.')

        super().__init__(loss, int(max_iteration_count), float(error_weight), float(sigmoid[0]), float(sigmoid[1]), float(tolerance), bool(normalizeError),
            float(l1_reg), int(thread_count), '')

    def train(self, X, Y, weight=None):
        """Trains the linear regression model.

        :param X: the training sample. The values will be converted 
            to ``dtype=np.float32``. If a sparse matrix is
            passed in, it will be converted to a sparse ``csr_matrix``.
        :type X: {array-like, sparse matrix} of shape (n_samples, n_features)

        :param Y: correct function values (``float``) for the training set vectors.
        :type Y: array-like of shape (n_samples,)

        :param weight: sample weights. If None, then samples are equally weighted.
        :type weight: array-like of shape (n_samples,), default=None

        :return: the trained regression model.
        :rtype: neoml.Linear.LinearRegressionModel
        """
        x = convert_data( X )
        y = np.asarray( Y, dtype=np.float32, order='C' )

        if x.shape[0] != y.size:
            raise ValueError('The `X` and `Y` inputs must be the same length.')

        if weight is None:
            weight = np.ones(y.size, np.float32, order='C')
        else:
            weight = np.asarray( weight, dtype=np.float32, order='C' )

        if np.any(weight < 0):
            raise ValueError('All `weight` elements must be >= 0.')

        return LinearRegressionModel(super().train_regressor(*get_data(x), int(x.shape[1]), y, weight))
