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

class SvmClassificationModel :
    """Support-vector machine (SVM) classification model.
    """
    def __init__(self, internal):
        self.internal = internal

    def classify(self, X):
        """Gets the classification results for the input sample.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Return values
        -------
        predictions : generator of ndarray of shape (n_samples, n_classes)
            The predictions of the input samples.
        """
        x = convert_data( X )
        return self.internal.classify(*get_data(x))

class SvmClassifier(PythonWrapper.Svm) :
    """Support-vector machine algorithm translates the input data
    into vectors in a high-dimensional space and searches 
    for a maximum-margin dividing hyperplane.

    
    Parameters
    ----------
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, default='linear'
        The kernel function to be used.

    max_iteration_count : int, default=1000
        The maximum number of iterations.

    error_weight : float, default=1.0
        The error weight relative to the regularization function.

    degree : int, default=1
        The degree for the gaussian kernel.

    gamma : float, default=1.0
        The kernel coefficient (for 'poly', 'rbf', 'sigmoid').

    coeff0 : float, default=1.0
        The kernel free term (for 'poly, 'sigmoid').
    
    tolerance : float, default=0.1
        The algorithm precision.
    
    thread_count : int, default=1
        The number of processing threads to be used while training the model.
    """

    def __init__(self, kernel='linear', max_iteration_count=1000, error_weight=1.0,
        degree=1, gamma=1.0, coeff0=1.0, tolerance=0.1, thread_count=1):

        if kernel != 'linear' and kernel != 'poly' and kernel != 'rbf' and kernel != 'sigmoid':
            raise ValueError('The `kernel` must be one of: `linear`, `poly`, `rbf`, `sigmoid`.')
        if max_iteration_count <= 0:
            raise ValueError('The `max_iteration_count` must be > 0.')
        if error_weight <= 0:
            raise ValueError('The `error_weight` must be >= 0.')

        if thread_count <= 0:
            raise ValueError('The `thread_count` must be > 0.')

        super().__init__(kernel, float(error_weight), int(max_iteration_count), int(degree),
            float(gamma), float(coeff0), float(tolerance), int(thread_count))

    def train(self, X, Y, weight=None):
        """Trains the SVM classification model.

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
            The trained ``SvmClassificationModel``.
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

        return SvmClassificationModel(super().train_classifier(*get_data(x), int(x.shape[1]), y, weight))
