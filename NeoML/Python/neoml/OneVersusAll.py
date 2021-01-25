""" Copyright (c) 2017-2020 ABBYY Production LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--------------------------------------------------------------------------------------------------------------*/
"""

import neoml.PythonWrapper as PythonWrapper
import numpy
from scipy.sparse import csr_matrix

class OneVersusAllClassificationModel:
    """One versus all classification model.
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
        x = csr_matrix( X, dtype=numpy.float32 )
        return self.internal.classify(x.indices, x.data, x.indptr)

class OneVersusAllClassifier(PythonWrapper.OneVersusAll):
    """One versus all classification method provides a way to solve
    a multi-class classification problem using only a binary classifier.
    
    The original classification problem is represented 
    as a series of binary classification problems, one for each class, 
    that determine the probability for the object to belong to this class.

    Parameters
    ----------
    base_classifier : object
        The binary classifier to be used in each problem.
        Can be an instance of ``GradientBoostClassifier``,
        ``LinearClassifier``, ``SvmClassifier``, or ``DecisionTreeClassifier``
        set up for two classes.
    """
    def __init__(self, base_classifier):
        super().__init__(base_classifier)

    def train(self, X, Y, weight=None):
        """Trains the one-versus-all classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Y : array-like of shape (n_samples,)
            Correct class labels (``int``) for the training set vectors.

        weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Return values
        -------
        model : object
            The trained ``OneVersusAllClassificationModel``.
        """
        x = csr_matrix( X, dtype=numpy.float32 )
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

        return OneVersusAllClassificationModel(super().train_classifier(x.indices, x.data, x.indptr, int(x.shape[1]), y, weight)) 
