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


def cross_validation_score(classifier, X, Y, weight=None, score="accuracy", parts=5, stratified=False):
    """Gets the classification results for the input sample.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input vectors, put into a matrix. The values will be 
        converted to ``dtype=np.float32``. If a sparse matrix is
        passed in, it will be converted to a sparse ``csr_matrix``.

    Y : array-like of shape (n_samples,)
        Correct function values (``float``) for the training set vectors.

    weight : array-like of shape (n_samples,), default=None
        Sample weights. If None, then samples are equally weighted.

    score : {'accuracy', 'f1'}, default='accuracy'
  
    parts :

    stratified: "It is guaranteed that the ratio of classes in each part is (almost) the same as in the input data"


    Return values
    -------
    predictions : generator of ndarray of shape (n_samples, n_classes)
        The predictions of class probability for each input vector.
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

    if score != "accuracy" and score != "f1":
        raise ValueError('The `score` must be one of: `accuracy`, `f1`.')

    if parts <= 0 or parts >= y.size / 2:
        raise ValueError('`parts` must be in (0, vectorCount).')

    return PythonWrapper._cross_validation_score(classifier, x.indices, x.data, x.indptr, int(x.shape[1]), y, weight, score, parts, bool(stratified))
