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

import numpy
from .Utils import convert_data, get_data
import neoml.PythonWrapper as PythonWrapper


def cross_validation_score(classifier, X, Y, weight=None, score="accuracy", parts=5, stratified=False):
    """Performs cross-validation of the given classifier on a set of data.
    The input sample is divided into the specified number of parts, then each of them in turn serves as the testing set while all the others are taken for the training set.
    Can calculate either accuracy or F-measure.

    :param classifier: the classifier to be tested.
    :type classifier: object

    :param X: the input vectors, put into a matrix. The values will be 
        converted to ``dtype=np.float32``. If a sparse matrix is
        passed in, it will be converted to a sparse ``csr_matrix``.
    :type X: {array-like, sparse matrix} of shape (n_samples, n_features)
        
    :param Y: correct function values (``float``) for the training set vectors.
    :type Y: array-like of shape (n_samples,)

    :param weight: sample weights. If None, then all vectors are equally weighted.
    :type weight: array-like of shape (n_samples,), default=None

    :param score: the metric that should be calculated.
    :type score: str, {'accuracy', 'f1'}, default='accuracy'
  
    :param parts: the number of parts into which the input sample should be divided.
    :type parts: int, default=5

    :param stratified: specifies if the input set should be divided so that 
        the ratio of classes in each part is (almost) the same as in the input data.
    :type stratified: bool, default=False

    :return: the calculated metrics.
    :rtype: *array-like of shape (parts,)*
    """

    x = convert_data( X )
    y = numpy.array( Y, dtype=numpy.int32, copy=False, order='C' )

    if x.shape[0] != y.size:
        raise ValueError('The `X` and `Y` inputs must be the same length.')

    if weight is None:
        weight = numpy.ones(y.size, numpy.float32, order='C')
    else:
        weight = numpy.array( weight, dtype=numpy.float32, copy=False, order='C' )

    if numpy.any(y < 0):
        raise ValueError('All `Y` elements must be >= 0.')

    if numpy.any(weight < 0):
        raise ValueError('All `weight` elements must be >= 0.')

    if score != "accuracy" and score != "f1":
        raise ValueError('The `score` must be one of: `accuracy`, `f1`.')

    if parts <= 0 or parts >= y.size / 2:
        raise ValueError('`parts` must be in (0, vectorCount).')

    return PythonWrapper._cross_validation_score(classifier, *get_data(x), int(x.shape[1]), y, weight, score, parts, bool(stratified))
