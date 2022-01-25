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
from scipy.sparse import csr_matrix
import neoml.PythonWrapper as PythonWrapper

"""
Singular Value Decomposition of a given matrix into matrices u, s, v.

:param compute_u: indicates whether matrix u will be returned
:type compute_u: bool, default=True
:param compute_v: indicates whether matrix v will be returned
:type compute_v: bool, default=False
:param algorithm: chooses an algorithm.
    'full' implements LAPACK SVD (LAPACKE_sgesvd).
    'sparse' implements the FEAST algorithm (mkl_sparse_s_svd).
:type algorithm: str, ['full', 'sparse'], default='full'
:param components: indicates a number of largest singular values
    to search (only for 'sparse' algorithm).
    Default value is min(matrix.height, matrix.width).
:type components: int
"""
def svd(matrix, compute_u = True, compute_v = False, algorithm = 'full', components = None):
    if algorithm not in ('full', 'sparse'):
        raise ValueError("`algorithm` must be one of ('full', 'sparse').")
    x = convert_data(matrix)
    if len(x.shape) != 2:
        raise ValueError("Matrix must be square.")
    if algorithm == 'sparse':
        if compute_u == compute_v:
            raise ValueError("Exactly one of u and v must be calculated.")
    if components is None:
        components = min(*x.shape)
    return PythonWrapper.singular_value_decomposition(*x.shape, *get_data(x),
        compute_u, compute_v, algorithm == 'full', components)

class PCA(PythonWrapper.PCA) :
    """PCA algorithm implementing linear dimensionality reduction
    using Singular Value Decomposition to project the data into
    a lower dimensional space.

    :param n_components: number of components.
        If 0 < n_components < 1, set the number of components such that variance is greater than n_components.
        If n_components = None, set the number of components as min(data.width, data.height).
    :type n_components: int, float, default=None
    """

    def __init__(self, n_components):

        if n_components is None:
            components = ('None', 0)
        elif 0 < n_components < 1:
            components = ('Float', n_components)
        else:
            if n_components <= 0:
                raise ValueError('`n_components` > 0.')
            components = ('Int', n_components)

        super().__init__(*components)

    def fit(self, X):
        """Performs linear dimensionality reduction of the given data.

        :param X: the input sample. Internally, it will be converted
            to ``dtype=np.float32``, and if a sparse matrix is provided -
            to a sparse ``scipy.csr_matrix``.
        :type X: {array-like, sparse matrix} of shape (n_samples, n_features)
        """
        x = convert_data( X )

        super().fit(*x.shape, *get_data(x))

    def fit_transform(self, X):
        """Performs linear dimensionality reduction of the given data.

        :param X: the input sample. Internally, it will be converted
            to ``dtype=np.float32``, and if a sparse matrix is provided -
            to a sparse ``scipy.csr_matrix``.
        :type X: {array-like, sparse matrix} of shape (n_samples, n_features)

        :return: projection of the data into a lower dimensional space.
        :rtype: *generator of ndarray of shape (n_samples, n_components)*
        """
        x = convert_data( X )

        return super().fit_transform(*x.shape, *get_data(x))

    @property
    def singular_values(self):
        """Returns singular values corresponding to the selected principal axis
        """
        return super().singular_values()

    @property
    def explained_variance(self):
        """Returns variance explained by each of the selected principal axis
        """
        return super().explained_variance()

    @property
    def explained_variance_ratio(self):
        """Returns percentage of variance explained by each of the selected principal axis
        """
        return super().explained_variance_ratio()

    @property
    def components(self):
        """Returns ndarray ( components x features ) with rows corresponding to the selected principal axis 
        """
        return super().components()

    @property
    def n_components(self):
        """Returns selected number of principal axis
        """
        return super().n_components()

    @property
    def noise_variance(self):
        """Returns mean of singular values not corresponding to the selected principal axis
        """
        return super().noise_variance()
