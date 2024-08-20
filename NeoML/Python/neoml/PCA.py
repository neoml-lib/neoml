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

from .Utils import convert_data, get_data
import neoml.PythonWrapper as PythonWrapper


"""
Singular Value Decomposition of a given matrix into matrices u, s, v.

:param compute_u: indicates whether matrix u should be returned
:type compute_u: bool, default=True
:param compute_v: indicates whether matrix v should be returned
:type compute_v: bool, default=False
:param algorithm: chooses an algorithm.
    'full' implements LAPACK SVD (LAPACKE_sgesvd).
    'randomized' - see `Halko, N., Martinsson, P. G., and Tropp, J. A. (2011).
    "Finding structure with randomness: Probabilistic algorithms for
    constructing approximate matrix decompositions".
    SIAM review, 53(2), 217-288.
    <10.1137/090771806>.`
:type algorithm: str, ['full', 'randomized'], default='full'
:param components: indicates a number of largest singular values to search.
    Default value is min(matrix.height, matrix.width).
:type components: int
"""
def svd(matrix, compute_u = True, compute_v = False, algorithm = 'full', components = None):
    if algorithm not in ('full', 'randomized'):
        raise ValueError("`algorithm` must be one of ('full', 'randomized').")
    x = convert_data(matrix)
    if len(x.shape) != 2:
        raise ValueError("Matrix must be square.")
    if components is None:
        components = min(*x.shape)
    if components <= 0 or components > min(*x.shape):
        raise ValueError("`components` must be in range [1, min(matrix.height, matrix.width)].")
    return PythonWrapper.singular_value_decomposition(*x.shape, *get_data(x),
        compute_u, compute_v, algorithm == 'full', components)


class PCA(PythonWrapper.PCA) :
    """Principal components analysis (PCA) algorithm. 
    It uses singular value decomposition to project the data into
    a lower dimensional space.

    :param n_components: number of components or the way it should be chosen.
        If it's an integer value, it's simply the number of components.
        If it's a float value and 0 < n_components < 1, set the number of components so that variance is greater than this value.
        If n_components = None, set the number of components as min(data.width, data.height).
    :type n_components: int, float, default=None
    :param svd_solver: full or randomized svd algorithm
    :type svd_solver: str, ['full', 'randomized'], default='full'
    """

    def __init__(self, n_components=None, svd_solver='full'):

        if n_components is None:
            components = ('None', 0)
        elif 0 < n_components < 1:
            components = ('Float', n_components)
        else:
            if n_components <= 0:
                raise ValueError('`n_components` > 0.')
            components = ('Int', n_components,)

        if svd_solver not in ('full', 'randomized'):
            raise ValueError("`svd_solver` must be 'full' or 'randomized'.")

        super().__init__(*components, svd_solver == 'full')

    def fit(self, X):
        """Performs linear dimensionality reduction of the given data:
        finds the singular vectors and selects the required number of them
        to be principal components, preferring the vectors that correspond
        to the largest singular values.

        :param X: the input sample. Internally, it will be converted
            to ``dtype=np.float32``, and if a sparse matrix is provided -
            to a sparse ``scipy.csr_matrix``.
        :type X: {array-like, sparse matrix} of shape (n_samples, n_features)
        """
        x = convert_data( X )

        super().fit(*x.shape, *get_data(x))

    def fit_transform(self, X):
        """Performs linear dimensionality reduction of the given data:
        finds the singular vectors and selects the required number of them
        to be principal components, preferring the vectors that correspond
        to the largest singular values.
        
        Then projects the dataset onto the principal components axes and returns the result.

        :param X: the input sample. Internally, it will be converted
            to ``dtype=np.float32``, and if a sparse matrix is provided -
            to a sparse ``scipy.csr_matrix``.
        :type X: {array-like, sparse matrix} of shape (n_samples, n_features)

        :return: projection of the data into a lower dimensional space.
        :rtype: *generator of ndarray of shape (n_samples, n_components)*
        """
        x = convert_data( X )

        return super().fit_transform(*x.shape, *get_data(x))

    def transform(self, X):
        """Projects the dataset onto the principal components axes and returns the result.

        :param X: the input sample. Internally, it will be converted
            to ``dtype=np.float32``, and if a sparse matrix is provided -
            to a sparse ``scipy.csr_matrix``.
        :type X: {array-like, sparse matrix} of shape (n_samples, n_features)

        :return: projection of the data into a lower dimensional space.
        :rtype: *generator of ndarray of shape (n_samples, n_components)*
        """
        x = convert_data( X )

        return super().transform(*x.shape, *get_data(x))

    def store(self, path):
        """Serializes the model.
        """
        return super().store(str(path))

    def load(self, path):
        """Loads the model from file.
        """
        return super().load(str(path))

    @property
    def singular_values(self):
        """Returns the singular values corresponding to the selected principal axes.
        """
        return super().singular_values()

    @property
    def explained_variance(self):
        """Returns the variance explained by each of the selected principal axes.
        """
        return super().explained_variance()

    @property
    def explained_variance_ratio(self):
        """Returns the percentage of variance explained by each of the selected principal axes.
        """
        return super().explained_variance_ratio()

    @property
    def components(self):
        """Returns ndarray ( components x features ) with rows corresponding to the selected principal axes.
        """
        return super().components()

    @property
    def n_components(self):
        """Returns the selected number of principal axes.
        """
        return super().n_components()

    @property
    def noise_variance(self):
        """Returns the mean of singular values not corresponding to the selected principal axes.
        """
        return super().noise_variance()

