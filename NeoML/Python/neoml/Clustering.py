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


class FirstCome(PythonWrapper.FirstCome) :
    """First come clustering creates a new cluster for each new vector
    that is far enough from the clusters already existing.

    :param min_vector_count: the smallest number of vectors in a cluster to consider that the variance is valid.
    :type min_vector_count: int, > 0, default=4

    :param default_variance: the default variance (for when the number of vectors is smaller than min_vector_count).
    :type default_variance: float, default=1.0

    :param threshold: the distance threshold for creating a new cluster.
    :type threshold: float, default=0.0

    :param min_cluster_size_ratio: the minimum ratio of the number elements in a cluster to the total number of vectors.
    :type min_cluster_size_ratio: float, default=0.05

    :param max_cluster_count: the maximum number of clusters to prevent algorithm divergence
        in case of great differences in data.
    :type max_cluster_count: int, default=100

    :param distance: the distance function to measure cluster size.
    :type distance: str, {'euclid', 'machalanobis', 'cosine'}, default='euclid'
    """

    def __init__(self, min_vector_count=4, default_variance=1.0, threshold=0.0, min_cluster_size_ratio=0.05, max_cluster_count=100, distance='euclid'):

        if distance != 'euclid' and distance != 'machalanobis' and distance != 'cosine':
            raise ValueError('The `distance` must be one of {`euclid`, `machalanobis`, `cosine`}.')
        if max_cluster_count <= 0:
            raise ValueError('The `max_cluster_count` must be > 0.')
        if min_cluster_size_ratio > 1 or min_cluster_size_ratio < 0:
            raise ValueError('The `min_cluster_size_ratio` must be in [0, 1].')

        super().__init__(distance, int(min_vector_count), float(default_variance), float(threshold), float(min_cluster_size_ratio), int(max_cluster_count))

    def clusterize(self, X, weight=None):
        """Performs clustering of the given data.

        :param X: the input sample. Internally, it will be converted
            to ``dtype=np.float32``, and if a sparse matrix is provided -
            to a sparse ``scipy.csr_matrix``.
        :type X: {array-like, sparse matrix} of shape (n_samples, n_features)

        :param weight: sample weights. If `None`, then samples are equally weighted.
        :type weight: array-like of shape (n_samples,) or None, default=None

        :return:
            - **clusters** - cluster indices for each object of `X`;
            - **centers** - cluster centers;
            - **vars** - cluster variances.
        :rtype:
            - tuple(clusters, centers, vars)
            - **clusters** - *numpy.ndarray(numpy.int32) of shape (n_samples,)*
            - **centers** - *numpy.ndarray(numpy.float32) of shape (init_cluster_count, n_features)*
            - **vars** - *numpy.ndarray(numpy.float32) of shape (init_cluster_count, n_features)*
        """
        x = convert_data( X )

        if weight is None:
            weight = numpy.ones(x.shape[0], numpy.float32, order='C')
        else:
            weight = numpy.array(weight, dtype=numpy.float32, copy=False, order='C')
            if numpy.any(weight < 0):
                raise ValueError('All `weight` elements must be >= 0.')

        return super().clusterize(*get_data(x), int(x.shape[1]), weight)

#-------------------------------------------------------------------------------------------------------------


class Hierarchical(PythonWrapper.Hierarchical) :
    """Hierarchical clustering. 
    First, it creates a cluster per element, the merges closest clusters on each step 
    until the final cluster is achieved.

    :param max_cluster_distance: the maximum distance between two clusters that still may be merged.
    :type max_cluster_distance: float

    :param min_cluster_count: the minimum number of clusters in the result.
    :type min_cluster_count: int

    :param distance: the distance function.
    :type distance: str, {'euclid', 'machalanobis', 'cosine'}, default='euclid'

    :param linkage: the approach used for distance calculation between clusters
    :type linkage: str, {'centroid', 'single', 'average', 'complete', 'ward'}, default='centroid'
    """

    def __init__(self, max_cluster_distance, min_cluster_count, distance='euclid', linkage='centroid'):

        if distance != 'euclid' and distance != 'machalanobis' and distance != 'cosine':
            raise ValueError('The `distance` must be one of {`euclid`, `machalanobis`, `cosine`}.')
        if min_cluster_count <= 0:
            raise ValueError('The `min_cluster_count` must be > 0.')
        if linkage not in {'centroid', 'single', 'average', 'complete', 'ward'}:
            raise ValueError('The `linkage` must be one of {`centroid`, `single`, `average`, `complete`, `ward`}')
        if linkage == 'ward' and distance != 'euclid':
            raise ValueError('`ward` linkage works only with `euclid` distance')

        super().__init__(distance, float(max_cluster_distance), int(min_cluster_count), linkage)

    def clusterize(self, X, weight=None):
        """Performs clustering of the given data.

        :param X: the input sample. Internally, it will be converted
            to ``dtype=np.float32``, and if a sparse matrix is provided -
            to a sparse ``scipy.csr_matrix``.
        :type X: {array-like, sparse matrix} of shape (n_samples, n_features)

        :param weight: sample weights. If `None`, then samples are equally weighted.
        :type weight: array-like of shape (n_samples,) or None, default=None

        :return:
            - **clusters** - cluster indices for each object of `X`;
            - **centers** - cluster centers;
            - **vars** - cluster variances.
        :rtype:
            - tuple(clusters, centers, vars)
            - **clusters** - *numpy.ndarray(numpy.int32) of shape (n_samples,)*
            - **centers** - *numpy.ndarray(numpy.float32) of shape (init_cluster_count, n_features)*
            - **vars** - *numpy.ndarray(numpy.float32) of shape (init_cluster_count, n_features)*
        """
        x = convert_data( X )

        if weight is None:
            weight = numpy.ones(x.shape[0], numpy.float32, order='C')
        else:
            weight = numpy.array(weight, dtype=numpy.float32, copy=False, order='C')
            if numpy.any(weight < 0):
                raise ValueError('All `weight` elements must be >= 0.')

        return super().clusterize(*get_data(x), int(x.shape[1]), weight)

#-------------------------------------------------------------------------------------------------------------


class IsoData(PythonWrapper.IsoData) :
    """IsoData clustering.
    A heuristic algorithm based on geometrical proximity of the data points.

    :param init_cluster_count: the number of initial clusters.
        The initial cluster centers are randomly selected from the input data.
    :type init_cluster_count: int

    :param max_cluster_count: the maximum number of clusters.
    :type max_cluster_count: int

    :param min_cluster_size: the minimum cluster size.
    :type min_cluster_size: int

    :param max_iteration_count: the maximum number of algorithm iterations. 
    :type max_iteration_count: int

    :param min_cluster_distance: the minimum distance between the clusters.
        Whenever two clusters are closer they are merged.
    :type min_cluster_distance: float

    :param max_cluster_diameter: the maximum cluster diameter.
        Whenever a cluster is larger it may be split.
    :type max_cluster_diameter: float

    :param mean_diameter_coef: indicates how much the cluster diameter may exceed 
        the mean diameter across all the clusters. If a cluster diameter is larger
        than the mean diameter multiplied by this value it may be split.
    :type mean_diameter_coef: float
    """

    def __init__(self, init_cluster_count, max_cluster_count, min_cluster_size, max_iteration_count, min_cluster_distance, max_cluster_diameter, mean_diameter_coef ):

        if max_iteration_count <= 0:
            raise ValueError('The `max_iteration_count` must be > 0.')
        if init_cluster_count <= 0:
            raise ValueError('The `init_cluster_count` must be > 0.')
        if min_cluster_size <= 0:
            raise ValueError('The `min_cluster_size` must be > 0.')

        super().__init__( int(init_cluster_count), int(max_cluster_count), int(min_cluster_size), int(max_iteration_count), float(min_cluster_distance), float(max_cluster_diameter), float(mean_diameter_coef) )

    def clusterize(self, X, weight=None):
        """Performs clustering of the given data.

        :param X: the input sample. Internally, it will be converted
            to ``dtype=np.float32``, and if a sparse matrix is provided -
            to a sparse ``scipy.csr_matrix``.
        :type X: {array-like, sparse matrix} of shape (n_samples, n_features)

        :param weight: sample weights. If `None`, then samples are equally weighted.
        :type weight: array-like of shape (n_samples,) or None, default=None

        :return:
            - **clusters** - cluster indices for each object of `X`;
            - **centers** - cluster centers;
            - **vars** - cluster variances.
        :rtype:
            - tuple(clusters, centers, vars)
            - **clusters** - *numpy.ndarray(numpy.int32) of shape (n_samples,)*
            - **centers** - *numpy.ndarray(numpy.float32) of shape (init_cluster_count, n_features)*
            - **vars** - *numpy.ndarray(numpy.float32) of shape (init_cluster_count, n_features)*
        """
        x = convert_data( X )

        if weight is None:
            weight = numpy.ones(x.shape[0], numpy.float32, order='C')
        else:
            weight = numpy.array(weight, dtype=numpy.float32, copy=False, order='C')
            if numpy.any(weight < 0):
                raise ValueError('All `weight` elements must be >= 0.')

        return super().clusterize(*get_data(x), int(x.shape[1]), weight)

#-------------------------------------------------------------------------------------------------------------


class KMeans(PythonWrapper.KMeans) :
    """K-means clustering.

    :param max_iteration_count: the maximum number of algorithm iterations.
    :type max_iteration_count: int

    :param cluster_count: the number of clusters.
    :type cluster_count: int

    :param algo: the algorithm used during clustering.
    :type algo: str, {'elkan', 'lloyd'}, default='lloyd'

    :param init: the algorithm used for selecting initial centers.
    :type init: str, {'k++', 'default'}, default='default'

    :param distance: the distance function.
    :type distance: str, {'euclid', 'machalanobis', 'cosine'}, default='euclid'

    :param thread_count: number of threads
    :type thread_count: int, > 0, default=1

    :param run_count: number of runs, the result is the best of the runs (based on inertia)
    :type run_count: int, > 0, default=1

    :param seed: the initial seed for random
    :type seed: int, default=3306
    """

    def __init__(self, max_iteration_count, cluster_count, algo='lloyd', init='default', distance='euclid',
                 thread_count=1, run_count=1, seed=3306):
        if algo != 'elkan' and algo != 'lloyd':
            raise ValueError('The `algo` must be one of {`elkan`, `lloyd`}.')
        if init != 'k++' and init != 'default':
            raise ValueError('The `init` must be one of {`k++`, `default`}.')
        if distance != 'euclid' and distance != 'machalanobis' and distance != 'cosine':
            raise ValueError('The `distance` must be one of {`euclid`, `machalanobis`, `cosine`}.')
        if max_iteration_count <= 0:
            raise ValueError('The `max_iteration_count` must be > 0.')
        if cluster_count <= 0:
            raise ValueError('The `cluster_count` must be > 0.')
        if thread_count <= 0:
            raise ValueError('The `thread_count` must be > 0')
        if run_count <= 0:
            raise ValueError('The `run_count` must be > 0')
        if not isinstance(seed, int):
            raise ValueError('The `seed` must be integer')
        super().__init__(algo, init, distance, int(max_iteration_count), int(cluster_count), int(thread_count),
            int(run_count), int(seed))

    def clusterize(self, X, weight=None):
        """Performs clustering of the given data.

        :param X: the input sample. Internally, it will be converted
            to ``dtype=np.float32``, and if a sparse matrix is provided -
            to a sparse ``scipy.csr_matrix``.
        :type X: array-like or sparse matrix of shape (n_samples, n_features)

        :param weight: sample weights. If `None`, then samples are equally weighted.
            `None` by default.
        :type weight: array-like of shape (n_samples,)

        :return:
            - **clusters** - array of integers with cluster indices for each object of `X`;
            - **centers** - cluster centers;
            - **vars** - cluster variances.
        :rtype:
            - tuple(clusters, centers, vars)
            - **clusters** - *numpy.ndarray(numpy.int32) of shape (n_samples,)*
            - **centers** - *numpy.ndarray(numpy.float32) of shape (cluster_count, n_features)*
            - **vars** - *numpy.ndarray(numpy.float32) of shape (cluster_count, n_features)*

        """
        x = convert_data(X)

        if weight is None:
            weight = numpy.ones(x.shape[0], numpy.float32, order='C')
        else:
            weight = numpy.array(weight, dtype=numpy.float32, copy=False, order='C')
            if numpy.any(weight < 0):
                raise ValueError('All `weight` elements must be >= 0.')

        return super().clusterize(*get_data(x), int(x.shape[1]), weight)
