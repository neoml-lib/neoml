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

class IsoData(PythonWrapper.IsoData) :
    """IsoData clustering.
    Parameters
    ----------

    init_cluster_count : the number of initial clusters.
        The initial cluster centers are randomly selected from the input data.

    max_iteration_count : the maximum number of clusters.

    min_cluster_size : the minimum cluster size.

    max_iteration_count : the maximum number of algorithm iterations. 

    min_cluster_distance : the minimum distance between the clusters.
        Whenever two clusters are closer they are merged.

    max_cluster_diameter : the maximum cluster diameter.
        Whenever a cluster is larger it may be split.

    mean_diameter_coef : indicates how much the cluster diameter may exceed.
        The mean diameter across all the clusters. If a cluster diameter is larger than the mean diameter multiplied by this value it may be split.

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
        """.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
        """
        x = csr_matrix( X, dtype=numpy.float32 )

        if weight is None:
            weight = numpy.ones(x.size, numpy.float32)
        else:
            weight = numpy.array(weight, dtype=numpy.float32, copy=False)
            if numpy.any(weight < 0):
                raise ValueError('All `weight` elements must be >= 0.')

        return super().clusterize(x.indices, x.data, x.indptr, int(x.shape[1]), weight)
