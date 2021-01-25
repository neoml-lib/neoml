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

class Hierarchical(PythonWrapper.Hierarchical) :
    """Hierarchical clustering.
    Parameters
    ----------
    distance : {'euclid', 'machalanobis', 'cosine'}, default='euclid' the distance function.

    max_cluster_distance : the maximum distance between two clusters that still may be merged.

    min_cluster_count : the minimum number of clusters in the result.
    """

    def __init__(self, max_cluster_distance, min_cluster_count, distance='euclid'):

        if distance != 'euclid' and distance != 'machalanobis' and distance != 'cosine':
            raise ValueError('The `distance` must be one of {`euclid`, `machalanobis`, `cosine`}.')
        if min_cluster_count <= 0:
            raise ValueError('The `min_cluster_count` must be > 0.')

        super().__init__(distance, int(max_cluster_distance), int(min_cluster_count))

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
