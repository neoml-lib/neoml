.. _py-clustering:

#####################
Clustering Algorithms
#####################

The `neoml` module contains multiple clustering algoithms such as:

- :ref:`py-kmeans`
- Yet Another Clustering Algorithm

.. _py-kmeans:
.. _py-isodata:

K-Means
#######

:doc:`K-Means <../API/Clustering/kMeans>` method is the most popular clustering algorithm.
It assigns each object to tthe cluster with the nearest center.

Class description
*****************

.. autoclass:: neoml.Clustering.KMeans
   :members:

Example
*******

.. code-block:: python
   import numpy as np
   import neoml

   data = np.rand(1000, 5)
   kmeans = neoml.Clustering.KMeans(init_cluster_count=4, init='k++', algo='elkan')
   labels, centers, disps = kmeans.clusterize(data)

.. _py-isodata

ISODATA
#######

:doc:`ISODATA <../API/Clustering/ISODATA>` clustering algorithm is base on geometrical proximity of the data points.
The clustering result will depend greatly on the initial settings.

Class description
*****************

.. autoclass:: neoml.Clustering.IsoData
   :members:

Example
*******

.. code-block:: python
   import numpy as np
   import neoml

   data = np.rand(1000, 5)
   isodata = neoml.Clustering.IsoData(init_cluster_count=2, max_cluster_count=10,
                                      max_iteration_count=100, min_cluster_distance=1.,
                                      max_cluster_diameter=10., mean_diameter_coef=1.)
   labels, centers, disps = kmeans.clusterize(data)

