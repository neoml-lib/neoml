.. _py-clustering-submodule:

################
neoml.Clustering
################

The `neoml` module contains multiple clustering algoithms such as:

- :ref:`py-kmeans`
- :ref:`py-isodata`
- :ref:`py-hierarchical`
- :ref:`py-first-come`

.. _py-kmeans:

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
   kmeans = neoml.Clustering.KMeans(cluster_count=4, init='k++', algo='elkan')
   labels, centers, disps = kmeans.clusterize(data)

.. _py-isodata:

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
   labels, centers, disps = isodata.clusterize(data)

.. _py-hierarchical:

Hierarchical clustering
#######################

The library provides a "naive" implemetation of upward :doc:`hierarchical clustering<../API/Clustering/Hierarchical>`.
First, it creates a cluster per element, the merges clusters on each step until the final cluster is achieved.

Class description
*****************

.. autoclass:: neoml.Clustering.Hierarchical
   :members:

Example
*******

.. code-block:: python

   import numpy as np
   import neoml

   data = np.rand(1000, 5)
   hierarchical = neoml.Clustering.Hierarchical(max_cluster_distance=2., min_cluster_count=2,
                                                distance='euclid')
   labels, centers, disps = hierarchical.clusterize(data)

.. _py-first-come:

First come clustering
#####################

A :doc:`simple clustering algorithm <../API/Clustering/FirstCome>` that create a new cluster for each new vector
that is far enough from the clusters already existing.

Class description
*****************

.. autoclass:: neoml.Clustering.FirstCome
   :members:

Example
*******

.. code-block:: python

   import numpy as np
   import neoml

   data = np.rand(1000, 5)
   first_come = neoml.Clustering.FirstCome(min_vector_count=5, default_variance=2.,
                                           threshold=0., min_cluster_size_ratio=0.1,
                                           max_cluster_count=25, distance='euclid')
   labels, centers, disps = first_come.clusterize(data)

