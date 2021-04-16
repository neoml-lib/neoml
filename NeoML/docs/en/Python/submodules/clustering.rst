.. _py-submodule-clustering:

################
neoml.Clustering
################

The `neoml` module provides several methods for clustering data.

- :ref:`py-clustering-kmeans`
- :ref:`py-clustering-isodata`
- :ref:`py-clustering-hierarchical`
- :ref:`py-clustering-first-come`

.. _py-clustering-kmeans:

K-means
#######

K-means method is the most popular clustering algorithm.

On each step, the mass center for each cluster is calculated, and then the vectors are reassigned to clusters with the nearest center. The algorithm stops on the step when the in-cluster distance does not change.

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

.. _py-clustering-isodata:

ISODATA
#######

`ISODATA clustering algorithm <https://github.com/neoml-lib/neoml/blob/master/NeoML/docs/en/API/Clustering/ISODATA.md>`_
is based on geometrical proximity of the data points.
The clustering result will depend greatly on the initial settings.

See Ball, Geoffrey H., Hall, David J. Isodata: a method of data analysis and pattern classification. (1965)

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

.. _py-clustering-hierarchical:

Hierarchical clustering
#######################

The library provides a "naive" implemetation of upward hierarchical clustering.
The initial state has a cluster for every element. On each step, the two closest 
clusters are merged. Once the target number of clusters is reached, or all clusters
are too far from each other to be merged, the process ends.

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

.. _py-clustering-first-come:

First come clustering
#####################

A simple clustering algorithm that works with only one run through the data set. 
Each new vector is added to the nearest cluster, or if all the clusters are too far, 
a new cluster will be created for this vector. At the end, the clusters that are too small are destroyed and their vectors redistributed.

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

