# First Come Clustering Algorithm CFirstComeClustering

<!-- TOC -->

- [First Come Clustering Algorithm CFirstComeClustering](#first-come-clustering-algorithm-cfirstcomeclustering)
	- [Parameters](#parameters)
	- [Sample](#sample)

<!-- /TOC -->

This simple clustering algorithm works with only one run through the data set. Each new vector is added to the nearest cluster, or if all the clusters are too far, a new cluster will be created for this vector. At the end, the clusters that are too small (smaller than *MinClusterSizeRatio*) are destroyed and their vectors reassigned to other clusters.

In **NeoML** the algorithm is implemented by the `CFirstComeClustering` class that provides the `IClustering` interface. Its `Clusterize` method is used to split the data set into clusters.

## Parameters

The clustering parameters are described by the  `CFirstComeClustering::CParam` structure.

- *DistanceFunc* — the distance function
- *MinVectorCountForVariance* — the smallest number of vectors in a cluster for which the variance is considered valid
- *DefaultVariance* — the default variance value (to be used when a cluster has less than *MinVectorCountForVariance* elements);
- *Threshold* — the distance threshold for a new cluster to be created
- *MinClusterSizeRatio* — the minimum number of vectors in a cluster (ratio of the total number of vectors, values from 0 to 1);
- *MaxClusterCount* — the maximum number of clusters (used to make sure the algorithm does not create too many clusters in the cases when the input data has great differences)

## Sample

This sample shows how to use the first come clustering algorithm to clusterize the [Iris Data Set](http://archive.ics.uci.edu/ml/datasets/Iris):

```c++
void Clusterize( const IClusteringData& irisDataSet, CClusteringResult& result )
{
	CFirstComeClustering::CParam params;
	params.Threshold = 5;

	CFirstComeClustering firstCome( params );
	firstCome.Clusterize( irisDataSet, result );
}
```