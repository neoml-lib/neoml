# Hierarchical Clustering CHierarchicalClustering

<!-- TOC -->

- [Hierarchical Clustering CHierarchicalClustering](#hierarchical-clustering-chierarchicalclustering)
	- [Parameters](#parameters)
	- [Sample](#sample)

<!-- /TOC -->

**NeoML** library provides a naive realization of upward hierarchical clustering.

The initial state has a cluster for every element. On each step, the two closest clusters are merged. Once the target number of clusters is reached, or all clusters are too far from each other to be merged, the process ends.

In **NeoML** algorithm is implemented by the `CHierarchicalClustering` class that provides the `IClustering` interface. Its `Clusterize` method is used to split the data set into clusters.

## Parameters

The clustering parameters are described by the `CHierarchicalClustering::CParam` structure.

- *DistanceType* — distance function
- *MaxClustersDistance* — maximum distance at which the two clusters may be merged
- *MinClustersCount* — minimum number of clusters in the result

## Sample

In this sample an input data set is split into two clusters:

```c++
void Clusterize( const IClusteringData& data, CClusteringResult& result )
{
	CHierarchicalClustering::CParam params;
	params.DistanceType = DF_Euclid;
	params.MinClustersCount = 2;
	params.MaxClustersDistance = 10.f;

	CHierarchicalClustering hierarchical( params );
	hierarchical.Clusterize( data, result );
}
```
