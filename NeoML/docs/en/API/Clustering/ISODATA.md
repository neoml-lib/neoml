# ISODATA Algorithm CIsoDataClustering

<!-- TOC -->

- [ISODATA Algorithm CIsoDataClustering](#isodata-algorithm-cisodataclustering)
	- [Parameters](#parameters)
	- [Sample](#sample)

<!-- /TOC -->

This method is a heuristic algorithm based on geometrical proximity of the data points. For the description, see Ball, Geoffrey H., Hall, David J. *Isodata: a method of data analysis and pattern classification.* (1965)

The final result depends greatly on the initial settings.

In **NeoML** the algorithm is implemented by the `CIsoDataClustering` class that provides the `IClustering` interface. Its `Clusterize` method is used to split the data set into clusters.

## Parameters

The clustering parameters are described by the `CIsoDataClustering::CParam` structure.

- *InitialClustersCount* — the initial number of clusters
- *MaxClustersCount* — the maximum number of clusters that can be reached during processing
- *MinClusterSize* — the minimum cluster size
- *MaxIterations* — the maximum number of iterations
- *MinClustersDistance* — the minimum distance between the clusters (if it is smaller, the clusters may be merged)
- *MaxClusterDiameter* — the maximum cluster diameter (if it is exceeded the cluster may be split)
- *MeanDiameterCoef* — how much the cluster diameter may exceed the mean diameter across all the clusters (if a cluster diameter is larger than the mean diameter multiplied by this value it may be split)

## Sample

This sample shows how to use the ISODATA algorithm to clusterize the [Iris Data Set](http://archive.ics.uci.edu/ml/datasets/Iris):

```c++
void Clusterize( const IClusteringData& irisDataSet, CClusteringResult& result )
{
	CIsoDataClustering::CParam params;
	params.InitialClustersCount = 1;
	params.MaxClustersCount = 20;
	params.MinClusterSize = 1;
	params.MinClustersDistance = 0.60;
	params.MaxClusterDiameter = 1.0;
	params.MeanDiameterCoef = 0.5;
	params.MaxIterations = 50;

	CIsoDataClustering isoData( params );
	isoData.Clusterize( irisDataSet, result );
}
```