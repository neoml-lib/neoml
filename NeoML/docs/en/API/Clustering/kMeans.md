# K-Means Method CKMeansClustering

<!-- TOC -->

- [K-Means Method CKMeansClustering](#k-means-method-ckmeansclustering)
	- [Parameters](#parameters)
	- [Sample](#sample)

<!-- /TOC -->

The k-means algorithm is the most popular for clustering.

On each step, the mass center for each cluster is calculated, and then the vectors are reassigned to clusters with the nearest center (according to the chosen distance function). 

The algorithm stops on the step when the in-cluster distance does not change. This is guaranteed to happen within a finite number of iterations because the number of possible splits of a finite set is finite, and the total quadratic deviation decreases every time.

In **NeoML** the algorithm is implemented by the `CKMeansClustering` class that provides the `IClustering` interface. Its `Clusterize` method is used to split the data set into clusters.

## Parameters

The clustering parameters are described by the `CKMeansClustering::CParam` structure.

- *Algo* - algorithm used during clusterization
- *DistanceFunc* — the distance function
- *InitialClustersCount* — the initial cluster count: when creating the object, you may pass the array (*InitialClustersCount* long) with the centers of the initial clusters to the constructor; otherwise, the random selection of input data will be taken as cluster centers on the first step
- *Initialization* - the initialization algorithm
- *MaxIterations* — the maximum number of algorithm iterations
- *Tolerance* - tolerance for stop criteria of Elkan algorithm
- *ThreadCount* - number of threads used during calculations
- *RunCount* - number of runs of the alogrithm (the result with least inertia will be returned)
- *Seed* - the initial seed for random

## Sample

This sample shows how to use the k-means algorithm to clusterize the [Iris Data Set](http://archive.ics.uci.edu/ml/datasets/Iris):

```c++
void Clusterize( const IClusteringData& irisDataSet, CClusteringResult& result )
{
	CKMeansClustering::CParam params;
	params.Algorithm = CKMeansClustering::KMA_Lloyd;
	params.DistanceFunc = DF_Euclid;
	params.InitialClustersCount = 3;
	params.Initialization = CKMeansClustering::KMI_Default;
	params.MaxIterations = 50;
	// params.Tolerance is omitted because it isn't used by Lloyd algo

	CKMeansClustering kMeans( params );
	kMeans.Clusterize( irisDataSet, result );
}
```