# Solving Clustering Problems

<!-- TOC -->

- [Solving Clustering Problems](#solving-clustering-problems)
	- [Algorithms](#algorithms)
		- [K-means](#k-means)
		- [ISODATA](#ISODATA)
		- [Hierarchical clustering](#hierarchical-clustering)
		- [First come clustering](#first-come-clustering)
	- [Auxiliary interfaces](#auxiliary-interfaces)
		- [Problem interface IClusteringData](#problem-interface-iclusteringdata)
		- [Algorithm interface IClustering](#algorithm-interface-iclustering)
		- [Clustering result CClusteringResult](#clustering-result-cclusteringresult)

<!-- /TOC -->

**NeoML** library provides several methods for clustering data.

## Algorithms

### K-means

[K-means method](kMeans.md) is the most popular clustering algorithm. It assigns each object to the cluster with the nearest center. Implemented by the `CKMeansClustering` class.

### ISODATA

[ISODATA clustering algorithm](ISODATA.md) is based on geometrical proximity of the data points. The clustering result will depend greatly on the initial settings. Implemented by the `CIsoDataClustering`class.

### Hierarchical clustering

The library provides a "naive" implementation of upward [hierarchical clustering](Hierarchical.md). First, it creates a cluster per element, then merges clusters on each step until the final cluster set is achieved. Implemented by the `CHierarchialClustering` class.

### First come clustering

A [simple clustering algorithm](FirstCome.md) that creates a new cluster for each new vector that is far enough from the clusters already existing. Implemented by the `CFirstComeClustering` class.

## Auxiliary interfaces

### Problem interface IClusteringData

The input data to be split into clusters is passed to any of the algorithms as a pointer to the object that implements the `IClusteringData` interface:

```c++
class IClusteringData : public virtual IObject {
public:
	// The number of vectors
	virtual int GetVectorCount() const = 0;

	// The number of features
	virtual int GetFeaturesCount() const = 0;

	// Gets all input vectors as a matrix of size GetVectorCount() x GetFeaturesCount()
	virtual CFloatMatrixDesc GetMatrix() const = 0;

	// Gets the vector weight
	virtual double GetVectorWeight( int index ) const = 0;
};
```

### Algorithm interface IClustering

Every clustering algorithm implements the `IClustering` interface.

```c++
class IClustering {
public:
	virtual ~IClustering() {};

	// Clusterizes the input data 
	// and returns true if successful with the given parameters
	virtual bool Clusterize( const IClusteringData* data, CClusteringResult& result ) = 0;
};
```

### Clustering result CClusteringResult

The clustering result is described by the `CClusteringResult` structure.

```c++
class NEOML_API CClusteringResult {
public:
	int ClusterCount;
	CArray<int> Data;
	CArray<CClusterCenter> Clusters;
};
```

- *ClusterCount* — the number of clusters
- *Data* — the array of cluster numbers for each of the input data elements (the clusters are numbered from 0 to ClusterCount - 1)
- *Clusters* — the cluster centers
