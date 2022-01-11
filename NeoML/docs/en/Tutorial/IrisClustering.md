# Solving a Sample Clustering Problem

<!-- TOC -->

- [Solving a Sample Clustering Problem](#solving-a-sample-clustering-problem)
	- [Preparing the input data](#preparing-the-input-data)
	- [Implementing the clustering interface](#implementing-the-clustering-interface)
	- [Running the clustering process](#running-the-clustering-process)
	- [Analyzing the results](#analyzing-the-results)

<!-- /TOC -->

This tutorial walks through using **NeoML** to clusterize the well-known [Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris). We will use the *k-means* clustering algorithm (represented by the [CKMeansClustering](../API/Clustering/kMeans.md) class).

## Preparing the input data

We assume that the input data set is serialized in a file on disk as a `CSparseFloatMatrix`. The library serialization methods can be used to load the data into memory for processing.

```c++
CSparseFloatMatrix matrix;
CArchiveFile file( "iris.carchive", CArchive::load );
CArchive archive( &file, CArchive::load );
archive >> matrix;
```

## Implementing the clustering interface

Each clustering algorithm receives the data as [IClusteringData](../API/Clustering/README.md) object; we will implement this interface over `CSparseFloatMatrix`.

```c++
class CClusteringData : public IClusteringData {
public:
	explicit CClusteringData( const CSparseFloatMatrix& _matrix ) :
		matrix( _matrix )
	{
	}

	virtual int GetVectorCount() const { return matrix.GetHeight(); }
	virtual int GetFeaturesCount() const { return matrix.GetWidth(); }
	virtual CFloatMatrixDesc GetMatrix() const { return matrix.GetDesc(); }
	virtual double GetVectorWeight( int /*index*/ ) const { return 1.0; }

private:
	CSparseFloatMatrix matrix;
};

CPtr<CClusteringData> data = new CClusteringData( matrix );
```

## Running the clustering process

Once the data is ready, we can set up the clustering algorithm. Use the `CParam` class object and set:

- *InitialClustersCount* to 3, as the data set has 3 clusters.
- *DistanceFunc* to `DF_Euclid`, so that euclidean distance will be used.
- *MaxIterations* to the number of elements in the data set.

```c++
CKMeansClustering::CParam params;
params.InitialClustersCount = 3;
params.DistanceFunc = DF_Euclid;	
params.MaxIterations = data->GetVectorCount();

CKMeansClustering kMeans( params );

CClusteringResult result;
kMeans.Clusterize( data, result );
```

## Analyzing the results

Printing out the clustering results:

```c++
printf("Count %d:\n", result.ClusterCount );
for( int i = 0; i < result.ClusterCount; i++ ) {
	for( int j = 0; j < result.Data.Size(); j++ ) {
		if( result.Data[j] == i ) {
			printf("%d ", j );
		}
	}
	printf("\n");
}
```

It can be seen that the algorithm actually split the incoming data into three clusters:

```
Count 3:
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
50 51 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 101 106 113 114 119 121 123 126 127 133 138 142 146 149
52 77 100 102 103 104 105 107 108 109 110 111 112 115 116 117 118 120 122 124 125 128 129 130 131 132 134 135 136 137 139 140 141 143 144 145 147 148
```
