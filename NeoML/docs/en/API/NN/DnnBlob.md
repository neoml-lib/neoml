# CDnnBlob Class

<!-- TOC -->

- [CDnnBlob Class](#cdnnblob-class)
    - [Blob as a tensor](#blob-as-a-tensor)
        - [Data types supported](#data-types-supported)
        - [Memory representation](#memory-representation)
    - [Creating blobs](#creating-blobs)
        - [Data blobs](#data-blobs)
        - [Window blobs](#window-blobs)
        - [Blobs for math operations](#blobs-for-math-operations)
    - [Getting the blob size](#getting-the-blob-size)
    - [Data exchange](#data-exchange)
    - [Creating blob copies](#creating-blob-copies)
    - [Operations with data](#operations-with-data)
    - [Blob merging](#blob-merging)
    - [Blob splitting](#blob-splitting)
	    - [Parameters](#parameters)

<!-- /TOC -->

This class is used to store and transmit data in neural networks.

## Blob as a tensor

A blob is a 7-dimensional array, and each of its dimensions has a specific meaning:

- `BatchLength` is a "time" axis, used to denote data sequences; it is mainly used in recurrent networks
- `BatchWidth` corresponds to the batch, used to pass several independent objects together
- `ListSize` is the dimensions for the objects that are connected (for example, pixels out of one image) but do not form a sequence
- `Height` is the height of a matrix or an image
- `Width` is the width of a matrix or an image
- `Depth` is the width of a 3-dimensional image
- `Channels` corresponds to channels for multi-channel image formats and is also used to work with one-dimensional vectors.

### Data types supported

The blobs may contain one of the two types of data: float (`CT_Float`) and integer (`CT_Int`). Both data types are 32-bit.

If the data type is not specified directly anywhere in this documentation, that means `float` is used.

### Memory representation

The data is stored in memory so that the elements next to each other have the adjacent coordinates along the **Channels** dimension. To go to the next element along the **Depth** coordinate, you need to shift by the number of channels, and so on in the order of dimensions specified above (also known as **channel-last ordering**).

## Creating blobs

A number of static functions are provided to create blobs.

### Data blobs

```c++
static CDnnBlob* CreateDataBlob( IMathEngine& mathEngine, TDnnType type, int batchLength, int batchWidth, int channelsCount );
```

Creates a blob with the `type` data that has `batchWidth` sequences of `batchLength` length, where each element has `channelsCount` channels.

```c++
static CDnnBlob* CreateListBlob( IMathEngine& mathEngine, TDnnType type, int batchLength, int batchWidth, int listSize, int channelsCount );
```

The same as `CreateDataBlob`, but with every sequence containing `listSize` lists of elements.

```c++
static CDnnBlob* Create2DImageBlob( IMathEngine& mathEngine, TDnnType type, int batchLength, int batchWidth, int imageHeight, int imageWidth, int channelsCount );
```

Creates a blob with the `type` data that has `batchWidth` sequences of `batchLength` length, where each element is a two-dimensional image of `imageHeight` height and `imageWidth` height, with `channelsCount` channels.

```c++
static CDnnBlob* Create3DImageBlob( IMathEngine& mathEngine, TDnnType type, int batchLength, int batchWidth, int imageHeight, int imageWidth, int imageDepth, int channelsCount );
```

The same as `Create2DImageBlob`, but with 3-dimensional images of `imageDepth` depth.

### Window blobs

```c++
static CDnnBlob* CreateWindowBlob( const CPtr<CDnnBlob>& parent, int windowSize = 1 );
```

Creates a blob that is a window of `windowSize` length over the `parent` blob. Its `BatchLength` dimension is equal to `windowSize`, the other dimensions are the same as for the `parent` blob. This blob has no memory buffer of its own: it is just an auxiliary pointer to a place in the parent blob buffer, which becomes **invalid** once the parent blob is destroyed. The user is responsible for not using the window blobs after the destruction of the parent.

```c++
CDnnBlob* GetParent();
const CDnnBlob* GetParent() const;
```

Retrieves the pointer to the parent blob. This method will return `0` if it is not a window blob.

```c++
CDnnBlob* GetOwner();
const CDnnBlob* GetOwner() const;
```

Retrieves the pointer to the blob that owns the data. If the blob has not been created using `CreateWindow`, `this` will be returned.

```c++
int GetParentPos() const;
void SetParentPos( int pos );
void ShiftParentPos( int shift );
```

Retrieves, sets, or shifts the window position on the parent blob. The positions are interpreted as `BatchLength` axis coordinates.

### Blobs for math operations

```c++
static CDnnBlob* CreateTensor(IMathEngine& mathEngine, TDnnType type, std::initializer_list<int> dimensions);

// CreateVector(x) is the same as CreateTensor({x})
static CDnnBlob* CreateVector(IMathEngine& mathEngine, TDnnType type, int vectorSize);

// CreateMatrix(x, y) is the same as CreateTensor({x, y})
static CDnnBlob* CreateMatrix(IMathEngine& mathEngine, TDnnType type, int matrixHeight, int matrixWidth);
```

Creates n-dimensional, 1-dimensional, and 2-dimensional blobs. If you call the `CreateTensor` method the `dimensions` list should have not more than 7 elements.

## Getting the blob size

```c++
int GetBatchLength() const;
int GetBatchWidth() const;
int GetListSize() const;
int GetHeight() const;
int GetWidth() const;
int GetDepth() const;
int GetChannelsCount() const;

int DimSize( int d ) const;
int DimSize( TBlobDim d ) const;
```

Gets the blob size along the specified axis.

```c++
int GetDataSize() const;
```

Gets the total blob size (the product of 7 dimensions).

```c++
int GetObjectCount() const;
```

Gets the number of objects in the blob. The method returns the `BatchLength * BatchWidth * ListSize` product. Use instead of `BatchWidth` when you do not need to separate the objects along the `BatchLength` and `ListSize` dimensions.

```c++
int GetObjectSize() const;
```

Gets the size of a single object in the blob. The method returns the `Height * Width * Depth * Channels` product. Use when all objects are to be interpreted as one-dimensional vectors.

```c++
int GetGeometricalSize() const;
```

Gets the blob "geometrical" size. The method returns the `Height * Width * Depth` product. Use when `Height`, `Width`, and `Depth` dimensions may be processed together.

```c++
bool HasEqualDimensions( const CDnnBlob* other ) const;
```

Checks that the other blob has the same dimensions.

## Data exchange

```c++
template<class T = float>
void CopyFrom( const T* src );

template<class T = float>
void CopyTo( T* dst ) const;
template<class T = float>
void CopyTo( T* dst, int size ) const;
```

Passes the data of the specified type to and from the external code. If no size is specified the whole blob is copied (of `GetDataSize` size).

```c++
void CopyFrom( const CDnnBlob* other );
```

Copies the data from another blob. The dimensions and data type should be the same.

```c++
void TransposeFrom( const CDnnBlob* other, int d1, int d2 );
```

Copies the data from another blob, switching the two specified dimensions.

## Creating blob copies

```c++
CDnnBlob* GetCopy() const;
```

Creates a blob copy independent of this blob.

```c++
CDnnBlob* GetClone() const;
CDnnBlob* GetClone( TDnnType type ) const;
```

Creates a blob of the same dimensions but with data contents not initialized. The second method also changes the data type.

```c++
CDnnBlob* GetTransposed( int d1, int d2 ) const;
```

Creates a blob copy with two specified dimensions switched. The data location in memory is changed.

## Operations with data

```c++
void Add( const CDnnBlob* other );
```

Elementwise adds the other blob. It should be of the same dimensions.

```c++
void Clear();
void ClearObject( int num );
```

Fills the blob or the specified object with zeros.

```c++
template<class T = float>
void Fill( typename CDnnType<T>::TDataType value );
template<class T = float>
void FillObject( int num, typename CDnnType<T>::TDataType value );
```

Fills the blob or the specified object with the given value.

## Blob merging

```c++
static void MergeByChannels( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
static void MergeByDepth( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
static void MergeByWidth( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
static void MergeByHeight( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
static void MergeByListSize( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
static void MergeByBatchWidth( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
static void MergeByBatchLength( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
static void MergeByObject( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );

static void MergeByDim( IMathEngine& mathEngine, TBlobDim d, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
static void MergeByDim( IMathEngine& mathEngine, int d, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
```

Merges the blobs along the specified dimension.

## Blob splitting

```c++
static void SplitByChannels( IMathEngine& mathEngine, const CPtr<CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
static void SplitByDepth( IMathEngine& mathEngine, const CPtr<CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
static void SplitByWidth( IMathEngine& mathEngine, const CPtr<CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
static void SplitByHeight( IMathEngine& mathEngine, const CPtr<CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
static void SplitByListSize( IMathEngine& mathEngine, const CPtr<CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
static void SplitByBatchWidth( IMathEngine& mathEngine, const CPtr<CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
static void SplitByBatchLength( IMathEngine& mathEngine, const CPtr<CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
static void SplitByObject( IMathEngine& mathEngine, const CPtr<CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );

static void SplitByDim( IMathEngine& mathEngine, TBlobDim d, const CPtr<CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
static void SplitByDim( IMathEngine& mathEngine, int d, const CPtr<CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
```

Splits the blob along the specified dimension. The `SplitByObject` method splits along the `BatchLength * BatchWidth * ListSize` dimensions.

### Parameters

* *mathEngine* is the reference to the math engine.
* *from* is the original blob.
* *to* is the array of blobs into which the split parts will be put. The size of the parts is determined by these blobs. All the dimensions except the one to split should be the same as for the original blob. The original length of the dimension used for splitting should be equal to the total of corresponding dimension lengths of the parts blobs.
