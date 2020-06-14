# Data Types for Machine Learning

<!-- TOC -->

- [Data Types for Machine Learning](#data-types-for-machine-learning)
    - [Vector](#vector)
        - [CFloatVector class](#cfloatvector-class)
    - [Sparse vector](#sparse-vector)
        - [The descriptor CSparseFloatVectorDesc](#the-descriptor-csparsefloatvectordesc)
        - [CSparseFloatVector class](#csparsefloatvector-class)
    - [Sparse matrix](#sparse-matrix)
        - [The descriptor CSparseFloatMatrixDesc](#the-descriptor-csparsefloatmatrixdesc)
        - [CSparseFloatMatrix class](#csparsefloatmatrix-class)

<!-- /TOC -->

## Vector

### CFloatVector class

This class implements a regular (non-sparse) vector with `float` elements.

#### Constructors

```c++
CFloatVector(); // the default constructor; used, for example, for creating an object before serializing the values from archive
// Creates a vector with 'size' elements from a sparse vector; the values absent in the sparse vector are set to 0.
CFloatVector( int size, const CSparseFloatVector& sparseVector );
CFloatVector( int size, const CSparseFloatVectorDesc& sparseVector );
explicit CFloatVector( int size ); // vectors with 'size' elements
CFloatVector( int size, float init ); // vector with 'size' elements, each equal to 'init'
CFloatVector( const CFloatVector& other );
```

#### Assignment operator

```c++
CFloatVector& operator = ( const CFloatVector& vector );
CFloatVector& operator = ( const CSparseFloatVector& vector );
```

#### Methods

```c++
CSparseFloatVector SparseVector() const;
```

Converts a regular vector into a sparse vector of the same values.

```c++
bool IsNull() const;
```

Checks that the vector is empty. Returns `true` if the vector was created by the default constructor and not initialized.

```c++
int Size() const;
```

Gets the vector size.

```c++
float operator [] ( int i );
void SetAt( int i, float what );
```

Accesses the vector elements.

```c++
void Nullify();
```

Sets all vector elements to zero.

```c++
CFloatVector& MultiplyAndAdd( const CFloatVector& vector, double factor );
CFloatVector& MultiplyAndAdd( const CSparseFloatVector& vector, double factor );
CFloatVector& MultiplyAndAdd( const CSparseFloatVectorDesc& vector, double factor );
```

Adds a vector multiplied by a factor to this one.

```c++
double Norm() const;
```

Calculates the vector L2-norm.

#### Mathematical operations

```c++
CFloatVector& operator += ( const CFloatVector& vector );
CFloatVector& operator -= ( const CFloatVector& vector );
CFloatVector& operator *= ( double factor );
CFloatVector& operator /= ( double factor );
CFloatVector& operator = ( const CSparseFloatVector& vector );
CFloatVector& operator += ( const CSparseFloatVector& vector );
CFloatVector& operator -= ( const CSparseFloatVector& vector );
```

#### Serialization

```c++
friend CArchive& operator << ( CArchive& archive, const CFloatVector& vector );
friend CArchive& operator >> ( CArchive& archive, CFloatVector& vector );
```

## Sparse vector

### The descriptor CSparseFloatVectorDesc

The descriptor stores only as much information as is necessary to retrieve the data. Changing the values via the descriptor is impossible.

```c++
struct NEOML_API CSparseFloatVectorDesc {
	int Size; // the number of allocated elements in the vector
	int* Indexes; // the coordinates of the elements in the full vector
	float* Values; // the values of the elements
};
```

### CSparseFloatVector class

Represents a sparse vector with `float` elements. The full vector size is not stored.

#### Constructors

```c++
CSparseFloatVector();
explicit CSparseFloatVector( int bufferSize );
explicit CSparseFloatVector( const CSparseFloatVectorDesc& desc );
CSparseFloatVector( const CSparseFloatVector& other );
```

#### Assignment operator

```c++
CSparseFloatVector& operator = ( const CSparseFloatVector& vector );
```

#### Methods

```c++
const CSparseFloatVectorDesc& GetDesc() const;
```

Gets the vector descriptor.

```c++
int NumberOfElements() const;
```

Gets the number of allocated elements in the vector.

```c++
bool GetValue( int index, float& value ) const; // returns true if the `index` element is allocated
float GetValue( int index ) const;
void SetAt( int index, float value );
```

Access to the elements.

```c++
void Nullify();
```

Sets all vector elements to `0`.

```c++
CSparseFloatVector& MultiplyAndAdd( const CSparseFloatVector& vector, double factor );
```

Adds a vector multiplied by a factor to this one.

```c++
double Norm() const;
```

Calculates the vector L2-norm.

#### Mathematical operations

```c++
CSparseFloatVector& operator += ( const CSparseFloatVector& vector );
CSparseFloatVector& operator -= ( const CSparseFloatVector& vector );
CSparseFloatVector& operator *= ( double factor );
CSparseFloatVector& operator /= ( double factor );
```

#### Serialization

```c++
void Serialize( CArchive& archive );
```

## Sparse matrix

### The descriptor CSparseFloatMatrixDesc

The descriptor stores only as much information as is necessary to retrieve the data. Changing the values via the descriptor is impossible.

```c++
// Sparse matrix descriptor
struct NEOML_API CSparseFloatMatrixDesc {
	int Height; // the matrix height
	int Width; // the matrix width
	int* Columns; // columns pointer
	float* Values; // pointer to the element values
	int* PointerB; // indices of vectors' beginnings in Columns/Values.
	int* PointerE; // indices of vectors' endings in Columns/Values.

	// Gets the row descriptor
	void GetRow( int index, CSparseFloatVectorDesc& desc ) const;
	CSparseFloatVectorDesc GetRow( int index ) const;
};
```

### CSparseFloatMatrix class

Represents a sparse matrix with `float` elements.

#### Constructors

```c++
CSparseFloatMatrix() {}
CSparseFloatMatrix( int width, int rowsBufferSize = 0, int elementsBufferSize = 0 );
explicit CSparseFloatMatrix( const CSparseFloatMatrixDesc& desc );
CSparseFloatMatrix( const CSparseFloatMatrix& other );
```

#### Assignment operator

```c++
CSparseFloatMatrix& operator = ( const CSparseFloatMatrix& vector );
```

#### Methods

```c++
const CSparseFloatMatrixDesc& GetDesc() const;
```

Gets the matrix descriptor.

```c++
int GetHeight() const;
int GetWidth() const;
```

Gets the matrix size.

```c++
void AddRow( const CSparseFloatVector& row );
void AddRow( const CSparseFloatVectorDesc& row );
```

Adds a row to the matrix.

```c++
CSparseFloatVectorDesc GetRow( int index ) const;
void GetRow( int index, CSparseFloatVectorDesc& desc ) const;
```

Gets a row descriptor (a row in a sparse matrix would be a sparse vector).

#### Serialization

```c++
void Serialize( CArchive& archive );
```
