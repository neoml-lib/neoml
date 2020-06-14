/* Copyright © 2017-2020 ABBYY Production LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--------------------------------------------------------------------------------------------------------------*/

#pragma once

namespace NeoML {

// A variable size matrix
template<class T>
class CVariableMatrix {
public:
	typedef T ElementType;

	CVariableMatrix();
	CVariableMatrix( int width, int height );

	void Set( const T& );
	void SetColumn( int xPosition, const T& );
	void SetColumn( int xPosition, const T* values, int valuesCount );
	void SetRow( int yPosition, const T& );

	// Search using the == operator
	bool Find( const T& what, int& x, int& y ) const;
	// Checks if an element is present in the matrix, using linear search and comparison operator
	bool Has( const T& what ) const;

	T& operator() ( int xPosition, int yPosition );
	const T& operator() ( int xPosition, int yPosition ) const;
	// Traverses the matrix by columns
	const T* Column(int xPosition) const;
	T* Column(int xPosition);

	bool operator ==( const CVariableMatrix<T>& ) const;
	bool operator !=( const CVariableMatrix<T>& ) const;

	void Serialize( CArchive& );

	int SizeX() const { return xSize; }
	int SizeY() const { return ySize; }
	int DataSize() const { return data.Size(); }

	void SetSize( int sizeX, int sizeY );
	void AddColumn();
	void Reset() { SetSize( 0, 0 ); }
	void SetBufferSize( int sizeX, int sizeY );

	void CopyTo(CVariableMatrix<T>& dest) const;

private:
	int xSize;		// the number of columns
	int ySize;		// the number of rows
	CArray<T> data; // the data array
	
	CVariableMatrix(const CVariableMatrix&); // copy constructor prohibited
};

//--------------------------------------------------------------------------------------------------

template<class T>
inline CVariableMatrix<T>::CVariableMatrix() : xSize(0), ySize(0)
{
}

template<class T>
inline CVariableMatrix<T>::CVariableMatrix( int width, int height )
{
	SetSize( width, height );
}

template<class T>
inline void CVariableMatrix<T>::SetSize( int width, int height )
{
	xSize = width;
	ySize = height;
	data.SetSize( xSize * ySize );
}

template<class T>
inline void CVariableMatrix<T>::SetBufferSize( int width, int height )
{
	data.SetBufferSize( width * height );
}

template<class T>
inline void CVariableMatrix<T>::AddColumn()
{
	// Made easier by the fact that the data is stored by columns
	SetSize( xSize + 1, ySize );
}

template<class T>
inline void CVariableMatrix<T>::Set( const T& src )
{
	for( int i = 0; i < data.Size(); i++ ) {
		data[i] = src;
	}
}

template<class T>
inline void CVariableMatrix<T>::SetColumn( int xPosition, const T& src )
{
	NeoPresume( xPosition >= 0 && xPosition < xSize );
	T* column = &data[xPosition * ySize];
	for( int y = 0; y < ySize; y++ ) {
		column[y] = src;
	}
}

template<class T>
inline void CVariableMatrix<T>::SetColumn( int xPosition, const T* values, int valuesCount )
{
	NeoPresume( xPosition >= 0 && xPosition < xSize );
	NeoPresume( valuesCount == ySize );

	T* column = &data[xPosition * valuesCount];
	for( int y = 0; y < valuesCount; y++ ) {
		column[y] = values[y];
	}
}

template<class T>
inline void CVariableMatrix<T>::SetRow( int yPosition, const T& src )
{
	NeoPresume( yPosition >= 0 && yPosition < ySize );
	T* row = &data[yPosition];
	for( int x = 0; x < xSize; x++ ) {
		*row = src;
		row += ySize;
	}
}

template<class T>
inline const T* CVariableMatrix<T>::Column(int xPosition) const
{
	NeoPresume( xPosition >= 0 && xPosition < xSize );
	return &data[xPosition * ySize]; 
}

template<class T>
inline T* CVariableMatrix<T>::Column(int xPosition)
{ 
	NeoPresume( xPosition >= 0 && xPosition < xSize );
	return &data[xPosition * ySize]; 
}

template<class T>
inline const T& CVariableMatrix<T>::operator() ( int xPosition, int yPosition ) const
{
	NeoPresume( xPosition >= 0 && xPosition < xSize );
	NeoPresume( yPosition >= 0 && yPosition < ySize );
	return data[xPosition * ySize + yPosition];
}

template<class T>
inline T& CVariableMatrix<T>::operator() ( int xPosition, int yPosition )
{
	NeoPresume( xPosition >= 0 && xPosition < xSize );
	NeoPresume( yPosition >= 0 && yPosition < ySize );
	return data[xPosition * ySize + yPosition];
}

template<class T>
inline bool CVariableMatrix<T>::Find( const T& what, int& x, int& y ) const
{
	x = NotFound;
	y = NotFound;
	int pos = data.Find(what);
	if(pos == NotFound) {
		return false;
	} else {
		x = pos / ySize;
		y = pos % ySize;
		return true;
	}
}

template<class T>
inline bool CVariableMatrix<T>::Has( const T& what ) const
{
	return data.Has(what);
}

template<class T>
inline bool CVariableMatrix<T>::operator ==( const CVariableMatrix<T>& matrix ) const
{
	if(xSize != matrix.xSize || ySize != matrix.ySize) {
		return false;
	}
	for( int i = 0; i < data.Size(); i++ ) {
		if( !( data[i] == matrix.data[i] ) ) {
			return false;
		}
	}
	return true;
}

template<class T>
inline bool CVariableMatrix<T>::operator !=( const CVariableMatrix<T>& matrix ) const
{
	return !( *this == matrix );
}

template<class T>
inline void CVariableMatrix<T>::CopyTo( CVariableMatrix<T>& dest ) const
{
	dest.xSize = xSize;
	dest.ySize = ySize;
	data.CopyTo(dest.data);
}

template<class T>
inline void CVariableMatrix<T>::Serialize( CArchive& ar )
{
	if( ar.IsLoading() ) {
		ar >> xSize;
		ar >> ySize;
		data.SetSize(xSize * ySize);
		for( int i = 0; i < data.Size(); i++ ) {
			ar >> data[i];
		}
	} else if( ar.IsStoring() ) {
		ar << xSize;
		ar << ySize;
		for( int i = 0; i < data.Size(); i++ ) {
			ar << data[i];
		}
	} else {
		NeoAssert( false );
	}
}

} // namespace NeoML
