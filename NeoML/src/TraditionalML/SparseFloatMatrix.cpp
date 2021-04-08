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

#include <common.h>
#pragma hdrstop

#include <NeoML/TraditionalML/SparseFloatMatrix.h>

namespace NeoML {

CFloatMatrixDesc CFloatMatrixDesc::Empty;
const int CSparseFloatMatrix::CSparseFloatMatrixBody::InitialRowsBufferSize;
const int CSparseFloatMatrix::CSparseFloatMatrixBody::InitialElementsBufferSize;

static void copyDescData( const CFloatMatrixDesc& dst, const CFloatMatrixDesc& src, int elementCount )
{
	::memcpy( dst.Columns, src.Columns, elementCount * sizeof( int ) );
	::memcpy( dst.Values, src.Values, elementCount * sizeof( float ) );
	::memcpy( dst.PointerB, src.PointerB, src.Height * sizeof( int ) );
	::memcpy( dst.PointerE, src.PointerE, src.Height * sizeof( int ) );
}

CSparseFloatMatrix::CSparseFloatMatrixBody::CSparseFloatMatrixBody( int height, int width, int elementCount,
		int rowsBufferSize, int elementsBufferSize ) :
	RowsBufferSize( rowsBufferSize ),
	ElementsBufferSize( elementsBufferSize ),
	ElementCount( elementCount )
{
	NeoAssert( height >= 0 && width >= 0 && elementCount >= 0 );
	NeoAssert( rowsBufferSize >= height && elementsBufferSize >= elementCount );

	Desc.Height = height;
	Desc.Width = width;

	ElementsBufferSize = max( ElementsBufferSize, InitialElementsBufferSize );
	Desc.Columns = FINE_DEBUG_NEW int[ElementsBufferSize];
	Desc.Values = FINE_DEBUG_NEW float[ElementsBufferSize];

	RowsBufferSize = max( RowsBufferSize, InitialRowsBufferSize );
	Desc.PointerB = FINE_DEBUG_NEW int[RowsBufferSize];
	Desc.PointerE = FINE_DEBUG_NEW int[RowsBufferSize];
}

CSparseFloatMatrix::CSparseFloatMatrixBody::CSparseFloatMatrixBody( const CFloatMatrixDesc& desc ) :
	RowsBufferSize( desc.Height ),
	ElementsBufferSize( 0 ),
	ElementCount( 0 )
{
	NeoAssert( desc.Height >= 0 && desc.Width >= 0 );

	Desc.Height = desc.Height;
	Desc.Width = desc.Width;

	RowsBufferSize = max( RowsBufferSize, InitialRowsBufferSize );
	Desc.PointerB = FINE_DEBUG_NEW int[RowsBufferSize];
	Desc.PointerE = FINE_DEBUG_NEW int[RowsBufferSize];

	// handle continuous sparse case (empty fits too)
	if( desc.Columns != nullptr ) {
		bool isContinuousSparse = true;
		for( int i = 0; i < desc.Height - 1 && isContinuousSparse; ++i ) {
			isContinuousSparse = desc.PointerE[i] == desc.PointerB[i+1];
		}
		if( isContinuousSparse ) {
			ElementCount = desc.Height > 0 ? ( desc.PointerE[desc.Height-1] - desc.PointerB[0] ) : 0;
			ElementsBufferSize = max( ElementCount, InitialElementsBufferSize );
			Desc.Columns = FINE_DEBUG_NEW int[ElementsBufferSize];
			Desc.Values = FINE_DEBUG_NEW float[ElementsBufferSize];
			copyDescData( Desc, desc, ElementCount );
			return;
		}
	}

	if( desc.Columns == nullptr ) {
		for( int i = 0; i < desc.Height; ++i ) {
			Desc.PointerB[i] = ElementCount;
			for( int pos = desc.PointerB[i]; pos < desc.PointerE[i]; ++pos ) {
				if( desc.Values[pos] != 0 ) {
					++ElementCount;
				}
			}
			Desc.PointerE[i] = ElementCount;
		}
	} else {
		for( int i = 0; i < desc.Height; ++i ) {
			Desc.PointerB[i] = ElementCount;
			ElementCount += desc.PointerE[i] - desc.PointerB[i];
			Desc.PointerE[i] = ElementCount;
		}
	}
	ElementsBufferSize = max( ElementCount, InitialElementsBufferSize );
	Desc.Columns = FINE_DEBUG_NEW int[ElementsBufferSize];
	Desc.Values = FINE_DEBUG_NEW float[ElementsBufferSize];
	ElementCount = 0;
	if( desc.Columns == nullptr ) {
		for( int i = 0; i < desc.Height; ++i ) {
			for( int pos = desc.PointerB[i], j = 0; pos < desc.PointerE[i]; ++pos, ++j ) {
				if( desc.Values[pos] != 0 ) {
					Desc.Columns[ElementCount] = j;
					Desc.Values[ElementCount] = desc.Values[pos];
					++ElementCount;
				}
			}
		}
	} else {
		for( int i = 0; i < desc.Height; ++i ) {
			CFloatVectorDesc vec = desc.GetRow( i );
			::memcpy( Desc.Columns + ElementCount, vec.Indexes, vec.Size * sizeof( int ) );
			::memcpy( Desc.Values + ElementCount, vec.Values, vec.Size * sizeof( float ) );
			ElementCount += vec.Size;
		}
	}
	NeoPresume( ( ElementCount == 0 && Desc.Height == 0 ) || ( ElementCount == Desc.PointerE[Desc.Height - 1] ) );
}

CSparseFloatMatrix::CSparseFloatMatrixBody::~CSparseFloatMatrixBody()
{
	delete[] Desc.Columns;
	delete[] Desc.Values;
	delete[] Desc.PointerB;
	delete[] Desc.PointerE;
}

//------------------------------------------------------------------------------------------------------------

const int CSparseFloatMatrix::MaxBufferSize;
const int sparseSignature = -1;
const int denseSignature = -2;

CSparseFloatMatrix::CSparseFloatMatrix( int width, int rowsBufferSize, int elementsBufferSize ) :
	body( FINE_DEBUG_NEW CSparseFloatMatrixBody( 0, width, 0, rowsBufferSize, elementsBufferSize ) )
{
}

CSparseFloatMatrix::CSparseFloatMatrix( const CFloatMatrixDesc& desc ) :
	body( FINE_DEBUG_NEW CSparseFloatMatrixBody( desc ) )
{
}

CSparseFloatMatrix::CSparseFloatMatrix( const CSparseFloatMatrix& matrix ) :
	body( matrix.body )
{
}

CSparseFloatMatrix& CSparseFloatMatrix::operator = ( const CSparseFloatMatrix& matrix )
{
	body = matrix.body;
	return *this;
}

void CSparseFloatMatrix::GrowInRows( int newRowsBufferSize )
{
	copyOnWriteAndGrow( newRowsBufferSize );
}

void CSparseFloatMatrix::GrowInElements( int newElementsBufferSize )
{
	copyOnWriteAndGrow( 0, newElementsBufferSize );
}

void CSparseFloatMatrix::AddRow( const CSparseFloatVector& row )
{
	AddRow( row.GetDesc() );
}

void CSparseFloatMatrix::AddRow( const CFloatVectorDesc& row )
{
	int size = row.Size;
	if( row.Indexes == nullptr ) {
		for( int i = 0; i < row.Size; ++i ) {
			if( row.Values[i] == 0 ) {
				--size;
			}
		}
	}
	int rowsBufferSize = 0;
	int elementsBufferSize = 0;
	if( body == nullptr ) {
		rowsBufferSize = 1;
		elementsBufferSize = size;
	} else {
		NeoAssert( body->Desc.Height <= MaxBufferSize - 1 );
		NeoAssert( body->ElementCount <= MaxBufferSize - size );

		rowsBufferSize = body->Desc.Height + 1;
		elementsBufferSize = body->ElementCount + size;
	}
	CSparseFloatMatrixBody* newBody = copyOnWriteAndGrow( rowsBufferSize, elementsBufferSize );
	int* indexes = newBody->Desc.Columns + newBody->ElementCount;
	float* values = newBody->Desc.Values + newBody->ElementCount;
	newBody->Desc.Height++;
	newBody->Desc.PointerB[newBody->Desc.Height - 1] = newBody->ElementCount;
	newBody->Desc.PointerE[newBody->Desc.Height - 1] = newBody->ElementCount + size;
	newBody->ElementCount += size;
	if( row.Indexes == nullptr && row.Values != nullptr ) {
		int k = 0;
		for( int i = 0; i < row.Size; ++i ) {
			if( row.Values[i] != 0 ) {
				indexes[k] = i;
				values[k] = row.Values[i];
				++k;
			}
		}
		NeoPresume( k == size );
	} else {
		NeoAssert( ( row.Indexes != nullptr && row.Values != nullptr ) || row.Size == 0 );
		::memcpy( indexes, row.Indexes, row.Size * sizeof( int ) );
		::memcpy( values, row.Values, row.Size * sizeof( float ) );
	}
	newBody->Desc.Width = max( body->Desc.Width, size == 0 ? 0 : indexes[size - 1] + 1 );
}

CFloatVectorDesc CSparseFloatMatrix::GetRow( int index ) const
{
	NeoAssert( body != nullptr );
	NeoAssert( 0 <= index && index < GetHeight() );
	return body->Desc.GetRow( index );
}

void CSparseFloatMatrix::GetRow( int index, CFloatVectorDesc& result ) const
{
	NeoAssert( body != nullptr );
	NeoAssert( 0 <= index && index < GetHeight() );
	body->Desc.GetRow( index, result );
}

void CSparseFloatMatrix::Serialize( CArchive& archive )
{
	archive.SerializeVersion( 0 );

	if( archive.IsLoading() ) {
		int elementCount = 0;
		archive >> elementCount;
		if( elementCount == 0 ) {
			body = nullptr;
			return;
		}
		int height = 0;
		int width = 0;
		archive >> height;
		archive >> width;

		int elementIndex = 0;
		CPtr<CSparseFloatMatrixBody> newBody = FINE_DEBUG_NEW CSparseFloatMatrixBody( height, width, elementCount, height, elementCount );
		for( int row = 0; row < height; row++ ) {
			newBody->Desc.PointerB[row] = elementIndex;
			int sign = archive.ReadSmallValue();
			check( sign == denseSignature || sign == sparseSignature, ERR_BAD_ARCHIVE, archive.Name() );

			if( sign == sparseSignature ) {
				int size = 0;
				archive >> size;

				if( size < 0 ) {
					check( false, ERR_BAD_ARCHIVE, archive.Name() );
				}

				for( int i = 0; i < size; i++ ) {
					archive >> newBody->Desc.Columns[elementIndex];
					archive >> newBody->Desc.Values[elementIndex];
					elementIndex++;
				}
			} else {
				int size = 0;
				archive >> size;
				int bodySize = 0;
				archive >> bodySize;
				for( int i = 0; i < size; ++i ) {
					float value;
					archive >> value;
					if( value != 0.f ) {
						newBody->Desc.Columns[elementIndex] = i;
						newBody->Desc.Values[elementIndex] = value;
						elementIndex += 1;
					}
				}
			}
			newBody->Desc.PointerE[row] = elementIndex;
		}
		body = newBody;
	} else if( archive.IsStoring() ) {
		if( body == nullptr ) {
			archive << static_cast<int>( 0 );
			return;
		}
		archive << body->ElementCount;
		archive << body->Desc.Height;
		archive << body->Desc.Width;

		for( int row = 0; row < body->Desc.Height; row++ ) {
			CFloatVectorDesc desc = GetRow( row );
			int notNullElementCount = 0;
			int lastNotNullElementIndex = NotFound;
			for( int i = 0; i < desc.Size; i++ ) {
				if( desc.Values[i] != 0.f ) {
					notNullElementCount++;
					lastNotNullElementIndex = i;
				}
			}

			// The expected serialization size
			const int denseSize = 2 * sizeof( int ) // the vector length and buffer size
				+ ( sizeof( float ) * ( notNullElementCount == 0 ? 0 : desc.Indexes[lastNotNullElementIndex] + 1 ) );
			const int sparseSize = sizeof( int ) // the buffer size
				+ ( ( sizeof( float ) + sizeof( int ) ) * notNullElementCount );

			if( sparseSize <= denseSize ) {
				archive.WriteSmallValue( sparseSignature );
				archive << notNullElementCount;
				for( int i = 0; i < desc.Size; i++ ) {
					if( desc.Values[i] != 0.f ) {
						archive << desc.Indexes[i];
						archive << desc.Values[i];
					}
				}
			} else {
				const int length = notNullElementCount == 0 ? 0 : desc.Indexes[lastNotNullElementIndex] + 1;
				archive.WriteSmallValue( denseSignature );
				archive << length;
				archive << notNullElementCount;
				for( int i = 0; i < length; ++i ) {
					archive << GetValue( desc, i );
				}
			}
		}
	} else {
		NeoAssert( false );
	}
}

template<typename T>
static void reallocAndCopy( T*& ptr, int newSize, int elementsToCopy )
{
	T* oldPtr = ptr;
	ptr = FINE_DEBUG_NEW T[newSize];
	::memcpy( ptr, oldPtr, elementsToCopy * sizeof( T ) );
	delete[] oldPtr;
}

// duplicate like CopyOnWrite but with the preset buffers' sizes
CSparseFloatMatrix::CSparseFloatMatrixBody* CSparseFloatMatrix::copyOnWriteAndGrow( int rowsBufferSize,
	int elementsBufferSize )
{
	NeoAssert( rowsBufferSize >= 0 && elementsBufferSize >= 0 );

	if( body == nullptr ) {
		body = FINE_DEBUG_NEW CSparseFloatMatrixBody( 0, 0, 0, rowsBufferSize, elementsBufferSize );
		return body.Ptr();
	}

	auto newBufferSize = []( int currentSize, int neededSize ) {
		if( neededSize > currentSize ) {
			return ( MaxBufferSize / 3 * 2 >= currentSize ) ? max( currentSize / 2 * 3, neededSize ) : MaxBufferSize;
		}
		return currentSize;
	};
	rowsBufferSize = newBufferSize( body->RowsBufferSize, rowsBufferSize );
	elementsBufferSize = newBufferSize( body->ElementsBufferSize, elementsBufferSize );
	if( body->RefCount() != 1 ) {
		auto oldBody = body.Ptr();
		body = FINE_DEBUG_NEW CSparseFloatMatrixBody( body->Desc.Height, body->Desc.Width,
			body->ElementCount, rowsBufferSize, elementsBufferSize );
		copyDescData( body->Desc, oldBody->Desc, oldBody->ElementCount );
	} else {
		if( rowsBufferSize > body->RowsBufferSize ) {
			reallocAndCopy( body->Desc.PointerB, rowsBufferSize, body->RowsBufferSize );
			reallocAndCopy( body->Desc.PointerE, rowsBufferSize, body->RowsBufferSize );
			body->RowsBufferSize = rowsBufferSize;
		}
		if( elementsBufferSize > body->ElementsBufferSize ) {
			reallocAndCopy( body->Desc.Columns, elementsBufferSize, body->ElementsBufferSize );
			reallocAndCopy( body->Desc.Values, elementsBufferSize, body->ElementsBufferSize );
			body->ElementsBufferSize = elementsBufferSize;
		}
	}
	return body.Ptr();
}

} // namespace NeoML
