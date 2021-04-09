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
const int CSparseFloatMatrix::MaxBufferSize;
const int CSparseFloatMatrix::InitialRowsBufferSize;
const int CSparseFloatMatrix::InitialElementsBufferSize;

CSparseFloatMatrix::CSparseFloatMatrixBody::CSparseFloatMatrixBody( int height, int width, int elementCount,
		int rowsBufferSize, int elementsBufferSize )
{
	NeoAssert( height >= 0 && width >= 0 && elementCount >= 0 );
	NeoAssert( rowsBufferSize >= 0 && elementsBufferSize >= 0 );

	Desc.Height = height;
	Desc.Width = width;

	rowsBufferSize = max( height, max( rowsBufferSize, InitialRowsBufferSize ) );
	BeginPointersBuf.SetBufferSize( rowsBufferSize );
	EndPointersBuf.SetBufferSize( rowsBufferSize );

	elementsBufferSize = max( elementCount, max( elementsBufferSize, InitialElementsBufferSize ) );
	ColumnsBuf.SetBufferSize( elementsBufferSize );
	ValuesBuf.SetBufferSize( elementsBufferSize );

	// leave Desc's buffers uninitialized, cause we don't fill buffers with the data in this ctor
}

CSparseFloatMatrix::CSparseFloatMatrixBody::CSparseFloatMatrixBody( const CFloatMatrixDesc& desc )
{
	NeoAssert( desc.Height >= 0 && desc.Width >= 0 );

	Desc.Height = desc.Height;
	Desc.Width = desc.Width;

	int rowsBufferSize = max( desc.Height, InitialRowsBufferSize );
	BeginPointersBuf.SetBufferSize( rowsBufferSize );
	EndPointersBuf.SetBufferSize( rowsBufferSize );
	int elementCount = 0;
	if( desc.Columns == nullptr ) {
		for( int i = 0; i < desc.Height; ++i ) {
			BeginPointersBuf.Add( elementCount );
			for( int pos = desc.PointerB[i]; pos < desc.PointerE[i]; ++pos ) {
				if( desc.Values[pos] != 0 ) {
					++elementCount;
				}
			}
			EndPointersBuf.Add( elementCount );
		}
	} else {
		for( int i = 0; i < desc.Height; ++i ) {
			BeginPointersBuf.Add( elementCount );
			elementCount += desc.PointerE[i] - desc.PointerB[i];
			EndPointersBuf.Add( elementCount );
		}
	}
	int elementsBufferSize = max( elementCount, InitialElementsBufferSize );
	ColumnsBuf.SetBufferSize( elementsBufferSize );
	ValuesBuf.SetBufferSize( elementsBufferSize );
	if( desc.Columns == nullptr ) {
		for( int i = 0; i < desc.Height; ++i ) {
			for( int pos = desc.PointerB[i], j = 0; pos < desc.PointerE[i]; ++pos, ++j ) {
				if( desc.Values[pos] != 0 ) {
					ColumnsBuf.Add( j );
					ValuesBuf.Add( desc.Values[pos] );
				}
			}
		}
	} else {
		ColumnsBuf.SetSize( elementCount );
		ValuesBuf.SetSize( elementCount );
		elementCount = 0;
		for( int i = 0; i < desc.Height; ++i ) {
			CFloatVectorDesc vec = desc.GetRow( i );
			::memcpy( ColumnsBuf.GetPtr() + elementCount, vec.Indexes, vec.Size * sizeof( int ) );
			::memcpy( ValuesBuf.GetPtr() + elementCount, vec.Values, vec.Size * sizeof( float ) );
			elementCount += vec.Size;
		}
		NeoPresume( elementCount == ValuesBuf.Size() );
	}

	Desc.Columns = ColumnsBuf.GetPtr();
	Desc.Values = ValuesBuf.GetPtr();
	Desc.PointerB = BeginPointersBuf.GetPtr();
	Desc.PointerE = EndPointersBuf.GetPtr();
}

//------------------------------------------------------------------------------------------------------------

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
	int rowsBufferSize = 1;
	int elementsBufferSize = size;
	int elementCount = 0;
	if( body != nullptr ) {
		NeoAssert( body->Desc.Height <= MaxBufferSize - 1 );
		NeoAssert( body->ValuesBuf.Size() <= MaxBufferSize - size );

		rowsBufferSize = body->Desc.Height + 1;
		elementsBufferSize = body->ValuesBuf.Size() + size;
		elementCount = body->ValuesBuf.Size();
	}
	
	CSparseFloatMatrixBody* newBody = copyOnWriteAndGrow( rowsBufferSize, elementsBufferSize );
	newBody->Desc.Height++;
	newBody->Desc.Width = max( newBody->Desc.Width, row.Indexes == nullptr ? row.Size : row.Indexes[row.Size - 1] + 1 );

	newBody->BeginPointersBuf.Add( elementCount );
	newBody->EndPointersBuf.Add( elementCount + size );
	
	if( row.Indexes == nullptr ) {
		for( int i = 0; i < row.Size; ++i ) {
			if( row.Values[i] != 0 ) {
				newBody->ColumnsBuf.Add( i );
				newBody->ValuesBuf.Add( row.Values[i] );
			}
		}
	} else {
		NeoAssert( ( row.Indexes != nullptr && row.Values != nullptr ) || row.Size == 0 );
		newBody->ColumnsBuf.SetSize( elementCount + size );
		newBody->ValuesBuf.SetSize( elementCount + size );
		::memcpy( newBody->ColumnsBuf.GetPtr() + elementCount, row.Indexes, row.Size * sizeof( int ) );
		::memcpy( newBody->ValuesBuf.GetPtr() + elementCount, row.Values, row.Size * sizeof( float ) );
	}

	if( newBody->Desc.PointerB == nullptr ) {
		NeoAssert( newBody->Desc.PointerE == nullptr );
		newBody->Desc.PointerB = newBody->BeginPointersBuf.GetPtr();
		newBody->Desc.PointerE = newBody->EndPointersBuf.GetPtr();
	}
	if( newBody->Desc.Columns == nullptr ) {
		NeoAssert( newBody->Desc.Values == nullptr );
		newBody->Desc.Columns = newBody->ColumnsBuf.GetPtr();
		newBody->Desc.Values = newBody->ValuesBuf.GetPtr();
	}
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
		archive << body->ValuesBuf.Size();
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

// duplicate like CopyOnWrite but with the preset buffers' sizes
CSparseFloatMatrix::CSparseFloatMatrixBody* CSparseFloatMatrix::copyOnWriteAndGrow( int rowsBufferSize,
	int elementsBufferSize )
{
	NeoAssert( rowsBufferSize >= 0 && elementsBufferSize >= 0 );

	if( body == nullptr ) {
		body = FINE_DEBUG_NEW CSparseFloatMatrixBody( 0, 0, 0, rowsBufferSize, elementsBufferSize );
		return body.Ptr();
	}

	if( body->RefCount() != 1 ) {
		CPtr<CSparseFloatMatrixBody> oldBody = body.Ptr();
		body = FINE_DEBUG_NEW CSparseFloatMatrixBody( oldBody->Desc.Height, oldBody->Desc.Width,
			oldBody->ValuesBuf.Size(), rowsBufferSize, elementsBufferSize );
		oldBody->ColumnsBuf.CopyTo( body->ColumnsBuf );
		oldBody->ValuesBuf.CopyTo( body->ValuesBuf );
		oldBody->BeginPointersBuf.CopyTo( body->BeginPointersBuf );
		oldBody->EndPointersBuf.CopyTo( body->EndPointersBuf );		
	} else {
		if( rowsBufferSize > body->BeginPointersBuf.BufferSize() ) {
			body->BeginPointersBuf.Grow( rowsBufferSize );
			body->EndPointersBuf.Grow( rowsBufferSize );			
		}
		if( elementsBufferSize > body->ValuesBuf.BufferSize() ) {
			body->ColumnsBuf.Grow( elementsBufferSize );
			body->ValuesBuf.Grow( elementsBufferSize );
		}
	}
	
	body->Desc.PointerB = body->BeginPointersBuf.GetPtr();
	body->Desc.PointerE = body->EndPointersBuf.GetPtr();
	body->Desc.Columns = body->ColumnsBuf.GetPtr();
	body->Desc.Values = body->ValuesBuf.GetPtr();
	return body.Ptr();
}

} // namespace NeoML
