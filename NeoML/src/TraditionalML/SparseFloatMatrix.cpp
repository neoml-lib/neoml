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
static int MaxBufferSize = INT_MAX;

namespace {
template<typename T>
class CBufferHolder {
public:
	CBufferHolder() : Data( nullptr ) {}
	CBufferHolder( T* data ) : Data( data ) {}
	CBufferHolder( CBufferHolder&& buf ) { swap( Data, buf.Data ); }
	CBufferHolder( const CBufferHolder& ) = delete;
	~CBufferHolder() { delete[] Data; }
	T* Data;
};
}

CSparseFloatMatrix::CSparseFloatMatrixBody* CSparseFloatMatrix::CSparseFloatMatrixBody::Duplicate() const
{
	CSparseFloatMatrixBody* body = FINE_DEBUG_NEW CSparseFloatMatrixBody( Desc.Height, Desc.Width, ElementCount, RowsBufferSize, ElementsBufferSize );
	CopyDataTo( body );
	return body;
}

void CSparseFloatMatrix::CSparseFloatMatrixBody::CopyDataTo( const CSparseFloatMatrixBody* dst ) const
{
	::memcpy( dst->Desc.Columns, Desc.Columns, ElementCount * sizeof( int ) );
	::memcpy( dst->Desc.Values, Desc.Values, ElementCount * sizeof( float ) );
	::memcpy( dst->Desc.PointerB, Desc.PointerB, Desc.Height * sizeof( int ) );
	::memcpy( dst->Desc.PointerE, Desc.PointerE, Desc.Height * sizeof( int ) );
}

CSparseFloatMatrix::CSparseFloatMatrixBody::CSparseFloatMatrixBody( int height, int width, int elementCount,
		int rowsBufferSize, int elementsBufferSize ) :
	RowsBufferSize( rowsBufferSize ),
	ElementsBufferSize( elementsBufferSize ),
	ElementCount( elementCount )
{
	Desc.Height = height;
	Desc.Width = width;

	Desc.Columns = new int[ElementsBufferSize];
	Desc.Values = new float[ElementsBufferSize];
	Desc.PointerB = new int[RowsBufferSize];
	Desc.PointerE = new int[RowsBufferSize];
}

CSparseFloatMatrix::CSparseFloatMatrixBody::CSparseFloatMatrixBody( const CFloatMatrixDesc& desc ) :
	RowsBufferSize( desc.Height ),
	ElementsBufferSize( desc.Height == 0 ? 0 : ( desc.Columns != nullptr ? desc.PointerE[desc.Height - 1] : 0 ) ),
	ElementCount( desc.Height == 0 ? 0 : ( desc.Columns != nullptr ? desc.PointerE[desc.Height - 1] : 0 ) )
{
	Desc.Height = desc.Height;
	Desc.Width = desc.Width;

	Desc.PointerB = new int[RowsBufferSize];
	Desc.PointerE = new int[RowsBufferSize];
	if( desc.Columns == nullptr ) {
		for( int i = 0; i < desc.Height; ++i ) {
			for( int pos = desc.PointerB[i]; pos < desc.PointerE[i]; ++pos ) {
				if( desc.Values[pos] != 0 ) {
					++ElementsBufferSize;
				}
			}
		}
		Desc.Columns = new int[ElementsBufferSize];
		Desc.Values = new float[ElementsBufferSize];
		for( int i = 0; i < desc.Height; ++i ) {
			Desc.PointerB[i] = ElementCount;
			for( int pos = desc.PointerB[i], j = 0; pos < desc.PointerE[i]; ++pos, ++j ) {
				if( desc.Values[pos] != 0 ) {
					Desc.Columns[ElementCount] = j;
					Desc.Values[ElementCount] = desc.Values[pos];
					++ElementCount;
				}
			}
			Desc.PointerE[i] = ElementCount;
		}
	} else {
		Desc.Columns = new int[ElementsBufferSize];
		Desc.Values = new float[ElementsBufferSize];
		::memcpy( Desc.Columns, desc.Columns, ElementsBufferSize * sizeof( int ) );
		::memcpy( Desc.Values, desc.Values, ElementsBufferSize * sizeof( float ) );
		::memcpy( Desc.PointerB, desc.PointerB, RowsBufferSize * sizeof( int ) );
		::memcpy( Desc.PointerE, desc.PointerE, RowsBufferSize * sizeof( int ) );
	}
}

CSparseFloatMatrix::CSparseFloatMatrixBody::~CSparseFloatMatrixBody()
{
	if( RowsBufferSize > 0 ) {
		delete[] Desc.PointerB;
		delete[] Desc.PointerE;
	}
	if( ElementsBufferSize > 0 ) {
		delete[] Desc.Columns;
		delete[] Desc.Values;
	}
}

//------------------------------------------------------------------------------------------------------------

const int sparseSignature = -1;
const int denseSignature = -2;
const int CSparseFloatMatrix::InitialRowBufferSize;
const int CSparseFloatMatrix::InitialElementBufferSize;

CSparseFloatMatrix::CSparseFloatMatrix( int width, int rowsBufferSize, int elementsBufferSize ) :
	body( FINE_DEBUG_NEW CSparseFloatMatrixBody( 0, width, 0, max( rowsBufferSize, InitialRowBufferSize ), max( elementsBufferSize, InitialElementBufferSize ) ) )
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
	if( newRowsBufferSize > body->RowsBufferSize ) {
		int newBufferSize = ( MaxBufferSize / 3 * 2 >= body->RowsBufferSize ) ?
			max( body->RowsBufferSize / 2 * 3, newRowsBufferSize ) :
			MaxBufferSize;
		const CSparseFloatMatrixBody* oldBody = body.Ptr();
		if( body->RefCount() != 1 ) {
			body = new CSparseFloatMatrixBody( body->Desc.Height, body->Desc.Width,
				body->ElementCount, body->ElementsBufferSize, newBufferSize );
			oldBody->CopyDataTo( const_cast<CSparseFloatMatrixBody*>( body.Ptr() ) );
		} else {
			CSparseFloatMatrixBody* modifiableBody = const_cast<CSparseFloatMatrixBody*>( body.Ptr() );
			{
				CBufferHolder<int> pointerB( new int[newBufferSize] );
				::memcpy( pointerB.Data, body->Desc.PointerB, body->Desc.Height * sizeof( int ) );
				swap( pointerB.Data, modifiableBody->Desc.PointerB );
			}
			{
				CBufferHolder<int> pointerE( new int[newBufferSize] );
				::memcpy( pointerE.Data, body->Desc.PointerE, body->Desc.Height * sizeof( int ) );
				swap( pointerE.Data, modifiableBody->Desc.PointerE );
			}
			modifiableBody->RowsBufferSize = newBufferSize;
		}
	}
}

void CSparseFloatMatrix::GrowInElements( int newElementsBufferSize )
{
	if( newElementsBufferSize > body->ElementsBufferSize ) {
		int newBufferSize = ( MaxBufferSize / 3 * 2 >= body->ElementsBufferSize ) ?
			max( body->ElementsBufferSize / 2 * 3, newElementsBufferSize ) :
			MaxBufferSize;
		const CSparseFloatMatrixBody* oldBody = body.Ptr();
		if( body->RefCount() != 1 ) {
			body = new CSparseFloatMatrixBody( body->Desc.Height, body->Desc.Width,
				body->ElementCount, body->ElementsBufferSize, newBufferSize );
			oldBody->CopyDataTo( body.Ptr() );
		} else {
			CSparseFloatMatrixBody* modifiableBody = const_cast<CSparseFloatMatrixBody*>( body.Ptr() );
			{
				CBufferHolder<int> columns( new int[newBufferSize] );
				::memcpy( columns.Data, body->Desc.Columns, body->ElementCount * sizeof( int ) );
				swap( columns.Data, modifiableBody->Desc.Columns );
			}
			{
				CBufferHolder<float> values( new float[newBufferSize] );
				::memcpy( values.Data, body->Desc.Values, body->ElementCount * sizeof( float ) );
				swap( values.Data, modifiableBody->Desc.Values );
			}
			modifiableBody->ElementsBufferSize = newBufferSize;
		}
	}
}

void CSparseFloatMatrix::AddRow( const CSparseFloatVector& row )
{
	AddRow( row.GetDesc() );
}

void CSparseFloatMatrix::AddRow( const CFloatVectorDesc& row )
{
	if( body == 0 ) {
		body = FINE_DEBUG_NEW CSparseFloatMatrixBody( 0, 0, 0, InitialRowBufferSize,
			max( row.Size, InitialElementBufferSize ) );
	}

	int size = row.Size;
	if( row.Indexes == nullptr ) {
		for( int i = 0; i < row.Size; ++i ) {
			if( row.Values[i] == 0 ) {
				--size;
			}
		}
	}

	GrowInRows( body->Desc.Height + 1 );
	if( size > 0 ) {
		assert( body->ElementCount <= MaxBufferSize - size );
		GrowInElements( body->ElementCount + size );
	}

	CSparseFloatMatrixBody* newBody = body.CopyOnWrite();
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
			body = 0;
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
		if( body == 0 ) {
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

} // namespace NeoML
