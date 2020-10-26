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
#include <NeoML/Buffer.h>

namespace NeoML {

CSparseFloatMatrixDesc CSparseFloatMatrixDesc::Empty;

inline CSparseFloatMatrix::CSparseFloatMatrixBody* CSparseFloatMatrix::CSparseFloatMatrixBody::Duplicate() const
{
	CSparseFloatMatrixBody* body = FINE_DEBUG_NEW CSparseFloatMatrixBody( Desc.Height, Desc.Width, ElementCount, RowsBufferSize, ElementsBufferSize );
	body->ColumnsBuf.CopyFrom( Desc.Columns, ElementsBufferSize );
	body->ValuesBuf.CopyFrom( Desc.Values, ElementsBufferSize );
	body->BeginPointersBuf.CopyFrom( Desc.PointerB, Desc.Height );
	body->EndPointersBuf.CopyFrom( Desc.PointerE, Desc.Height );
	return body;
}

CSparseFloatMatrix::CSparseFloatMatrixBody::CSparseFloatMatrixBody( int height, int width, int elementCount,
		int rowsBufferSize, int elementsBufferSize ) :
	RowsBufferSize( rowsBufferSize ),
	ElementsBufferSize( elementsBufferSize ),
	ElementCount( elementCount ),
	ColumnsBuf( ElementsBufferSize ),
	ValuesBuf( ElementsBufferSize ),
	BeginPointersBuf( RowsBufferSize ),
	EndPointersBuf( RowsBufferSize )
{
	NeoAssert( RowsBufferSize >= 0 );
	NeoAssert( ElementsBufferSize >= 0 );
	Desc.Height = height;
	Desc.Width = width;

	Desc.Columns = ColumnsBuf;
	Desc.Values = ValuesBuf;
	Desc.PointerB = BeginPointersBuf;
	Desc.PointerE = EndPointersBuf;
}

CSparseFloatMatrix::CSparseFloatMatrixBody::CSparseFloatMatrixBody( const CSparseFloatMatrixDesc& desc ) :
	RowsBufferSize( desc.Height ),
	ElementsBufferSize( desc.Height == 0 ? 0 : desc.PointerE[desc.Height - 1] ),
	ElementCount( desc.Height == 0 ? 0 : desc.PointerE[desc.Height - 1] ),
	ColumnsBuf( ElementsBufferSize ),
	ValuesBuf( ElementsBufferSize ),
	BeginPointersBuf( RowsBufferSize ),
	EndPointersBuf( RowsBufferSize )
{
	NeoAssert( RowsBufferSize >= 0 );
	NeoAssert( ElementsBufferSize >= 0 );
	Desc.Height = desc.Height;
	Desc.Width = desc.Width;

	ColumnsBuf.CopyFrom( Desc.Columns, ElementsBufferSize );
	ValuesBuf.CopyFrom( Desc.Values, ElementsBufferSize );
	BeginPointersBuf.CopyFrom( Desc.PointerB, RowsBufferSize );
	EndPointersBuf.CopyFrom( Desc.PointerE, RowsBufferSize );

	Desc.Columns = ColumnsBuf;
	Desc.Values = ValuesBuf;
	Desc.PointerB = BeginPointersBuf;
	Desc.PointerE = EndPointersBuf;
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

CSparseFloatMatrix::CSparseFloatMatrix( const CSparseFloatMatrixDesc& desc ) :
	body( FINE_DEBUG_NEW CSparseFloatMatrixBody( desc ) )
{
}

CSparseFloatMatrix::CSparseFloatMatrix( const CSparseFloatMatrix& vector ) :
	body( vector.body )
{
}

CSparseFloatMatrix& CSparseFloatMatrix::operator = ( const CSparseFloatMatrix& vector )
{
	body = vector.body;
	return *this;
}

void CSparseFloatMatrix::GrowInRows( int newRowsBufferSize )
{
	assert( newRowsBufferSize > 0 );
	if( newRowsBufferSize > body->RowsBufferSize ) {
		CSparseFloatMatrixBody* modifiableBody = body.CopyOnWrite();
		int newBufferSize = max( body->RowsBufferSize * 3 / 2, newRowsBufferSize );

		// enter scope to free unused memory immediately
		{
			CBuffer<int> pointerB( newBufferSize );
			pointerB.CopyFrom( body->Desc.PointerB, body->Desc.Height );
			pointerB.Swap( modifiableBody->BeginPointersBuf );
		}

		CBuffer<int> pointerE( newBufferSize );
		pointerE.CopyFrom( body->Desc.PointerE, body->Desc.Height );
		pointerE.Swap( modifiableBody->EndPointersBuf );

		modifiableBody->Desc.PointerB = modifiableBody->BeginPointersBuf;
		modifiableBody->Desc.PointerE = modifiableBody->EndPointersBuf;
		modifiableBody->RowsBufferSize = newBufferSize;
	}
}

void CSparseFloatMatrix::GrowInElements( int newElementsBufferSize )
{
	assert( newElementsBufferSize > 0 );
	if( newElementsBufferSize > body->ElementsBufferSize ) {
		CSparseFloatMatrixBody* modifiableBody = body.CopyOnWrite();
		int newBufferSize = max( body->ElementsBufferSize * 3 / 2, newElementsBufferSize );

		// enter scope to free unused memory immediately
		{
			CBuffer<int> columns( newBufferSize );
			columns.CopyFrom( body->Desc.Columns, body->ElementCount );
			columns.Swap( modifiableBody->ColumnsBuf );
		}

		CBuffer<float> values( newBufferSize );
		values.CopyFrom( body->Desc.Values, body->ElementCount );
		values.Swap( modifiableBody->ValuesBuf );

		modifiableBody->Desc.Columns = modifiableBody->ColumnsBuf;
		modifiableBody->Desc.Values = modifiableBody->ValuesBuf;
		modifiableBody->ElementsBufferSize = newBufferSize;
	}
}

void CSparseFloatMatrix::AddRow( const CSparseFloatVector& row )
{
	AddRow( row.GetDesc() );
}

void CSparseFloatMatrix::AddRow( const CSparseFloatVectorDesc& row )
{
	if( body == 0 ) {
		body = FINE_DEBUG_NEW CSparseFloatMatrixBody( 0, 0, 0, InitialRowBufferSize, max( row.Size, InitialElementBufferSize ) );
	}

	GrowInRows( body->Desc.Height + 1 );
	GrowInElements( body->ElementCount + row.Size );

	CSparseFloatMatrixBody* newBody = body.CopyOnWrite();
	newBody->Desc.Height++;
	newBody->Desc.Width = max( body->Desc.Width, row.Size == 0 ? 0 : row.Indexes[row.Size - 1] + 1 );
	newBody->Desc.PointerB[newBody->Desc.Height - 1] = newBody->ElementCount;
	newBody->Desc.PointerE[newBody->Desc.Height - 1] = newBody->ElementCount + row.Size;
	::memcpy( newBody->Desc.Columns + newBody->ElementCount, row.Indexes, row.Size * sizeof( int ) );
	::memcpy( newBody->Desc.Values + newBody->ElementCount, row.Values, row.Size * sizeof( float ) );
	newBody->ElementCount += row.Size;
}

CSparseFloatVectorDesc CSparseFloatMatrix::GetRow( int index ) const
{
	NeoAssert( 0 <= index && index < GetHeight() );

	CSparseFloatVectorDesc res;
	res.Size = body == 0 ? 0 : body->Desc.PointerE[index] - body->Desc.PointerB[index];
	res.Indexes = body == 0 ? 0 : body->Desc.Columns + body->Desc.PointerB[index];
	res.Values = body == 0 ? 0 : body->Desc.Values + body->Desc.PointerB[index];
	return res;
}

void CSparseFloatMatrix::GetRow( int index, CSparseFloatVectorDesc& result ) const
{
	NeoAssert( 0 <= index && index < GetHeight() );

	result.Size = body->Desc.PointerE[index] - body->Desc.PointerB[index];
	result.Indexes = body->Desc.Columns + body->Desc.PointerB[index];
	result.Values = body->Desc.Values + body->Desc.PointerB[index];
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
			CSparseFloatVectorDesc desc = GetRow( row );
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
