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

#include <NeoML/TraditionalML/FloatMatrix.h>

namespace NeoML {

CFloatMatrixDesc CFloatMatrixDesc::Empty;

inline CFloatMatrix::CFloatMatrixBody* CFloatMatrix::CFloatMatrixBody::Duplicate() const
{
	CFloatMatrixBody* body = FINE_DEBUG_NEW CFloatMatrixBody( Desc.Height, Desc.Width );
	::memcpy( body->Desc.Values, Desc.Values, Desc.Height * Desc.Width * sizeof( float ) );
	return body;
}

CFloatMatrix::CFloatMatrixBody::CFloatMatrixBody( int width, int rowBufferSize )
{
	Desc.Height = 0;
	Desc.Width = width;

	ValuesBuf.SetSize( rowBufferSize * width );
	Desc.Values = ValuesBuf.GetPtr();
}

CFloatMatrix::CFloatMatrixBody::CFloatMatrixBody( const CFloatMatrixDesc& desc )
{
	Desc.Height = desc.Height;
	Desc.Width = desc.Width;

	ValuesBuf.SetSize( desc.Height * desc.Width );
	::memcpy( ValuesBuf.GetPtr(), desc.Values, desc.Height * desc.Width * sizeof( float ) );
	Desc.Values = ValuesBuf.GetPtr();
}

//------------------------------------------------------------------------------------------------------------

CFloatMatrix::CFloatMatrix( int width, int rowBufferSize ) :
	body( FINE_DEBUG_NEW CFloatMatrixBody( width, rowBufferSize ) )
{
}

CFloatMatrix::CFloatMatrix( const CFloatMatrixDesc& desc ) :
	body( FINE_DEBUG_NEW CFloatMatrixBody( desc ) )
{
}

CFloatMatrix::CFloatMatrix( const CFloatMatrix& vector ) :
	body( vector.body )
{
}

CFloatMatrix& CFloatMatrix::operator = ( const CFloatMatrix& vector )
{
	body = vector.body;
	return *this;
}

void CFloatMatrix::Grow( int newHeight )
{
	NeoAssert( newHeight > 0 );
	if( newHeight * body->Desc.Width > body->ValuesBuf.Size() ) {
		CFloatMatrixBody* modifiableBody = body.CopyOnWrite();
		int newBufferHeight = max( ( body->ValuesBuf.Size() / body->Desc.Width ) * 3 / 2, newHeight );
		modifiableBody->ValuesBuf.SetSize( newBufferHeight * body->Desc.Width );
		modifiableBody->Desc.Values = modifiableBody->ValuesBuf.GetPtr();
	}
}

void CFloatMatrix::AddRow( const CFloatVector& row )
{
	AddRow( row.GetPtr(), row.Size() );
}

void CFloatMatrix::AddRow( const float* row, int size )
{
	if( body == nullptr ) {
		body = FINE_DEBUG_NEW CFloatMatrixBody( size, 1 );
	} else {
		NeoAssert( body->Desc.Width == size );
	}

	Grow( body->Desc.Height + 1 );

	CFloatMatrixBody* newBody = body.CopyOnWrite();
	newBody->Desc.Width = max( body->Desc.Width, size );
	::memcpy( newBody->Desc.Values + newBody->Desc.Height * newBody->Desc.Width, row, size * sizeof( float ) );
	newBody->Desc.Height++;
}

const float* CFloatMatrix::GetRow( int index ) const
{
	NeoAssert( 0 <= index && index < GetHeight() );
	NeoAssert( body != nullptr );
	return body->Desc.Values + index * body->Desc.Width;
}

void CFloatMatrix::GetRow( int index, float* row ) const
{
	NeoAssert( 0 <= index && index < GetHeight() );
	NeoAssert( body != nullptr );
	::memcpy( row, body->Desc.Values + index * body->Desc.Width, body->Desc.Width * sizeof( float ) );
}

void CFloatMatrix::Serialize( CArchive& archive )
{
	archive.SerializeVersion( 0 );

	if( archive.IsLoading() ) {
		int height = 0;
		int width = 0;
		archive >> height;
		archive >> width;

		CPtr<CFloatMatrixBody> newBody = FINE_DEBUG_NEW CFloatMatrixBody( width, height );
		if( height > 0 ) {
			newBody->Desc.Height = height;
			archive.Read( newBody->Desc.Values, height * width * sizeof( float ) );
		}
		body = newBody;
	} else if( archive.IsStoring() ) {
		if( body == 0 ) {
			archive << static_cast<int>( 0 );
			return;
		}
		archive << body->Desc.Height;
		archive << body->Desc.Width;
		if( body->Desc.Height > 0 ) {
			archive.Write( body->Desc.Values, body->Desc.Height * body->Desc.Width * sizeof( float ) );
		}
	} else {
		NeoAssert( false );
	}
}

} // namespace NeoML
