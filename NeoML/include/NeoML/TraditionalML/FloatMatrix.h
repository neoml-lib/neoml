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

#include <NeoML/NeoMLDefs.h>

#include <NeoML/TraditionalML/FloatVector.h>

namespace NeoML {

// Float matrix description
struct NEOML_API CFloatMatrixDesc {
	int Height; // the matrix height
	int Width; // the matrix width
	float* Values; // data array

	CFloatMatrixDesc() : Height( 0 ), Width( 0 ), Values( nullptr ) {}

	const float* GetRow( int index ) const;

	static CFloatMatrixDesc Empty; // the empty matrix descriptor
};

inline const float* CFloatMatrixDesc::GetRow( int index ) const
{
	NeoAssert( 0 <= index && index < Height );
	NeoAssert( Values != nullptr );
	return Values + index * Width;
}

//---------------------------------------------------------------------------------------------------------

class NEOML_API CFloatMatrix {
public:
	CFloatMatrix() {}
	explicit CFloatMatrix( int width, int rowBufferSize = 0 );
	explicit CFloatMatrix( const CFloatMatrixDesc& desc );
	CFloatMatrix( const CFloatMatrix& other );

	CFloatMatrixDesc* CopyOnWrite() { return body == nullptr ? nullptr : &body.CopyOnWrite()->Desc; }
	const CFloatMatrixDesc& GetDesc() const { return body == nullptr ? CFloatMatrixDesc::Empty : body->Desc; }

	int GetHeight() const { return body == nullptr ? 0 : body->Desc.Height; }
	int GetWidth() const { return body == nullptr ? 0 : body->Desc.Width; }

	void Grow( int newHeight );

	void AddRow( const CFloatVector& row );
	void AddRow( const float* row, int size );
	const float* GetRow( int index ) const;
	void GetRow( int index, float* row ) const;

	CFloatMatrix& operator = ( const CFloatMatrix& vector );

	void Serialize( CArchive& archive );

private:
	struct NEOML_API CFloatMatrixBody : public IObject {
		CFloatMatrixDesc Desc;
		CArray<float> ValuesBuf;

		CFloatMatrixBody( int height, int width );
		explicit CFloatMatrixBody( const CFloatMatrixDesc& desc );
		~CFloatMatrixBody() override = default;

		CFloatMatrixBody* Duplicate() const;
	};

	CCopyOnWritePtr<CFloatMatrixBody> body; // The matrix body.
};

inline CTextStream& operator<<( CTextStream& stream, const CFloatMatrix& matrix )
{
	for( int rowIndex = 0; rowIndex < matrix.GetHeight(); ++rowIndex ) {
		const float* row = matrix.GetRow( rowIndex );
		stream << "( ";
		if( row == nullptr ) {
			stream << "empty";
		} else {
			stream << row[0];
			for( int colIndex = 1; colIndex < matrix.GetWidth(); ++colIndex ) {
				stream << ", ";
				stream << row[colIndex];
			}
		}
		stream << " )";
		stream << '\n';
	}
	return stream;
}

inline CArchive& operator << ( CArchive& archive, const CFloatMatrix& matrix )
{
	NeoPresume( archive.IsStoring() );
	const_cast<CFloatMatrix&>( matrix ).Serialize( archive );
	return archive;
}

inline CArchive& operator >> ( CArchive& archive, CFloatMatrix& matrix )
{
	NeoPresume( archive.IsLoading() );
	matrix.Serialize( archive );
	return archive;
}

} // namespace NeoML
