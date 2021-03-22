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

#include <NeoML/TraditionalML/SparseFloatVector.h>

namespace NeoML {

// A matrix descriptor
// If Columns field is not set assume dense representation, otherwise assume sparse.
// Note that PointerB and PointerE must be defined for both! 
struct NEOML_API CFloatMatrixDesc {
	int Height; // the matrix height
	int Width; // the matrix width
	int* Columns; // the columns array
	float* Values; // the values array
	size_t* PointerB; // the array of indices for vector start in Columns/Values
	size_t* PointerE; // the array of indices for vector end in Columns/Values

	CFloatMatrixDesc() : Height(0), Width(0), Columns(nullptr), Values(nullptr), PointerB(nullptr), PointerE(nullptr) {}

	// Retrieves the descriptor of a row (as a sparse vector)
	void GetRow( int index, CFloatVectorDesc& desc ) const;
	CFloatVectorDesc GetRow( int index ) const;

	static CFloatMatrixDesc Empty; // the empty matrix descriptor
};

// DEPRECATED: for compatibility
typedef CFloatMatrixDesc CSparseFloatMatrixDesc;

inline void CFloatMatrixDesc::GetRow( int index, CFloatVectorDesc& desc ) const
{
	NeoAssert( 0 <= index && index < Height );
	desc.Size = static_cast<int>( PointerE[index] - PointerB[index] );
	desc.Values = Values + PointerB[index];
	if( Columns == nullptr ) { // dense representation
		NeoPresume( desc.Size == Width );
		desc.Indexes = nullptr;
	} else {
		desc.Indexes = Columns + PointerB[index];
	}
}

inline CFloatVectorDesc CFloatMatrixDesc::GetRow( int index ) const
{
	CFloatVectorDesc res;
	GetRow( index, res );
	return res;
}

//---------------------------------------------------------------------------------------------------------

// A sparse matrix
// Any value that is not specified is 0
class NEOML_API CSparseFloatMatrix {
	static const size_t InitialRowBufferSize = 32;
	static const size_t InitialElementBufferSize = 512;
public:
	CSparseFloatMatrix() {}
	explicit CSparseFloatMatrix( int width, size_t rowsBufferSize = 0, size_t elementsBufferSize = 0 );
	explicit CSparseFloatMatrix( const CFloatMatrixDesc& desc );
	CSparseFloatMatrix( const CSparseFloatMatrix& other );

	CFloatMatrixDesc* CopyOnWrite() { return body == 0 ? 0 : &body.CopyOnWrite()->Desc; }
	const CFloatMatrixDesc& GetDesc() const { return body == 0 ? CFloatMatrixDesc::Empty : body->Desc; }

	int GetHeight() const { return body == 0 ? 0 : body->Desc.Height; }
	int GetWidth() const { return body == 0 ? 0 : body->Desc.Width; }

	void GrowInRows( size_t newRowsBufferSize );
	void GrowInElements( size_t newElementsBufferSize );

	void AddRow( const CSparseFloatVector& row );
	void AddRow( const CFloatVectorDesc& row );
	CFloatVectorDesc GetRow( int index ) const;
	void GetRow( int index, CFloatVectorDesc& desc ) const;

	CSparseFloatMatrix& operator = ( const CSparseFloatMatrix& matrix );

	void Serialize( CArchive& archive );

private:
	// The matrix body, that is, the object that stores all its data
	struct NEOML_API CSparseFloatMatrixBody : public IObject {
		size_t RowsBufferSize;
		size_t ElementsBufferSize;
		size_t ElementCount;
		CFloatMatrixDesc Desc;

		CSparseFloatMatrixBody( int height, int width, size_t elementCount, size_t rowsBufferSize, size_t elementsBufferSize );
		explicit CSparseFloatMatrixBody( const CFloatMatrixDesc& desc );
		~CSparseFloatMatrixBody();

		CSparseFloatMatrixBody* Duplicate() const;
		void CopyDataTo( const CSparseFloatMatrixBody* dst ) const;
	};
 
	CCopyOnWritePtr<CSparseFloatMatrixBody> body; // The matrix body.
};

// Writing into a CTextStream
inline CTextStream& operator<<( CTextStream& stream, const CSparseFloatMatrix& matrix )
{
	for( int i = 0; i < matrix.GetHeight(); i++ ) {
		CFloatVectorDesc desc;
		matrix.GetRow( i, desc );
		stream << "( ";
		if( desc.Size == 0 ) {
			stream << "empty";
		} else {
			stream << desc.Indexes[0] << ": " << desc.Values[0];
			for( int j = 1; j < desc.Size; j++ ) {
				stream << ", ";
				stream << desc.Indexes[j] << ": " << desc.Values[j];
			}
		}
		stream << " )";
		stream << "\n";
	}
	return stream;
}

inline CArchive& operator << ( CArchive& archive, const CSparseFloatMatrix& matrix )
{
	NeoPresume( archive.IsStoring() );
	const_cast<CSparseFloatMatrix&>(matrix).Serialize( archive );
	return archive;
}

inline CArchive& operator >> ( CArchive& archive, CSparseFloatMatrix& matrix )
{
	NeoPresume( archive.IsLoading() );
	matrix.Serialize( archive );
	return archive;
}

} // namespace NeoML
