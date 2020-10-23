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

// Sparse matrix descriptor
struct NEOML_API CSparseFloatMatrixDesc {
	int Height; // the matrix height
	int Width; // the matrix width
	int* Columns; // the columns array
	float* Values; // the values array
	int* PointerB; // the array of indices for vector start in Columns/Values
	int* PointerE; // the array of indices for vector end in Columns/Values

	CSparseFloatMatrixDesc() : Height(0), Width(0), Columns(0), Values(0), PointerB(0), PointerE(0) {}

	// Retrieves the descriptor of a row (as a sparse vector)
	void GetRow( int index, CSparseFloatVectorDesc& desc ) const;
	CSparseFloatVectorDesc GetRow( int index ) const;

	static CSparseFloatMatrixDesc Empty; // the empty matrix descriptor
};

inline void CSparseFloatMatrixDesc::GetRow( int index, CSparseFloatVectorDesc& desc ) const
{
	NeoAssert( 0 <= index && index < Height );
	desc.Size = PointerE[index] - PointerB[index];
	desc.Indexes = Columns + PointerB[index];
	desc.Values = Values + PointerB[index];
}

inline CSparseFloatVectorDesc CSparseFloatMatrixDesc::GetRow( int index ) const
{
	CSparseFloatVectorDesc res;
	GetRow( index, res );
	return res;
}

//---------------------------------------------------------------------------------------------------------

// A sparse matrix
// Any value that is not specified is 0
class NEOML_API CSparseFloatMatrix {
	static const int InitialRowBufferSize = 32;
	static const int InitialElementBufferSize = 512;
public:
	CSparseFloatMatrix() {}
	explicit CSparseFloatMatrix( int width, int rowsBufferSize = 0, int elementsBufferSize = 0 );
	explicit CSparseFloatMatrix( const CSparseFloatMatrixDesc& desc );
	CSparseFloatMatrix( const CSparseFloatMatrix& other );

	CSparseFloatMatrixDesc* CopyOnWrite() { return body == 0 ? 0 : &body.CopyOnWrite()->Desc; }
	const CSparseFloatMatrixDesc& GetDesc() const { return body == 0 ? CSparseFloatMatrixDesc::Empty : body->Desc; }

	int GetHeight() const { return body == 0 ? 0 : body->Desc.Height; }
	int GetWidth() const { return body == 0 ? 0 : body->Desc.Width; }

	void GrowInRows( int newRowsBufferSize );
	void GrowInElements( int newElementsBufferSize );

	void AddRow( const CSparseFloatVector& row );
	void AddRow( const CSparseFloatVectorDesc& row );
	CSparseFloatVectorDesc GetRow( int index ) const;
	void GetRow( int index, CSparseFloatVectorDesc& desc ) const;

	CSparseFloatMatrix& operator = ( const CSparseFloatMatrix& vector );

	void Serialize( CArchive& archive );

private:
	// The matrix body, that is, the object that stores all its data
	struct NEOML_API CSparseFloatMatrixBody : public IObject {
		int RowsBufferSize;
		int ElementsBufferSize;
		int ElementCount;
		CSparseFloatMatrixDesc Desc;

		CSparseFloatMatrixBody( int height, int width, int elementCount, int rowsBufferSize, int elementsBufferSize );
		explicit CSparseFloatMatrixBody( const CSparseFloatMatrixDesc& desc );
		virtual ~CSparseFloatMatrixBody();

		CSparseFloatMatrixBody* Duplicate() const;
	};
 
	CCopyOnWritePtr<CSparseFloatMatrixBody> body; // The matrix body.
};

// Writing into a CTextStream
inline CTextStream& operator<<( CTextStream& stream, const CSparseFloatMatrix& matrix )
{
	for( int i = 0; i < matrix.GetHeight(); i++ ) {
		CSparseFloatVectorDesc desc;
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
