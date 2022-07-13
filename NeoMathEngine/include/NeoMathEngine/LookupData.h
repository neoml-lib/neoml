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

#include <NeoMathEngine/MemoryHandle.h>

namespace NeoML {

// The size of a vector representation table for lookup
struct CLookupDimension {
	int VectorCount;
	int VectorSize;
	CLookupDimension() : VectorCount( 0 ), VectorSize( 0 ) {}
	CLookupDimension( int count, int size ) : VectorCount( count ), VectorSize( size ) {}
};

// The structure stores a vector representation for lookup (does NOT own the memory, only refers to it)
// It may have a set of several vectors working from the same table; in this case, the size of Vector = BatchSize
struct CLookupVector {
	CLookupDimension Dims; // lookup table size
	CConstFloatHandle Table; // lookup table
	CConstIntHandle Vector; // the vector index in the lookup table

	int VectorSize() const { return Dims.VectorSize; }

	CLookupVector() {}

	CLookupVector( const CConstFloatHandle& table, int vectorCount, int vectorSize,
		const CConstIntHandle& vector = CConstIntHandle() ) : Table( table ), Vector( vector )
	{
		Dims.VectorCount = vectorCount;
		Dims.VectorSize = vectorSize;
	}
};

// The structure stores a matrix for lookup (does NOT own the memory, only refers to it)
// The matrix rows may be "duplicated"
// It may have a set of several matrices working from the same table; in this case, the size of Rows = RowCount * BatchSize
struct CLookupMatrix {
	CLookupDimension Dims; // lookup table size
	CConstFloatHandle Table; // lookup table
	CConstIntHandle Rows; // the indices of matrix rows in the lookup table
	int RowCount; // the matrix height (the number of rows)

	int Height() const { return RowCount; }
	int Width() const { return Dims.VectorSize; }

	CLookupMatrix() : RowCount( 0 ) {}

	CLookupMatrix( const CConstFloatHandle& table, int vectorCount, int vectorSize,
		const CConstIntHandle& rows = CConstIntHandle(), int rowCount = 0 ) :
		Table( table ), Rows( rows ), RowCount( rowCount )
	{
		Dims.VectorCount = vectorCount;
		Dims.VectorSize = vectorSize;
	}
};

} // namespace NeoML
