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
#include <NeoMathEngine/MemoryHandle.h>
#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoML/TraditionalML/Problem.h>

namespace NeoML {

// A sparse matrix for IMathEngine
class NEOML_API CDnnSparseMatrix {
public:
	CDnnSparseMatrix( IMathEngine& _mathEngine, int rowCount, int columnCount );
	~CDnnSparseMatrix();

	// Creates a matrix from the specified problem
	void Create( const IProblem* problem, int startVectorIndex, int batchCount );

	// Destroys the matrix; the allocated memory is kept
	void Destroy();

	// Retrieves the number of batches in the matrix
	int GetBatchCount() const { return matrixes.Size(); }

	// Retrieves the batch with the specified index
	CSparseMatrixDesc GetBatchDesc( int index ) const;

private:
	struct CMatrix {
		int ElementCount;
		int RowPos;
		int ElementPos;
	};

	IMathEngine& mathEngine; // the math engine
	const int rowCount; // the number of rows
	const int columnCount; // the number of columns
	CArray<CSparseFloatVectorDesc> vectors; // the vectors (matrix columns)
	CArray<CMatrix> matrixes; // the matrix descriptions
	int totalElementSize; // the size of matrix elements (aligned)
	int totalRowSize; // the size of matrix rows (aligned)
	CIntHandle mathEngineData; // the data allocated for the matrix by IMathEngine
	size_t mathEngineDataSize; // the size of the data allocated by IMathEngine

	CDnnSparseMatrix( const CDnnSparseMatrix& );
	CDnnSparseMatrix& operator=( const CDnnSparseMatrix& );
};

} // namespace NeoML
