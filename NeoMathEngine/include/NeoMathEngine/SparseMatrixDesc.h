/* Copyright © 2017-2024 ABBYY

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

// Sparse matrix descriptor (CSR format)
struct CSparseMatrixDesc final {
	// The number of elements
	const int ElementCount;
	// This array of (matrix height + 1) length stores the indices where the new row starts, as (Columns + Values)
	// The last array element is equal to the number of elements in the matrix
	const CIntHandle Rows;
	// This array of ElementCount length contains the indices of the columns to which the corresponding values belong
	const CIntHandle Columns;
	// This array of ElementCount length contains the matrix elements' values
	const CFloatHandle Values;

	CSparseMatrixDesc( int count, const CIntHandle& rows, const CIntHandle& columns, const CFloatHandle& values ) :
		ElementCount( count ), Rows( rows ), Columns( columns ), Values( values ) {}
};

} // namespace NeoML
