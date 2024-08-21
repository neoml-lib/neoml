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

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/DnnSparseMatrix.h>
#include <NeoML/Dnn/Layers/BackLinkLayer.h>
#include <NeoML/Dnn/Layers/CompositeLayer.h>

namespace NeoML {

CDnnSparseMatrix::CDnnSparseMatrix( IMathEngine& _mathEngine, int _rowCount, int _columnCount ) :
	mathEngine( _mathEngine ),
	rowCount( _rowCount ),
	columnCount( _columnCount )
{
	NeoAssert( rowCount > 0 );
	NeoAssert( columnCount > 0 );
}

CDnnSparseMatrix::~CDnnSparseMatrix()
{
	if( !mathEngineData.IsNull() ) {
		mathEngine.HeapFree( mathEngineData );
	}
}

void CDnnSparseMatrix::Create( const IProblem* problem, int startVectorIndex, int batchCount )
{
	NeoAssert( problem != 0 );
	NeoAssert( startVectorIndex >= 0 );
	NeoAssert( batchCount > 0 );

	const int problemVectorCount = problem->GetVectorCount();
	vectors.SetBufferSize( rowCount * batchCount );
	matrixes.SetBufferSize( batchCount );

	CFloatMatrixDesc problemMatrix = problem->GetMatrix();
	totalElementSize = 0;
	totalRowSize = 0;
	for( int i = 0; i < rowCount * batchCount; i++ ) {
		vectors.Add( problemMatrix.GetRow( ( startVectorIndex + i ) % problemVectorCount ) );
		
		if( i % rowCount == 0 ) {
			// The batches should be aligned
			totalElementSize = CeilTo( totalElementSize, 4 );
			if( i > 0 ) {
				totalRowSize++;
			}
			totalRowSize = CeilTo( totalRowSize, 4 );
			matrixes.Add( CMatrix( /*elementCount*/0, totalRowSize, totalElementSize ) );
		}
		matrixes.Last().ElementCount += vectors.Last().Size;
		totalElementSize += vectors.Last().Size;
		totalRowSize++;
	}
	totalElementSize = CeilTo( totalElementSize, 4 );
	totalRowSize++;
	totalRowSize = CeilTo( totalRowSize, 4 );

	static_assert( sizeof( int ) == sizeof( float ), "sizeof( int ) != sizeof( float )" );
	CArray<int> data;
	data.SetSize( totalRowSize + 2 * totalElementSize );

	int* rowsPtr = data.GetPtr();
	int* columnsPtr = data.GetPtr() + totalRowSize;
	float* valuesPtr = reinterpret_cast<float*>( data.GetPtr() ) + totalRowSize + totalElementSize;

	int rowIndex = 0;
	int elementIndex = 0;
	for( int i = 0; i < vectors.Size(); i++ ) {
		if( i % rowCount == 0 ) {
			// The batches should be aligned
			if( i > 0 ) {
				rowsPtr[rowIndex] = elementIndex;
				rowIndex++;
			}
			rowIndex = CeilTo( rowIndex, 4 );
			columnsPtr = columnsPtr + CeilTo( elementIndex, 4 );
			valuesPtr = valuesPtr + CeilTo( elementIndex, 4 );
			elementIndex = 0;
		}
		rowsPtr[rowIndex] = elementIndex;

		const int vectorElementCount = vectors[i].Size;
		for( int j = 0; j < vectorElementCount; j++ ) {
			columnsPtr[elementIndex] = vectors[i].Indexes[j];
			valuesPtr[elementIndex] = vectors[i].Values[j];
			elementIndex++;
		}
		rowIndex++;
	}
	rowsPtr[rowIndex] = elementIndex;

	if( mathEngineDataSize < data.Size() * sizeof( int ) ) {
		if( !mathEngineData.IsNull() ) {
			mathEngine.HeapFree( mathEngineData );
			mathEngineData = CIntHandle();
			mathEngineDataSize = 0;
		}
		mathEngineData = mathEngine.HeapAllocTyped<int>( data.Size() );
		mathEngineDataSize = data.Size() * sizeof( int );
	}

	mathEngine.DataExchangeTyped( mathEngineData, data.GetPtr(), data.Size() );
}

void CDnnSparseMatrix::Destroy()
{
	if( !vectors.IsEmpty() ) {
		vectors.DeleteAll();
		matrixes.DeleteAll();
	}
}

CSparseMatrixDesc CDnnSparseMatrix::GetBatchDesc( int index ) const
{
	NeoAssert( index >= 0 );
	NeoAssert( index < matrixes.Size() );
	NeoAssert( !vectors.IsEmpty() );
	NeoAssert( !matrixes.IsEmpty() );

	return CSparseMatrixDesc( matrixes[index].ElementCount,
		mathEngineData + matrixes[index].RowPos,
		mathEngineData + totalRowSize + matrixes[index].ElementPos,
		CFloatHandle( mathEngineData ) + totalRowSize + totalElementSize + matrixes[index].ElementPos );
}

} // namespace NeoML
