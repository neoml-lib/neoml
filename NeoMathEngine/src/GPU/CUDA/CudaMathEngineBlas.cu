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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_CUDA

#include <CudaMathEngine.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>
#include <CudaDevice.h>

#include <Kernels/CudaBlasKernels.h>

namespace NeoML {

void CCudaMathEngine::SetVectorToMatrixRows(const CFloatHandle& resultHandle,
	int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle)
{
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, matrixHeight * matrixWidth );

	SetVectorToMatrixRowsKernel<<<blockCount, threadCount>>>
		(GetRaw(resultHandle), matrixHeight, matrixWidth, GetRaw(vectorHandle));
}

void CCudaMathEngine::AddVectorToMatrixElements(const CFloatHandle& matrix, int height, int width,
	const CConstIntHandle& indices, const CConstFloatHandle& vector)
{
	ASSERT_EXPR( matrix.GetMathEngine() == this );
	ASSERT_EXPR( indices.GetMathEngine() == this );
	ASSERT_EXPR( vector.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, height, AddVectorToMatrixElementsCombine);

	AddVectorToMatrixElementsKernel<<<blockCount, threadCount>>>(GetRaw(matrix),
		height, width, GetRaw(indices), GetRaw(vector));
}

void CCudaMathEngine::AddVectorToMatrixElements(const CFloatHandle& matrixHandle, int height, int width,
	const CConstIntHandle& rowIndicesHandle, const CConstIntHandle& columnIndicesHandle,
	const CConstFloatHandle& vectorHandle, int vectorSize)
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( rowIndicesHandle.GetMathEngine() == this );
	ASSERT_EXPR( columnIndicesHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, AddVectorToMatrixElementsMulCombine);

	AddVectorToMatrixElementsKernel<<<blockCount, threadCount>>>(GetRaw(matrixHandle), height, width,
		GetRaw(rowIndicesHandle), GetRaw(columnIndicesHandle), GetRaw(vectorHandle), vectorSize);
}

// Assigns the values: matrix[rowIndices[i], columnIndices[i]] = vector[i].
void CCudaMathEngine::setVectorToMatrixElements(
	const CFloatHandle& matrixHandle, int height, int width,
	const CConstIntHandle& rowIndicesHandle, const CConstIntHandle& columnIndicesHandle,
	const CConstFloatHandle& vectorHandle, int vectorSize )
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( rowIndicesHandle.GetMathEngine() == this );
	ASSERT_EXPR( columnIndicesHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(
		blockCount, threadCount, vectorSize, SetVectorToMatrixElementsMulCombine );

	SetVectorToMatrixElementsKernel<<<blockCount, threadCount>>>(
		GetRaw( matrixHandle ), height, width,
		GetRaw( rowIndicesHandle ), GetRaw( columnIndicesHandle ),
		GetRaw( vectorHandle ), vectorSize );
}

void CCudaMathEngine::AddMatrixElementsToVector(const CConstFloatHandle& matrix, int height, int width,
	const CConstIntHandle& indices, const CFloatHandle& result, int vectorSize)
{
	ASSERT_EXPR( matrix.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );
	ASSERT_EXPR( indices.GetMathEngine() == this );
	ASSERT_EXPR(vectorSize >= height);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, height, AddMatrixElementsToVectorCombine);

	AddMatrixElementsToVectorKernel<<<blockCount, threadCount>>>(GetRaw(matrix),
		height, width, GetRaw(indices), GetRaw(result));
}

void CCudaMathEngine::AddMatrixElementsToVector(const CConstFloatHandle& matrix, int height, int width,
	const CConstIntHandle& rowIndices, const CConstIntHandle& columnIndices,
	const CFloatHandle& result, int vectorSize)
{
	ASSERT_EXPR( matrix.GetMathEngine() == this );
	ASSERT_EXPR( rowIndices.GetMathEngine() == this );
	ASSERT_EXPR( columnIndices.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, AddMatrixElementsToVectorMulCombine);

	AddMatrixElementsToVectorKernel<<<blockCount, threadCount>>>(GetRaw(matrix),
		height, width, GetRaw(rowIndices), GetRaw(columnIndices), GetRaw(result), vectorSize);
}

void CCudaMathEngine::AddMatrixElementsToMatrix(const CConstFloatHandle& matrix, int height, int width,
	const CFloatHandle& result, const CConstIntHandle& indices)
{
	ASSERT_EXPR( matrix.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );
	ASSERT_EXPR( indices.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, height, AddMatrixElementsToMatrixCombine);

	AddMatrixElementsToMatrixKernel<<<blockCount, threadCount>>>(GetRaw(matrix),
		height, width, GetRaw(result), GetRaw(indices));
}

void CCudaMathEngine::AddDiagMatrixToMatrix( const CConstFloatHandle& diagMatrix, const CConstFloatHandle& matrix,
	int height, int width, const CFloatHandle& result )
{
	ASSERT_EXPR( matrix.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );
	ASSERT_EXPR( diagMatrix.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const int widthNorm = ( width + AddDiagMatrixToMatrixCombine - 1 ) / AddDiagMatrixToMatrixCombine;
	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2D( blockCount, threadCount, height, widthNorm );

	AddDiagMatrixToMatrixKernel<<<blockCount, threadCount>>>( GetRaw( diagMatrix ), GetRaw( matrix ),
		height, width, widthNorm, GetRaw( result ) );
}

void CCudaMathEngine::AddVectorToMatrixRows(int batchSize,
	const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle, int matrixHeight,
	int matrixWidth, const CConstFloatHandle& vectorHandle)
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int widthNorm = (matrixWidth + BatchAddVectorToMatrixRowsCombine - 1) /
		BatchAddVectorToMatrixRowsCombine;
	widthNorm = alignXSizeForWarp(widthNorm);

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2DMinYX(1, device->ThreadMax3DCountX, blockCount, threadCount, batchSize * matrixHeight, widthNorm);

	AddVectorToMatrixRowsKernel<<<blockCount, threadCount>>>(batchSize,
		GetRaw(matrixHandle), GetRaw(resultHandle), matrixHeight,
		matrixWidth, GetRaw(vectorHandle));
}

void CCudaMathEngine::AddVectorToMatrixColumns( const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
	int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle )
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2D( blockCount, threadCount, matrixHeight, matrixWidth );

	AddVectorToMatrixColumnsKernel<<<blockCount, threadCount>>>
		( GetRaw(matrixHandle), GetRaw(resultHandle), matrixHeight, matrixWidth, GetRaw(vectorHandle) );
}

void CCudaMathEngine::AddVectorToMatrixColumns( const CConstIntHandle& matrixHandle, const CIntHandle& resultHandle,
	int matrixHeight, int matrixWidth, const CConstIntHandle& vectorHandle )
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2D( blockCount, threadCount, matrixHeight, matrixWidth );

	AddVectorToMatrixColumnsKernel<<<blockCount, threadCount>>>
		( GetRaw(matrixHandle), GetRaw(resultHandle), matrixHeight, matrixWidth, GetRaw(vectorHandle) );
}

void CCudaMathEngine::SubVectorFromMatrixColumns(const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
	int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle)
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2D(blockCount, threadCount, matrixHeight, matrixWidth);

	SubVectorFromMatrixColumnsKernel<<<blockCount, threadCount>>>
		(GetRaw(matrixHandle), GetRaw(resultHandle), matrixHeight, matrixWidth, GetRaw(vectorHandle));
}

void CCudaMathEngine::SumMatrixRows( int batchSize, const CIntHandle& resultHandle,
	const CConstIntHandle& matrixHandle, int matrixHeight, int matrixWidth )
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	VectorFill( resultHandle, 0, batchSize * matrixWidth );

	const int height = ( matrixHeight + SumMatrixRowsAddCombineCount - 1 ) / SumMatrixRowsAddCombineCount;

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid3D( blockCount, threadCount, batchSize, height, matrixWidth );

	SumMatrixRowsAddKernel<<<blockCount, threadCount>>>
		( batchSize, GetRaw(resultHandle), GetRaw(matrixHandle), matrixHeight, matrixWidth );
}

void CCudaMathEngine::SumMatrixRowsAdd( 
	int batchSize, const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
	int matrixHeight, int matrixWidth )
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const int height = ( matrixHeight + SumMatrixRowsAddCombineCount - 1 ) / SumMatrixRowsAddCombineCount;

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid3D( blockCount, threadCount, batchSize, height, matrixWidth );

	SumMatrixRowsAddKernel<<<blockCount, threadCount>>>
		( batchSize, GetRaw(resultHandle), GetRaw(matrixHandle), matrixHeight, matrixWidth );
}

void CCudaMathEngine::SumMatrixColumns(const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
	int matrixHeight, int matrixWidth)
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	sumMatrixColumnsKernelFunc(resultHandle, GetRaw(matrixHandle), matrixHeight, matrixWidth, false);
}

void CCudaMathEngine::MatrixColumnsEltwiseDivide( const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth,
	const CConstFloatHandle& vectorHandle, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const int widthNorm = ( matrixWidth + MatrixColumnsEltwiseDivideCombine - 1 ) / MatrixColumnsEltwiseDivideCombine;
	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2D( blockCount, threadCount, matrixHeight, widthNorm );

	MatrixColumnsEltwiseDivideKernel<<<blockCount, threadCount>>>( GetRaw( matrixHandle ),
		matrixHeight, matrixWidth, widthNorm, GetRaw( vectorHandle ), GetRaw( resultHandle ) );
}

void CCudaMathEngine::MatrixLogSumExpByRows(const CConstFloatHandle& matrix, int height, int width,
	const CFloatHandle& result, int resultSize)
{
	ASSERT_EXPR( matrix.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );
	ASSERT_EXPR(resultSize >= height);
	SetCudaDevice( device->DeviceNumber );

	int widthNorm = (width + MatrixLogSumExpByRowsCombine - 1) / MatrixLogSumExpByRowsCombine;
	widthNorm = alignXSizeForWarp(widthNorm);

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2DMinYX(1, 1024, blockCount, threadCount, height, widthNorm);
	blockCount.x = 1;
	const int sharedSize = threadCount.x * threadCount.y * sizeof(float);
	MatrixLogSumExpByRowsKernel<<<blockCount, threadCount, sharedSize>>>(GetRaw(matrix), height, width,
		GetRaw(result), widthNorm);
}

void CCudaMathEngine::MatrixSoftmaxByRows(const CConstFloatHandle& matrix, int height, int width,
	const CFloatHandle& result)
{
	ASSERT_EXPR( matrix.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int widthNorm = (width + MatrixSoftmaxByRowsCombine - 1) / MatrixSoftmaxByRowsCombine;
	widthNorm = alignXSizeForWarp(widthNorm);

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2DMinYX(1, 1024, blockCount, threadCount, height, widthNorm);
	blockCount.x = 1;

	const int sharedSize = threadCount.x * threadCount.y * sizeof(float);
	MatrixSoftmaxByRowsKernel<<<blockCount, threadCount, sharedSize>>>(GetRaw(matrix), height, width,
		GetRaw(result), widthNorm);
}

void CCudaMathEngine::MatrixSoftmaxDiffOpByRows(const CConstFloatHandle& first, const CConstFloatHandle& second,
	int height, int width, const CFloatHandle& result)
{
	ASSERT_EXPR( first.GetMathEngine() == this );
	ASSERT_EXPR( second.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int widthNorm = (width + MatrixSoftmaxDiffOpByRowsCombine - 1) / MatrixSoftmaxDiffOpByRowsCombine;
	widthNorm = alignXSizeForWarp(widthNorm);

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2DMinYX(1, 1024, blockCount, threadCount, height, widthNorm);
	blockCount.x = 1;

	const int sharedSize = threadCount.x * threadCount.y * sizeof(float);
	MatrixSoftmaxDiffOpByRowsKernel<<<blockCount, threadCount, sharedSize>>>(GetRaw(first), GetRaw(second),
		height, width, GetRaw(result), widthNorm);
}

void CCudaMathEngine::MatrixSoftmaxByColumns(const CConstFloatHandle& matrix, int height, int width,
	const CFloatHandle& result)
{
	ASSERT_EXPR( matrix.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int heightNorm = (height + MatrixSoftmaxByColumnsCombine - 1) / MatrixSoftmaxByColumnsCombine;
	heightNorm = alignXSizeForWarp(heightNorm);

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2DMinYX(1, 1024, blockCount, threadCount, width, heightNorm);
	blockCount.x = 1;

	const int sharedSize = threadCount.x * threadCount.y * sizeof(float);
	MatrixSoftmaxByColumnsKernel<<<blockCount, threadCount, sharedSize>>>(GetRaw(matrix), height, width,
		GetRaw(result), heightNorm);
}

void CCudaMathEngine::MatrixSoftmaxDiffOpByColumns(const CConstFloatHandle& first, const CConstFloatHandle& second,
	int height, int width, const CFloatHandle& result)
{
	ASSERT_EXPR( first.GetMathEngine() == this );
	ASSERT_EXPR( second.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int heightNorm = (height + MatrixSoftmaxDiffOpByColumnsCombine - 1) / MatrixSoftmaxDiffOpByColumnsCombine;
	heightNorm = alignXSizeForWarp(heightNorm);

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2DMinYX(1, 1024, blockCount, threadCount, width, heightNorm);
	blockCount.x = 1;

	const int sharedSize = threadCount.x * threadCount.y * sizeof(float);
	MatrixSoftmaxDiffOpByColumnsKernel<<<blockCount, threadCount, sharedSize>>>(
		GetRaw(first), GetRaw(second), height, width, GetRaw(result), heightNorm);
}

void CCudaMathEngine::FindMaxValueInRows(const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth,
	const CFloatHandle& resultHandle, const CIntHandle& columnIndices, int vectorSize)
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR(vectorSize >= matrixHeight);
	SetCudaDevice( device->DeviceNumber );

	int widthNorm = (matrixWidth + FindMaxValueInRowsCombine - 1) / FindMaxValueInRowsCombine;
	widthNorm = alignXSizeForWarp(widthNorm);

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2DMinYX(1, 1024, blockCount, threadCount, matrixHeight, widthNorm);
	blockCount.x = 1;

	const int sharedSize = threadCount.y * threadCount.x * sizeof(CValueWithIndex);
	FindMaxValueWithIndicesInRowsKernel<<<blockCount, threadCount, sharedSize>>>(
		GetRaw(matrixHandle), matrixHeight, matrixWidth, GetRaw(resultHandle), GetRaw(columnIndices), widthNorm);
}

void CCudaMathEngine::FindMaxValueInRows(const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR(vectorSize >= matrixHeight);
	SetCudaDevice( device->DeviceNumber );

	int widthNorm = (matrixWidth + FindMaxValueInRowsCombine - 1) / FindMaxValueInRowsCombine;
	widthNorm = alignXSizeForWarp(widthNorm);

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2DMinYX(1, 1024, blockCount, threadCount, matrixHeight, widthNorm);
	blockCount.x = 1;

	const int sharedSize = threadCount.y * threadCount.x * sizeof(float);
	FindMaxValueInRowsKernel<<<blockCount, threadCount, sharedSize>>>(GetRaw(matrixHandle), matrixHeight,
		matrixWidth, GetRaw(resultHandle), widthNorm);
}

void CCudaMathEngine::FindMaxValueInColumns( int batchSize, const CConstFloatHandle& matrixHandle, int matrixHeight,
	int matrixWidth, const CFloatHandle& resultHandle, const CIntHandle& rowIndices, int vectorSize )
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( rowIndices.GetMathEngine() == this );
	ASSERT_EXPR( vectorSize >= batchSize * matrixWidth );
	SetCudaDevice( device->DeviceNumber );

	int heightNorm = ( matrixHeight + FindMaxValueInColumnsCombine - 1 ) / FindMaxValueInColumnsCombine;
	heightNorm = alignXSizeForWarp( heightNorm );

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid3D( blockCount, threadCount, batchSize, matrixWidth, heightNorm );
	blockCount.x = 1;

	const int sharedSize = threadCount.z * threadCount.y * threadCount.x * sizeof( CValueWithIndex );
	FindMaxValueInColumnsKernel<<<blockCount, threadCount, sharedSize>>>( batchSize,
		GetRaw( matrixHandle ), matrixHeight, matrixWidth, GetRaw( resultHandle ), GetRaw( rowIndices ),
		heightNorm );
}

void CCudaMathEngine::FindMinValueInColumns( const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth,
	const CFloatHandle& resultHandle, const CIntHandle& columnIndices )
{
	SetCudaDevice( device->DeviceNumber );
	// Initialize using the first row data
	VectorCopy( resultHandle, matrixHandle, matrixWidth );
	VectorFill( columnIndices, 0, matrixWidth );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, matrixWidth );

	FindMinValueInColumnsKernel<<<blockCount, threadCount>>>( GetRaw( matrixHandle ), matrixHeight,
		matrixWidth, GetRaw( resultHandle ), GetRaw( columnIndices ) );
}

void CCudaMathEngine::VectorMultichannelLookupAndCopy(int batchSize, int channelCount, const CConstFloatHandle& inputHandle,
	const CConstFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount, 
	const CFloatHandle& outputHandle, int outputChannelsCount)
{
	ASSERT_EXPR( inputHandle.GetMathEngine() == this );
	ASSERT_EXPR( outputHandle.GetMathEngine() == this );

	vectorMultichannelLookupAndCopy(batchSize, channelCount, inputHandle,
		lookupHandles, lookupDimensions, lookupCount, outputHandle, outputChannelsCount);
}

void CCudaMathEngine::VectorMultichannelLookupAndCopy(int batchSize, int channelCount, const CConstIntHandle& inputHandle,
	const CConstFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
	const CFloatHandle& outputHandle, int outputChannelsCount)
{
	ASSERT_EXPR( inputHandle.GetMathEngine() == this );
	ASSERT_EXPR( outputHandle.GetMathEngine() == this );

	vectorMultichannelLookupAndCopy(batchSize, channelCount, inputHandle,
		lookupHandles, lookupDimensions, lookupCount, outputHandle, outputChannelsCount);
}

void CCudaMathEngine::VectorMultichannelLookupAndCopy(int batchSize, int channelCount, const CConstIntHandle& inputHandle,
	const CConstIntHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
	const CIntHandle& outputHandle, int outputChannelsCount)
{
	ASSERT_EXPR( inputHandle.GetMathEngine() == this );
	ASSERT_EXPR( outputHandle.GetMathEngine() == this );

	vectorMultichannelLookupAndCopy(batchSize, channelCount, inputHandle,
		lookupHandles, lookupDimensions, lookupCount, outputHandle, outputChannelsCount);
}

void CCudaMathEngine::VectorMultichannelLookupAndAddToTable(int batchSize, int channelCount, const CConstFloatHandle& inputHandle,
	const CFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
	const CConstFloatHandle& multHandle,
	const CConstFloatHandle& matrixHandle, int outputChannelsCount)
{
	ASSERT_EXPR( inputHandle.GetMathEngine() == this );
	ASSERT_EXPR( multHandle.GetMathEngine() == this );
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );

	vectorMultichannelLookupAndAddToTable(batchSize, channelCount, inputHandle,
		lookupHandles, lookupDimensions, lookupCount, multHandle, matrixHandle, outputChannelsCount);
}

void CCudaMathEngine::VectorMultichannelLookupAndAddToTable(int batchSize, int channelCount, const CConstIntHandle& inputHandle,
	const CFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
	const CConstFloatHandle& multHandle,
	const CConstFloatHandle& matrixHandle, int outputChannelsCount)
{
	ASSERT_EXPR( inputHandle.GetMathEngine() == this );
	ASSERT_EXPR( multHandle.GetMathEngine() == this );
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );

	vectorMultichannelLookupAndAddToTable(batchSize, channelCount, inputHandle,
		lookupHandles, lookupDimensions, lookupCount, multHandle, matrixHandle, outputChannelsCount);
}

void CCudaMathEngine::BitSetBinarization(int batchSize, int bitSetSize,
	const CConstIntHandle& inputHandle, int outputVectorSize, const CFloatHandle& resultHandle)
{
	ASSERT_EXPR( inputHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, batchSize * outputVectorSize );

	BitSetBinarizationKernel<<<blockCount, threadCount>>>(batchSize, bitSetSize,
		GetRaw(inputHandle), outputVectorSize, GetRaw(resultHandle));
}

void CCudaMathEngine::MultiplyLookupMatrixByLookupVector(int batchSize, const CLookupMatrix& matrix,
	const CLookupVector& vector, const CFloatHandle& resultHandle, int resultSize)
{
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	ASSERT_EXPR(matrix.Width() == vector.VectorSize());
	ASSERT_EXPR(resultSize >= batchSize * matrix.Height());

	int widthNorm = (matrix.Width() + MultiplyLookupMatrixByLookupVectorCombine - 1) /
		MultiplyLookupMatrixByLookupVectorCombine;
	widthNorm = alignXSizeForWarp(widthNorm);

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2DMinYX(1, 1024, blockCount, threadCount, batchSize * matrix.Height(), widthNorm);

	if(blockCount.x > 0) {
		// Several GPUs may take part in adding up one row, need atomic operations
		// Set resultHandle to zeros
		VectorFill(resultHandle, 0, batchSize * matrix.Height());
	}

	const int sharedSize = threadCount.x * threadCount.y * sizeof(float);
	MultiplyLookupMatrixByLookupVectorKernel<<<blockCount, threadCount, sharedSize>>>(batchSize,
		GetRaw(matrix.Table), matrix.Dims.VectorCount, matrix.Dims.VectorSize,
		GetRaw(matrix.Rows), matrix.RowCount, GetRaw(vector.Table), vector.Dims.VectorCount,
		GetRaw(vector.Vector), GetRaw(resultHandle), resultSize, widthNorm);
}

void CCudaMathEngine::MultiplyTransposedLookupMatrixByVector(int batchSize, const CLookupMatrix& matrix,
	const CConstFloatHandle& vectorHandle, const CFloatHandle& resultHandle, int resultSize)
{
	ASSERT_EXPR( vectorHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	multiplyVectorByLookupMatrixImpl(batchSize, matrix, vectorHandle, resultHandle, resultSize, false);
}

void CCudaMathEngine::MultiplyTransposedLookupMatrixByVectorAndAdd(int batchSize, const CLookupMatrix& matrix,
	const CConstFloatHandle& vectorHandle, const CFloatHandle& resultHandle, int resultSize)
{
	ASSERT_EXPR( vectorHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	multiplyVectorByLookupMatrixImpl(batchSize, matrix, vectorHandle, resultHandle, resultSize, true);
}

void CCudaMathEngine::MultiplyVectorByTransposedLookupVectorAndAddToTable(int batchSize,
	const CFloatHandle& table, int vectorCount, int vectorSize, const CConstIntHandle& indexHandle,
	const CConstFloatHandle& firstHandle, int firstSize, const CLookupVector& second)
{
	ASSERT_EXPR( table.GetMathEngine() == this );
	ASSERT_EXPR( indexHandle.GetMathEngine() == this );
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR(vectorSize == second.VectorSize());
	SetCudaDevice( device->DeviceNumber );

	int vectorSizeNorm = (vectorSize + MultiplyVectorByTransposedLookupVectorAndAddToTableCombine - 1) /
		MultiplyVectorByTransposedLookupVectorAndAddToTableCombine;

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2D(blockCount, threadCount, batchSize * firstSize, vectorSizeNorm);

	MultiplyVectorByTransposedLookupVectorAndAddToTableKernel<<<blockCount, threadCount>>>(batchSize,
		GetRaw(table), vectorCount, vectorSize, GetRaw(indexHandle),
		GetRaw(firstHandle), firstSize, GetRaw(second.Table), GetRaw(second.Vector), vectorSizeNorm);
}

void CCudaMathEngine::MultiplyDiagMatrixByMatrix(const CConstFloatHandle& firstHandle, int firstSize,
	const CConstFloatHandle& secondHandle, int secondWidth,
	const CFloatHandle& resultHandle, int)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	dim3 blockCount;
	dim3 threadCount;

	getCudaTaskGrid2D(blockCount, threadCount, firstSize, secondWidth);

	MultiplyDiagMatrixByMatrixKernel<<<blockCount, threadCount>>>
		(GetRaw(firstHandle), firstSize, GetRaw(secondHandle), secondWidth, GetRaw(resultHandle));
}

void CCudaMathEngine::Multiply1DiagMatrixByMatrix(int batchSize, const CConstFloatHandle& firstHandle, int firstSize,
	const CConstFloatHandle& secondHandle, int secondWidth,
	const CFloatHandle& resultHandle, int)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	dim3 blockCount;
	dim3 threadCount;

	int batchNorm = (batchSize + Multiply1DiagMatrixByMatrixCombine - 1) /
		Multiply1DiagMatrixByMatrixCombine;

	getCudaTaskGrid2DMinYX(1, 256, blockCount, threadCount, batchNorm, firstSize * secondWidth);

	Multiply1DiagMatrixByMatrixKernel<<<blockCount, threadCount>>>
		(batchSize, GetRaw(firstHandle), firstSize, GetRaw(secondHandle), secondWidth, GetRaw(resultHandle), batchNorm);
}

void CCudaMathEngine::TransposeMatrix(int batchSize, const CConstFloatHandle& firstHandle,
	int height, int medium, int width, int channels, const CFloatHandle& resultHandle, int resultBufferSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	transposeMatrixImpl(batchSize, firstHandle, height, medium, width, channels, resultHandle, resultBufferSize);
}

void CCudaMathEngine::TransposeMatrix(int batchSize, const CConstIntHandle& firstHandle,
	int height, int medium, int width, int channels, const CIntHandle& resultHandle, int resultBufferSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	transposeMatrixImpl(batchSize, firstHandle, height, medium, width, channels, resultHandle, resultBufferSize);
}

void CCudaMathEngine::MultiplyDiagMatrixByMatrixAndAdd( int batchSize, const CConstFloatHandle& firstHandle,
	int firstSize, const CConstFloatHandle& secondHandle, int secondWidth, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int batchSizeNorm = ( batchSize + MultiplyDiagMatrixByMatrixAndSumCombine - 1 )
		/ MultiplyDiagMatrixByMatrixAndSumCombine;
	batchSizeNorm = alignXSizeForWarp( batchSizeNorm );

	dim3 blockCount;
	dim3 threadCount;

	getCudaTaskGrid3DMinZYX( 1, 1, 512, blockCount, threadCount, firstSize, secondWidth, batchSizeNorm );

	int sharedSize = threadCount.x * threadCount.y * threadCount.z * sizeof( float );

	MultiplyDiagMatrixByMatrixAndSumKernel<<<blockCount, threadCount, sharedSize>>>( batchSize,
		GetRaw( firstHandle ), firstSize, GetRaw( secondHandle ), secondWidth, GetRaw( resultHandle ),
		batchSizeNorm );
}

void CCudaMathEngine::RowMultiplyMatrixByMatrix( const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, int height, int width, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	VectorFill( resultHandle, 0, height );
	int widthNorm = ( width + RowMultiplyMatrixByMatrixCombine - 1 ) / RowMultiplyMatrixByMatrixCombine;

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2DMinYX(1, 256, blockCount, threadCount, height, widthNorm );

	const int sharedSize = threadCount.y * threadCount.x * sizeof( float );
	RowMultiplyMatrixByMatrixKernel<<<blockCount, threadCount, sharedSize>>>( GetRaw( firstHandle ),
		GetRaw( secondHandle ), height, width, GetRaw( resultHandle ), widthNorm );
}

void CCudaMathEngine::MatrixSpreadRows(const CConstFloatHandle& sourceHandle, int height, int width,
	const CFloatHandle& resultHandle, int resultHeight, const CConstIntHandle& indexHandle,
	const CConstFloatHandle& fillValue)
{
	ASSERT_EXPR( indexHandle.GetMathEngine() == this );
	ASSERT_EXPR( sourceHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( fillValue.IsNull() || fillValue.GetMathEngine() == this );

	matrixSpreadRowsImpl(GetRaw(sourceHandle), height, width,
		resultHandle, resultHeight, GetRaw(indexHandle), fillValue);
}

void CCudaMathEngine::MatrixSpreadRowsAdd(const CConstFloatHandle& sourceHandle, int height, int width,
	const CFloatHandle& resultHandle, int resultHeight, const CConstIntHandle& indexHandle)
{
	ASSERT_EXPR( indexHandle.GetMathEngine() == this );
	ASSERT_EXPR( sourceHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int widthNorm = (width + MatrixSpreadRowsCombine - 1) / MatrixSpreadRowsCombine;

	dim3 blockCount;
	dim3 threadCount;

	getCudaTaskGrid2D(blockCount, threadCount, height, widthNorm);

	MatrixSpreadRowsAddKernel<<<blockCount, threadCount>>>(GetRaw(sourceHandle), height, width,
		GetRaw(resultHandle), GetRaw(indexHandle), widthNorm);
}

void CCudaMathEngine::MatrixSpreadRows(const CConstIntHandle& sourceHandle, int height, int width,
	const CIntHandle& resultHandle, int resultHeight, const CConstIntHandle& indexHandle,
	const CConstIntHandle& fillValue)
{
	ASSERT_EXPR( indexHandle.GetMathEngine() == this );
	ASSERT_EXPR( sourceHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( fillValue.IsNull() || fillValue.GetMathEngine() == this );

	matrixSpreadRowsImpl(GetRaw(sourceHandle), height, width,
		resultHandle, resultHeight, GetRaw(indexHandle), fillValue);
}

void CCudaMathEngine::SingularValueDecomposition( const CFloatHandle& a, int height, int width, const CFloatHandle& u, const CFloatHandle& s,
	const CFloatHandle& vt, const CFloatHandle& superb, bool returnLeftVectors, bool returnRightVectors )
{
	ASSERT_EXPR( false );
}

void CCudaMathEngine::QRFactorization( int height, int width, const CFloatHandle& matrixHandle, const CFloatHandle* qHandle, const CFloatHandle* rHandle,
	bool inplace, bool returnQ, bool returnR )
{
	ASSERT_EXPR( false );
}

void CCudaMathEngine::LookupAndSum( const CConstIntHandle& indicesHandle, int batchSize, int indexCount,
	const CConstFloatHandle& tableHandle, int vectorSize, const CFloatHandle& result )
{
	ASSERT_EXPR( indicesHandle.GetMathEngine() == this );
	ASSERT_EXPR( tableHandle.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	dim3 blockCount, threadCount;
	getCudaTaskGrid2D( blockCount, threadCount, batchSize, vectorSize );

	LookupAndSumKernel<<<blockCount, threadCount>>>( GetRaw( indicesHandle ), batchSize, indexCount,
		GetRaw( tableHandle ), vectorSize, GetRaw( result ) );
}

void CCudaMathEngine::LookupAndAddToTable( const CConstIntHandle& indicesHandle, int batchSize, int indexCount,
	const CConstFloatHandle& additionsHandle, int vectorSize, const CFloatHandle& tableHandle, int vectorCount )
{
	ASSERT_EXPR( indicesHandle.GetMathEngine() == this );
	ASSERT_EXPR( tableHandle.GetMathEngine() == this );
	ASSERT_EXPR( additionsHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	VectorFill( tableHandle, 0.f, vectorSize * vectorCount );

	dim3 blockCount, threadCount;
	getCudaTaskGrid3D( blockCount, threadCount, batchSize, indexCount, vectorSize );

	LookupAndAddToTableKernel<<<blockCount, threadCount>>>( GetRaw( indicesHandle ), batchSize, indexCount,
		GetRaw( additionsHandle ), vectorSize, GetRaw( tableHandle ) );
}

void CCudaMathEngine::EnumBinarization(int batchSize,
	const CConstFloatHandle& inputHandle, int enumSize, const CFloatHandle& resultHandle)
{
	ASSERT_EXPR( inputHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, batchSize * enumSize, EnumBinarizationCombine);

	EnumBinarizationKernel<<<blockCount, threadCount>>>(batchSize,
		GetRaw(inputHandle), enumSize, GetRaw(resultHandle));
}

void CCudaMathEngine::EnumBinarization(int batchSize,
	const CConstIntHandle& inputHandle, int enumSize, const CFloatHandle& resultHandle)
{
	ASSERT_EXPR( inputHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, batchSize * enumSize, EnumBinarizationCombine);

	EnumBinarizationKernel<<<blockCount, threadCount>>>(batchSize,
		GetRaw(inputHandle), enumSize, GetRaw(resultHandle));
}

template<class T>
void CCudaMathEngine::transposeMatrixImpl(int batchSize, const CTypedMemoryHandle<const T>& firstHandle,
	int height, int medium, int width, int channels, const CTypedMemoryHandle<T>& resultHandle, int resultBufferSize)
{
	int size = batchSize * height * medium * width * channels;
	ASSERT_EXPR(resultBufferSize >= size);

	if( medium == 1 && ( height == 1 || width == 1 ) ) {
		VectorCopy( resultHandle, firstHandle, size );
		return;
	}

	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, size, TransposeMatrixCombine);

	TransposeMatrixKernel<<<blockCount, threadCount>>>(batchSize, GetRaw(firstHandle),
		height, medium, width, channels, GetRaw(resultHandle), size);
}

void CCudaMathEngine::sumMatrixColumnsKernelFunc(const CFloatHandle& resultHandle, const float* matrix,
	int matrixHeight, int matrixWidth, bool isNeg)
{
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int widthNorm = (matrixWidth + SumMatrixColumnsCombine - 1) / SumMatrixColumnsCombine;
	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2DMinYX(1, 256, blockCount, threadCount, matrixHeight, widthNorm);
	int maxAtomicPerX =  SumMatrixColumnsMaxAtomic / blockCount.y;
	if(maxAtomicPerX <= 0) {
		maxAtomicPerX = 1;
	}
	if((int)blockCount.x > maxAtomicPerX) {
		blockCount.x = maxAtomicPerX;
	}
	int totalThreadXCount = threadCount.x * blockCount.x;
	int combine = (matrixWidth + totalThreadXCount - 1) / totalThreadXCount;

	if( blockCount.x > 1 ) {
		VectorFill(resultHandle, 0, matrixHeight);
	}

	const int sharedSize = threadCount.y * threadCount.x * sizeof(float);
	SumMatrixColumnsKernel<<<blockCount, threadCount, sharedSize>>>
		(GetRaw(resultHandle), matrix, matrixHeight, matrixWidth, isNeg, widthNorm, combine);
}

void CCudaMathEngine::multiplyVectorByLookupMatrixImpl(int batchSize, const CLookupMatrix& matrix,
	const CConstFloatHandle& vectorHandle, const CFloatHandle& resultHandle, int resultSize, bool isAdd)
{
	ASSERT_EXPR( vectorHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR(resultSize >= batchSize * matrix.Width());
	SetCudaDevice( device->DeviceNumber );

	int heightNorm = (matrix.Height() + MultiplyTransposedLookupMatrixByVectorCombine - 1) /
		MultiplyTransposedLookupMatrixByVectorCombine;
	heightNorm = alignXSizeForWarp(heightNorm);
	// X coordinate is Height to allow for warp reduction

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2DMinYX(1, 1024, blockCount, threadCount, batchSize * matrix.Width(), heightNorm);

	if(blockCount.x > 0 && !isAdd) {
		// Several GPUs may take part in adding up one column, need atomic operations
		// Set resultHandle to zeros
		VectorFill(resultHandle, 0, batchSize * matrix.Width());
	}

	const int sharedSize = threadCount.x * threadCount.y * sizeof(float);
	MultiplyTransposedLookupMatrixByVectorKernel<<<blockCount, threadCount, sharedSize>>>(batchSize,
		GetRaw(matrix.Table), matrix.Dims.VectorCount, matrix.Dims.VectorSize, GetRaw(matrix.Rows), matrix.RowCount,
		GetRaw(vectorHandle), GetRaw(resultHandle), isAdd, heightNorm);
}

template<class T>
void CCudaMathEngine::matrixSpreadRowsImpl(const T* source, int height, int width,
	CTypedMemoryHandle<T> result, int resultHeight, const int* index, const CTypedMemoryHandle<const T>& fillValue)
{
	SetCudaDevice( device->DeviceNumber );
	if(fillValue.IsNull()) {
		VectorFill( result, 0, resultHeight * width);
	} else {
		VectorFill( result, resultHeight * width, fillValue);
	}

	int widthNorm = (width + MatrixSpreadRowsCombine - 1) / MatrixSpreadRowsCombine;

	dim3 blockCount;
	dim3 threadCount;

	getCudaTaskGrid2D(blockCount, threadCount, height, widthNorm);

	MatrixSpreadRowsKernel<T><<<blockCount, threadCount>>>(source, height, width,
		GetRaw( result ), index, widthNorm);
}

template<class TInput, class TLookup>
void CCudaMathEngine::vectorMultichannelLookupAndCopy(int batchSize, int channelCount, const CTypedMemoryHandle<const TInput>& inputHandle,
	const CTypedMemoryHandle<const TLookup>* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
	const CTypedMemoryHandle<TLookup>& outputHandle, int outputChannelsCount)
{
	SetCudaDevice( device->DeviceNumber );
	int batchNorm = (batchSize + BatchVectorLookupAndCopyCombineBatch - 1) / BatchVectorLookupAndCopyCombineBatch;

	int outputChannel = 0;
	for(int j = 0; j < lookupCount; ++j) {
		dim3 blockCount;
		dim3 threadCount;
		getCudaTaskGrid2D(blockCount, threadCount, batchNorm, lookupDimensions[j].VectorSize);

		VectorChannelLookupAndCopyKernel<<<blockCount, threadCount>>>(batchSize, GetRaw(inputHandle) + j, channelCount,
			GetRaw(lookupHandles[j]), lookupDimensions[j].VectorSize, GetRaw(outputHandle) + outputChannel, outputChannelsCount, batchNorm);

		outputChannel += lookupDimensions[j].VectorSize;
	}
	if(lookupCount < channelCount) {
		dim3 blockCount;
		dim3 threadCount;
		getCudaTaskGrid2D(blockCount, threadCount, batchNorm, channelCount - lookupCount);

		BatchVectorChannelCopyKernel<<<blockCount, threadCount>>>(batchSize, GetRaw(inputHandle) + lookupCount, channelCount, channelCount - lookupCount,
			GetRaw(outputHandle) + outputChannel, outputChannelsCount, batchNorm);
	}
}

template<class T>
void CCudaMathEngine::vectorMultichannelLookupAndAddToTable(int batchSize, int channelCount, const CTypedMemoryHandle<const T>& inputHandle,
	const CFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
	const CConstFloatHandle& multHandle, const CConstFloatHandle& matrixHandle, int outputChannelsCount)
{
	SetCudaDevice( device->DeviceNumber );
	int batchNorm = (batchSize + BatchVectorLookupAndAddToTableCombine - 1) / BatchVectorLookupAndAddToTableCombine;

	float mult = multHandle.GetValue();

	int outputChannel = 0;
	for (int j = 0; j < lookupCount; ++j) {
		dim3 blockCount;
		dim3 threadCount;
		getCudaTaskGrid2D(blockCount, threadCount, batchNorm, lookupDimensions[j].VectorSize);

		VectorChannelLookupAndAddToTableKernel<<<blockCount, threadCount>>>(batchSize, GetRaw(inputHandle) + j, channelCount,
			GetRaw(lookupHandles[j]), lookupDimensions[j].VectorSize, mult, GetRaw(matrixHandle) + outputChannel, outputChannelsCount, batchNorm);

		outputChannel += lookupDimensions[j].VectorSize;
	}
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
