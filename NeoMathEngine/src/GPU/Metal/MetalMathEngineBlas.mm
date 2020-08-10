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

#ifdef NEOML_USE_METAL

#include <MetalMathEngine.h>
#include <MetalKernel.h>
#include <MathEngineCommon.h>
#include <algorithm>

@import Foundation;
@import MetalKit;

namespace NeoML {
    
// The number of combined values for the vector kernels
static const int VectorCombineCount = 8;

void CMetalMathEngine::AddMatrixElementsToVector(const CConstFloatHandle& matrix, int height, int width,
	const CConstIntHandle& indices, const CFloatHandle& result, int)
{
    ASSERT_EXPR( matrix.GetMathEngine() == this );
	ASSERT_EXPR( indices.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelAddMatrixElementsToVector", VectorCombineCount, height );
    kernel.SetParam( matrix, 0 );
    kernel.SetParam( height, 1 );
    kernel.SetParam( width, 2 );
    kernel.SetParam( indices, 3 );
    kernel.SetParam( result, 4 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::AddMatrixElementsToVector(const CConstFloatHandle& matrix, int height, int width,
	const CConstIntHandle& rowIndices, const CConstIntHandle& columnIndices,
	const CFloatHandle& result, int vectorSize)
{
    ASSERT_EXPR( matrix.GetMathEngine() == this );
	ASSERT_EXPR( rowIndices.GetMathEngine() == this );
	ASSERT_EXPR( columnIndices.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this ); 

    C1DKernel kernel( *queue, "vectorKernelAddMatrixElementsToVectorEx", VectorCombineCount, vectorSize );
    kernel.SetParam( matrix, 0 );
    kernel.SetParam( height, 1 );
    kernel.SetParam( width, 2 );
    kernel.SetParam( rowIndices, 3 );
    kernel.SetParam( columnIndices, 4 );
    kernel.SetParam( result, 5 );
    kernel.SetParam( vectorSize, 6 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorMultichannelLookupAndCopy(int batchSize, int channelCount, const CConstFloatHandle& inputHandle,
    const CConstFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
    const CFloatHandle& outputHandle, int outputChannelsCount)
{
    ASSERT_EXPR( inputHandle.GetMathEngine() == this );
	ASSERT_EXPR( outputHandle.GetMathEngine() == this );

    int outputChannel = 0;
    for( int i = 0; i < lookupCount; ++i ) {
        C2DKernel kernel( *queue, "matrixKernelBatchVectorChannelLookupAndCopyFloat", 4, 1, batchSize, lookupDimensions[i].VectorSize );
        kernel.SetParam( batchSize, 0 );
        kernel.SetParam( inputHandle + i, 1 );
        kernel.SetParam( channelCount, 2 );
        kernel.SetParam( lookupHandles[i], 3 );
        kernel.SetParam( lookupDimensions[i].VectorSize, 4 );
        kernel.SetParam( outputHandle + outputChannel, 5 );
        kernel.SetParam( outputChannelsCount, 6 );
        kernel.SetParam( kernel.GetGridWidth(), 7 );
        ASSERT_EXPR( kernel.Run() );
        
        outputChannel += lookupDimensions[i].VectorSize;
    }
    
    if( lookupCount < channelCount ) {
        C2DKernel kernel( *queue, "matrixKernelBatchVectorChannelCopyFloat",
            4, 1, batchSize, channelCount - lookupCount );
        kernel.SetParam( batchSize, 0 );
        kernel.SetParam( inputHandle + lookupCount, 1 );
        kernel.SetParam( channelCount, 2 );
        kernel.SetParam( channelCount - lookupCount, 3 );
        kernel.SetParam( outputHandle + outputChannel, 5 );
        kernel.SetParam( outputChannelsCount, 6 );
        kernel.SetParam( kernel.GetGridWidth(), 7 );
        ASSERT_EXPR( kernel.Run() );
    }
}

void CMetalMathEngine::VectorMultichannelLookupAndCopy(int batchSize, int channelCount, const CConstIntHandle& inputHandle,
    const CConstFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
    const CFloatHandle& outputHandle, int outputChannelsCount)
{
    ASSERT_EXPR( inputHandle.GetMathEngine() == this );
	ASSERT_EXPR( outputHandle.GetMathEngine() == this );

    int outputChannel = 0;
    for( int i = 0; i < lookupCount; ++i ) {
        C2DKernel kernel( *queue, "matrixKernelBatchVectorChannelLookupAndCopyInt", 4, 1, batchSize, lookupDimensions[i].VectorSize );
        kernel.SetParam( batchSize, 0 );
        kernel.SetParam( inputHandle + i, 1 );
        kernel.SetParam( channelCount, 2 );
        kernel.SetParam( lookupHandles[i], 3 );
        kernel.SetParam( lookupDimensions[i].VectorSize, 4 );
        kernel.SetParam( outputHandle + outputChannel, 5 );
        kernel.SetParam( outputChannelsCount, 6 );
        kernel.SetParam( kernel.GetGridWidth(), 7 );
        ASSERT_EXPR( kernel.Run() );
        
        outputChannel += lookupDimensions[i].VectorSize;
    }
    
    if( lookupCount < channelCount ) {
        C2DKernel kernel( *queue, "matrixKernelBatchVectorChannelCopyInt", 4, 1, batchSize, channelCount - lookupCount );
        kernel.SetParam( batchSize, 0 );
        kernel.SetParam( inputHandle + lookupCount, 1 );
        kernel.SetParam( channelCount, 2 );
        kernel.SetParam( channelCount - lookupCount, 3 );
        kernel.SetParam( outputHandle + outputChannel, 5 );
        kernel.SetParam( outputChannelsCount, 6 );
        kernel.SetParam( kernel.GetGridWidth(), 7 );
        ASSERT_EXPR( kernel.Run() );
    }
}

void CMetalMathEngine::VectorMultichannelLookupAndAddToTable(int batchSize, int channelCount, const CConstFloatHandle& inputHandle,
    const CFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount, const CConstFloatHandle& multHandle,
    const CConstFloatHandle& matrixHandle, int outputChannelsCount)
{
    ASSERT_EXPR( inputHandle.GetMathEngine() == this );
	ASSERT_EXPR( multHandle.GetMathEngine() == this );
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );

    int outputChannel = 0;
    for( int i = 0; i < lookupCount; ++i ) {
        C2DKernel kernel( *queue, "matrixKernelBatchVectorChannelLookupAndAddToTableFloat", 1, 1, lookupDimensions[i].VectorCount, lookupDimensions[i].VectorSize );
        kernel.SetParam( batchSize, 0 );
        kernel.SetParam( inputHandle, 1 );
        kernel.SetParam( i, 2 );
        kernel.SetParam( channelCount, 3 );
        kernel.SetParam( lookupHandles[i], 4 );
        kernel.SetParam( lookupDimensions[i].VectorCount, 5 );
        kernel.SetParam( lookupDimensions[i].VectorSize, 6 );
        kernel.SetParam( multHandle, 7 );
        kernel.SetParam( matrixHandle, 8 );
        kernel.SetParam( outputChannel, 9 );
        kernel.SetParam( outputChannelsCount, 10 );
        ASSERT_EXPR( kernel.Run() );
        
        outputChannel += lookupDimensions[i].VectorSize;
    }
}

void CMetalMathEngine::VectorMultichannelLookupAndAddToTable(int batchSize, int channelCount, const CConstIntHandle& inputHandle,
    const CFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount, const CConstFloatHandle& multHandle,
    const CConstFloatHandle& matrixHandle, int outputChannelsCount)
{
    ASSERT_EXPR( inputHandle.GetMathEngine() == this );
	ASSERT_EXPR( multHandle.GetMathEngine() == this );
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );

    int outputChannel = 0;
    for( int i = 0; i < lookupCount; ++i ) {
        C2DKernel kernel( *queue, "matrixKernelBatchVectorChannelLookupAndAddToTableInt", 1, 1, lookupDimensions[i].VectorCount, lookupDimensions[i].VectorSize );
        kernel.SetParam( batchSize, 0 );
        kernel.SetParam( inputHandle, 1 );
        kernel.SetParam( i, 2 );
        kernel.SetParam( channelCount, 3 );
        kernel.SetParam( lookupHandles[i], 4 );
        kernel.SetParam( lookupDimensions[i].VectorCount, 5 );
        kernel.SetParam( lookupDimensions[i].VectorSize, 6 );
        kernel.SetParam( multHandle, 7 );
        kernel.SetParam( matrixHandle, 8 );
        kernel.SetParam( outputChannel, 9 );
        kernel.SetParam( outputChannelsCount, 10 );
        ASSERT_EXPR( kernel.Run() );
        
        outputChannel += lookupDimensions[i].VectorSize;
    }
}

void CMetalMathEngine::LookupAndSum( const CConstIntHandle& indicesHandle, int batchSize, int indexCount,
    const CConstFloatHandle& tableHandle, int vectorSize, const CFloatHandle& result )
{
    ASSERT_EXPR( indicesHandle.GetMathEngine() == this );
	ASSERT_EXPR( tableHandle.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );

    C2DKernel kernel( *queue, "matrixKernelLookupAndSum", 1, 1, batchSize, vectorSize );
    kernel.SetParam( indicesHandle, 0 );
    kernel.SetParam( batchSize, 1 );
    kernel.SetParam( indexCount, 2 );
    kernel.SetParam( tableHandle, 3 );
    kernel.SetParam( vectorSize, 4 );
    kernel.SetParam( result, 5 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::LookupAndAddToTable( const CConstIntHandle& indicesHandle, int batchSize, int indexCount,
    const CConstFloatHandle& additionsHandle, int vectorSize, const CFloatHandle& tableHandle, int vectorCount )
{
    ASSERT_EXPR( indicesHandle.GetMathEngine() == this );
	ASSERT_EXPR( additionsHandle.GetMathEngine() == this );
	ASSERT_EXPR( tableHandle.GetMathEngine() == this );

    C2DKernel kernel( *queue, "matrixKernelLookupAndAddToTable", 1, 1, vectorCount, vectorSize );
    kernel.SetParam( indicesHandle, 0 );
    kernel.SetParam( batchSize, 1 );
    kernel.SetParam( indexCount, 2 );
    kernel.SetParam( additionsHandle, 3 );
    kernel.SetParam( vectorSize, 4 );
    kernel.SetParam( tableHandle, 5 );
    kernel.SetParam( vectorCount, 6 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::AddMatrixElementsToMatrix(const CConstFloatHandle& matrix, int height, int width,
	const CFloatHandle& result, const CConstIntHandle& indices)
{
    ASSERT_EXPR( matrix.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );
	ASSERT_EXPR( indices.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelAddMatrixElementsToMatrix", VectorCombineCount, height );
    kernel.SetParam( matrix, 0 );
    kernel.SetParam( height, 1 );
    kernel.SetParam( width, 2 );
    kernel.SetParam( result, 3 );
    kernel.SetParam( indices, 4 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::EltwiseLogSumExpVectorToMatrixElements(const CFloatHandle& matrix, int height, int width,
	const CConstIntHandle& rowIndices, const CConstIntHandle& columnIndices,
	const CConstFloatHandle& vector, int vectorSize)
{
    ASSERT_EXPR( matrix.GetMathEngine() == this );
	ASSERT_EXPR( rowIndices.GetMathEngine() == this );
	ASSERT_EXPR( columnIndices.GetMathEngine() == this );
	ASSERT_EXPR( vector.GetMathEngine() == this ); 

    C2DKernel kernel( *queue, "matrixKernelEltwiseLogSumExpVectorToMatrixElements", 1, 1, height, width );
    kernel.SetParam( matrix, 0 );
    kernel.SetParam( height, 1 );
    kernel.SetParam( width, 2 );
    kernel.SetParam( rowIndices, 3 );
    kernel.SetParam( columnIndices, 4 );
    kernel.SetParam( vector, 5 );
    kernel.SetParam( vectorSize, 6 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::BitSetBinarization(int batchSize, int bitSetSize,
    const CConstIntHandle& inputHandle, int outputVectorSize, const CFloatHandle& resultHandle)
{
    ASSERT_EXPR( inputHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelBitSetBinarization", 1, batchSize * outputVectorSize );
    kernel.SetParam( batchSize, 0 );
    kernel.SetParam( bitSetSize, 1 );
    kernel.SetParam( inputHandle, 2 );
    kernel.SetParam( outputVectorSize, 3 );
    kernel.SetParam( resultHandle, 4 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::EnumBinarization(int batchSize, const CConstFloatHandle& inputHandle, int enumSize,
    const CFloatHandle& resultHandle)
{
    ASSERT_EXPR( inputHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelEnumBinarizationFloat", VectorCombineCount, batchSize * enumSize );
    kernel.SetParam( batchSize, 0 );
    kernel.SetParam( inputHandle, 1 );
    kernel.SetParam( enumSize, 2 );
    kernel.SetParam( resultHandle, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::EnumBinarization(int batchSize, const CConstIntHandle& inputHandle, int enumSize,
    const CFloatHandle& resultHandle)
{
    ASSERT_EXPR( inputHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelEnumBinarizationInt", VectorCombineCount, batchSize * enumSize );
    kernel.SetParam( batchSize, 0 );
    kernel.SetParam( inputHandle, 1 );
    kernel.SetParam( enumSize, 2 );
    kernel.SetParam( resultHandle, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::RowMultiplyMatrixByMatrix( const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, int height, int width, const CFloatHandle& resultHandle )
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	VectorFill( resultHandle, 0.0f, height );

    C2DKernel kernel( *queue, "matrixKernelRowMultiplyMatrixByMatrix", 1, 1, height, width );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( height, 2 );
    kernel.SetParam( width, 3 );
    kernel.SetParam( resultHandle, 4 );
    kernel.SetSharedParam( kernel.GetThreadCount() * sizeof( float ), 5 );
    
    // threadgroupCount.width = 1;
    ASSERT_EXPR( kernel.Run( 0, 0, 1 ) );
}

void CMetalMathEngine::VectorMultiplyAndAdd(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multHandle)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( multHandle.GetMathEngine() == this ); 

    C1DKernel kernel( *queue, "vectorKernelVectorMultiplyAndAdd", 1, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    kernel.SetParam( multHandle, 4 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorMultiplyAndSub(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multHandle)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( multHandle.GetMathEngine() == this ); 

    C1DKernel kernel( *queue, "vectorKernelVectorMultiplyAndSub", 1, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    kernel.SetParam( multHandle, 4 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorEltwiseNotNegative( const CConstIntHandle& firstHanle, const CFloatHandle& resultHandle,
    int vectorSize )
{
    ASSERT_EXPR( firstHanle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelEltwiseNotNegative", 1, vectorSize );
    kernel.SetParam( firstHanle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::SetVectorToMatrixRows(const CFloatHandle& resultHandle,
	int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle)
{
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorHandle.GetMathEngine() == this );

    C2DKernel kernel( *queue, "matrixKernelSetVectorToMatrixRows", 1, 1, matrixHeight, matrixWidth );
    kernel.SetParam( resultHandle, 0 );
    kernel.SetParam( matrixHeight, 1 );
    kernel.SetParam( matrixWidth, 2 );
    kernel.SetParam( vectorHandle, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::SetVectorToMatrixElements( const CFloatHandle& matrixHandle, int height, int width,
    const CConstIntHandle& rowIndicesHandle, const CConstIntHandle& columnIndicesHandle,
    const CConstFloatHandle& vectorHandle, int vectorSize )
{
    ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( rowIndicesHandle.GetMathEngine() == this );
	ASSERT_EXPR( columnIndicesHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorHandle.GetMathEngine() == this ); 

    C1DKernel kernel( *queue, "vectorKernelSetVectorToMatrixElements", 4, vectorSize );
    kernel.SetParam( matrixHandle, 0 );
    kernel.SetParam( height, 1 );
    kernel.SetParam( width, 2 );
    kernel.SetParam( rowIndicesHandle, 3 );
    kernel.SetParam( columnIndicesHandle, 4 );
    kernel.SetParam( vectorHandle, 5 );
    kernel.SetParam( vectorSize, 6 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::AddVectorToMatrixColumns( const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
	int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle )
{
    ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorHandle.GetMathEngine() == this );

    C2DKernel kernel( *queue, "matrixKernelAddVectorToMatrixColumnsFloat", 1, 1, matrixHeight, matrixWidth );
    kernel.SetParam( matrixHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( matrixHeight, 2 );
    kernel.SetParam( matrixWidth, 3 );
    kernel.SetParam( vectorHandle, 4 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::AddVectorToMatrixColumns( const CConstIntHandle& matrixHandle, const CIntHandle& resultHandle,
	int matrixHeight, int matrixWidth, const CConstIntHandle& vectorHandle )
{
    ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorHandle.GetMathEngine() == this );

    C2DKernel kernel( *queue, "matrixKernelAddVectorToMatrixColumnsInt", 1, 1, matrixHeight, matrixWidth );
    kernel.SetParam( matrixHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( matrixHeight, 2 );
    kernel.SetParam( matrixWidth, 3 );
    kernel.SetParam( vectorHandle, 4 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::SubVectorFromMatrixColumns(const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
	int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle)
{
    ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorHandle.GetMathEngine() == this );

    C2DKernel kernel( *queue, "matrixKernelSubVectorFromMatrixColumns", 1, 1, matrixHeight, matrixWidth );
    kernel.SetParam( matrixHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( matrixHeight, 2 );
    kernel.SetParam( matrixWidth, 3 );
    kernel.SetParam( vectorHandle, 4 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::SumMatrixColumns(const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
	int matrixHeight, int matrixWidth)
{
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );

    VectorFill( resultHandle, 0.0f, matrixHeight );
    
    C2DKernel kernel( *queue, "matrixKernelSumMatrixColumns", 1, 1, matrixHeight, matrixWidth );
    kernel.SetParam( resultHandle, 0 );
    kernel.SetParam( matrixHandle, 1 );
    kernel.SetParam( matrixHeight, 2 );
    kernel.SetParam( matrixWidth, 3 );
    kernel.SetParam( 0, 4 );
    kernel.SetSharedParam( kernel.GetThreadCount() * sizeof( float ), 5 );
    
    // threadgroupCount.width = 1;
    ASSERT_EXPR( kernel.Run( 0, 0, 1 ) );
}

void CMetalMathEngine::FindMaxValueInRows(const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth,
	const CFloatHandle& resultHandle, const CIntHandle& columnIndices, int /*vectorSize*/)
{
    ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( columnIndices.GetMathEngine() == this );

    VectorFill( resultHandle, -FLT_MAX, matrixHeight );
   
    C2DKernel kernel( *queue, "matrixKernelFindMaxValueWithIndicesInRows", 1, 1, matrixHeight, matrixWidth );
    kernel.SetParam( matrixHandle, 0 );
    kernel.SetParam( matrixHeight, 1 );
    kernel.SetParam( matrixWidth, 2 );
    kernel.SetParam( resultHandle, 3 );
    kernel.SetParam( columnIndices, 4 );
    kernel.SetSharedParam( kernel.GetThreadCount() * sizeof( float ), 5 );
    kernel.SetSharedParam( kernel.GetThreadCount() * sizeof( int ), 6 );
    
    // threadgroupCount.width = 1;
    ASSERT_EXPR( kernel.Run( 0, 0, 1 ) );
}

void CMetalMathEngine::FindMaxValueInRows(const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth,
	const CFloatHandle& resultHandle, int /*vectorSize*/)
{
    ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    VectorFill( resultHandle, -FLT_MAX, matrixHeight );

    C2DKernel kernel( *queue, "matrixKernelFindMaxValueInRows", 1, 1, matrixHeight, matrixWidth );
    kernel.SetParam( matrixHandle, 0 );
    kernel.SetParam( matrixHeight, 1 );
    kernel.SetParam( matrixWidth, 2 );
    kernel.SetParam( resultHandle, 3 );
    kernel.SetSharedParam( kernel.GetThreadCount() * sizeof( float ), 4 );
    
    // threadgroupCount.width = 1;
    ASSERT_EXPR( kernel.Run( 0, 0, 1 ) );
}

void CMetalMathEngine::FindMaxValueInColumns( int batchSize, const CConstFloatHandle& matrixHandle, int matrixHeight,
    int matrixWidth, const CFloatHandle& resultHandle, const CIntHandle& rowIndices, int /*vectorSize*/ )
{
    ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( rowIndices.GetMathEngine() == this );

    C2DKernel kernel( *queue, "matrixKernelFindMaxValueInColumns", 1, 1, batchSize, matrixWidth );
    kernel.SetParam( batchSize, 0);
    kernel.SetParam( matrixHandle, 1 );
    kernel.SetParam( matrixHeight, 2 );
    kernel.SetParam( matrixWidth, 3 );
    kernel.SetParam( resultHandle, 4 );
    kernel.SetParam( rowIndices, 5 );
    
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::FindMinValueInColumns( const CConstFloatHandle&, int, int, const CFloatHandle&, const CIntHandle& )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::MultiplyMatrixByMatrix( int batchSize, const CConstFloatHandle& firstHandle, int firstHeight,
    int firstWidth, const CConstFloatHandle& secondHandle, int secondWidth, const CFloatHandle& resultHandle, int /*resultBufferSize*/)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    if( batchSize == 1 ) {
        if( firstHeight >= 4 && secondWidth >= 4 ) {
            C2DKernel kernel( *queue, "matrixKernelMultiplyMatrixByMatrixThread4x4", 4, 4, firstHeight - firstHeight % 4, secondWidth - secondWidth % 4 );
            kernel.SetParam( firstHandle, 0 );
            kernel.SetParam( firstHeight, 1 );
            kernel.SetParam( firstWidth, 2 );
            kernel.SetParam( secondHandle, 3 );
            kernel.SetParam( secondWidth, 4 );
            kernel.SetParam( resultHandle, 5 );
            ASSERT_EXPR( kernel.Run() );

            int leftOffset = secondWidth - secondWidth % 4;
            int topOffset = firstHeight - firstHeight % 4;
            int count = secondWidth * firstHeight - ( leftOffset * topOffset );
            if( count > 0 ) {
                C1DKernel kernel2( *queue, "matrixKernelMultiplyMatrixByMatrixThread4x4Borders", 1, count);
                kernel2.SetParam( firstHandle, 0 );
                kernel2.SetParam( firstHeight, 1 );
                kernel2.SetParam( firstWidth, 2 );
                kernel2.SetParam( secondHandle, 3 );
                kernel2.SetParam( secondWidth, 4 );
                kernel2.SetParam( leftOffset, 5 );
                kernel2.SetParam( topOffset, 6 );
                kernel2.SetParam( resultHandle, 7 );
                kernel2.Run();
            }
        } else {
            C2DKernel kernel( *queue, "matrixKernelMultiplyMatrixByMatrixThread1x1", 1, 1, firstHeight, secondWidth );
            kernel.SetParam( firstHandle, 0 );
            kernel.SetParam( firstHeight, 1 );
            kernel.SetParam( firstWidth, 2 );
            kernel.SetParam( secondHandle, 3 );
            kernel.SetParam( secondWidth, 4 );
            kernel.SetParam( resultHandle, 5 );
            ASSERT_EXPR( kernel.Run() );
        }
    } else {
        C3DKernel kernel( *queue, "cubeKernelMultiplyMatrixByMatrix", 1, 1, 1, batchSize, firstHeight, secondWidth );
        kernel.SetParam( batchSize, 0 );
        kernel.SetParam( firstHandle, 1 );
        kernel.SetParam( firstHeight, 2 );
        kernel.SetParam( firstWidth, 3 );
        kernel.SetParam( secondHandle, 4 );
        kernel.SetParam( secondWidth, 5 );
        kernel.SetParam( resultHandle, 6 );
        ASSERT_EXPR( kernel.Run() );
    }
}

void CMetalMathEngine::MultiplyMatrixByTransposedMatrix(const CConstFloatHandle& firstHandle, int firstHeight,
    int firstWidth, int firstRowSize, const CConstFloatHandle& secondHandle, int secondHeight, int secondRowSize,
    const CFloatHandle& resultHandle, int /*resultRowSize*/, int /*resultBufferSize */ )
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    if( firstHeight >= 4 && secondHeight >= 4 ) {
        {
            C2DKernel kernel( *queue, "matrixKernelMultiplyMatrixByTransposedMatrixThread4x4",
                4, 4, firstHeight - firstHeight % 4, secondHeight - secondHeight % 4 );
            kernel.SetParam( firstHandle, 0 );
            kernel.SetParam( firstHeight, 1 );
            kernel.SetParam( firstWidth, 2 );
            kernel.SetParam( secondHandle, 3 );
            kernel.SetParam( secondHeight, 4 );
            kernel.SetParam( resultHandle, 5 );
            ASSERT_EXPR( kernel.Run() );
        }

        int leftOffset = secondHeight - secondHeight % 4;
        int topOffset = firstHeight - firstHeight % 4;
        int count = secondHeight * firstHeight - ( leftOffset * topOffset );
        if( count > 0 ) {
            C1DKernel kernel( *queue, "matrixKernelMultiplyMatrixByTransposedMatrixThread4x4Borders", 1, count);
            kernel.SetParam( firstHandle, 0 );
            kernel.SetParam( firstHeight, 1 );
            kernel.SetParam( firstWidth, 2 );
            kernel.SetParam( secondHandle, 3 );
            kernel.SetParam( secondHeight, 4 );
            kernel.SetParam( leftOffset, 5 );
            kernel.SetParam( topOffset, 6 );
            kernel.SetParam( resultHandle, 7 );
            ASSERT_EXPR( kernel.Run() );
        }
    } else if( firstWidth % 4 == 0 && firstRowSize % 4 == 0 && secondRowSize % 4 == 0) {
        C2DKernel kernel( *queue, "matrixKernelMultiplyMatrixByTransposedMatrixThread1x1Float4", 1, 1, firstHeight, secondHeight );
        kernel.SetParam( firstHandle, 0 );
        kernel.SetParam( firstHeight, 1 );
        kernel.SetParam( firstWidth, 2 );
        kernel.SetParam( secondHandle, 3 );
        kernel.SetParam( secondHeight, 4 );
        kernel.SetParam( resultHandle, 5 );
        ASSERT_EXPR( kernel.Run() );
    } else {
        C2DKernel kernel( *queue, "matrixKernelMultiplyMatrixByTransposedMatrixThread1x1", 1, 1, firstHeight, secondHeight );
        kernel.SetParam( firstHandle, 0 );
        kernel.SetParam( firstHeight, 1 );
        kernel.SetParam( firstWidth, 2 );
        kernel.SetParam( secondHandle, 3 );
        kernel.SetParam( secondHeight, 4 );
        kernel.SetParam( resultHandle, 5 );
        ASSERT_EXPR( kernel.Run() );        
    }
}

void CMetalMathEngine::MultiplyMatrixByTransposedMatrix(int batchSize, const CConstFloatHandle& firstHandle, int firstHeight,
    int firstWidth, const CConstFloatHandle& secondHandle, int secondHeight, const CFloatHandle& resultHandle, int /*resultBufferSize*/)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C3DKernel kernel( *queue, "cubeKernelBatchMultiplyMatrixByTransposedMatrix", 1, 1, 1, batchSize, firstHeight, secondHeight );
    kernel.SetParam( batchSize, 0 );
    kernel.SetParam( firstHandle, 1 );
    kernel.SetParam( firstHeight, 2 );
    kernel.SetParam( firstWidth, 3 );
    kernel.SetParam( secondHandle, 4 );
    kernel.SetParam( secondHeight, 5 );
    kernel.SetParam( resultHandle, 6 );
    kernel.Run();
}

// result = first * T(second). The result size is firstHeight * secondHeight:
void CMetalMathEngine::MultiplySparseMatrixByTransposedMatrix( int firstHeight, int firstWidth, int secondHeight,
	const CSparseMatrixDesc& firstDesc, const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( firstDesc.Rows.GetMathEngine() == this );
	ASSERT_EXPR( firstDesc.Columns.GetMathEngine() == this );
	ASSERT_EXPR( firstDesc.Values.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	C2DKernel kernel( *queue, "matrixKernelMultiplySparseMatrixByTransposedMatrix", 1, 1, secondHeight, firstHeight );
	kernel.SetParam( firstDesc.Rows, 0 );
	kernel.SetParam( firstDesc.Columns, 1 );
	kernel.SetParam( firstDesc.Values, 2 );
	kernel.SetParam( secondHandle, 3 );
	kernel.SetParam( firstHeight, 4 );
	kernel.SetParam( firstWidth, 5 );
	kernel.SetParam( secondHeight, 6 );
	kernel.SetParam( resultHandle, 7 );
	ASSERT_EXPR( kernel.Run() ); 
}

// result = result + T(first) * second. The result size is firstWidth * secondWidth:
void CMetalMathEngine::MultiplyTransposedMatrixBySparseMatrixAndAdd( int firstHeight, int firstWidth, int secondWidth,
	const CConstFloatHandle& firstHandle, const CSparseMatrixDesc& secondDesc, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondDesc.Rows.GetMathEngine() == this );
	ASSERT_EXPR( secondDesc.Columns.GetMathEngine() == this );
	ASSERT_EXPR( secondDesc.Values.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	C1DKernel kernel( *queue, "matrixKernelMultiplyTransposedMatrixBySparseMatrix", 1, firstWidth );
	kernel.SetParam( firstHandle, 0 );
	kernel.SetParam( secondDesc.Rows, 1 );
	kernel.SetParam( secondDesc.Columns, 2 );
	kernel.SetParam( secondDesc.Values, 3 );
	kernel.SetParam( firstHeight, 4 );
	kernel.SetParam( firstWidth, 5 );
	kernel.SetParam( secondWidth, 6 );
	kernel.SetParam( resultHandle, 7 );
	ASSERT_EXPR( kernel.Run() ); 
}

void CMetalMathEngine::multiplyMatrixByTransposedMatrixAndAdd(const CConstFloatHandle& firstHandle,
	int firstHeight, int firstWidth, int /*firstRowSize*/,
	const CConstFloatHandle& secondHandle, int secondHeight, int /*secondRowSize*/,
	const CFloatHandle& resultHandle, int /*resultRowSize*/, int /*resultSize*/ )
{
    if( firstHeight >= 4 && secondHeight >= 4 ) {
        {
            C2DKernel kernel( *queue, "matrixKernelMultiplyMatrixByTransposedMatrixAndAddThread4x4", 4, 4, firstHeight - firstHeight % 4, secondHeight - secondHeight % 4 );
            kernel.SetParam( firstHandle, 0 );
            kernel.SetParam( firstHeight, 1 );
            kernel.SetParam( firstWidth, 2 );
            kernel.SetParam( secondHandle, 3 );
            kernel.SetParam( secondHeight, 4 );
            kernel.SetParam( resultHandle, 5 );
            ASSERT_EXPR( kernel.Run() );
        }

        int leftOffset = secondHeight - secondHeight % 4;
        int topOffset = firstHeight - firstHeight % 4;
        int count = secondHeight * firstHeight - ( leftOffset * topOffset );
        if( count > 0 ) {
            C1DKernel kernel( *queue, "matrixKernelMultiplyMatrixByTransposedMatrixAndAddThread4x4Borders", 1, count);
            kernel.SetParam( firstHandle, 0 );
            kernel.SetParam( firstHeight, 1 );
            kernel.SetParam( firstWidth, 2 );
            kernel.SetParam( secondHandle, 3 );
            kernel.SetParam( secondHeight, 4 );
            kernel.SetParam( leftOffset, 5 );
            kernel.SetParam( topOffset, 6 );
            kernel.SetParam( resultHandle, 7 );
            ASSERT_EXPR( kernel.Run() );
        }
    } else {
        C2DKernel kernel( *queue, "matrixKernelMultiplyMatrixByTransposedMatrixAndAddThread1x1", 1, 1, firstHeight, secondHeight );
        kernel.SetParam( firstHandle, 0 );
        kernel.SetParam( firstHeight, 1 );
        kernel.SetParam( firstWidth, 2 );
        kernel.SetParam( secondHandle, 3 );
        kernel.SetParam( secondHeight, 4 );
        kernel.SetParam( resultHandle, 5 );
        ASSERT_EXPR( kernel.Run() );
    }
}

void CMetalMathEngine::MultiplyTransposedMatrixByMatrixAndAdd(const CConstFloatHandle& firstHandle, int firstHeight, int firstWidth,
    int /*firstRowSize*/, const CConstFloatHandle& secondHandle, int secondWidth, int /*secondRowSize*/,
	const CFloatHandle& resultHandle, int /*resultRowSize*/, int /*resultSize*/)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    if( firstWidth >= 4 && secondWidth >= 4 ) {
        {
            C2DKernel kernel( *queue, "matrixKernelMultiplyTransposedMatrixByMatrixAndAddThread4x4", 4, 4, firstWidth - firstWidth % 4, secondWidth - secondWidth % 4 );
            kernel.SetParam( firstHandle, 0 );
            kernel.SetParam( firstHeight, 1 );
            kernel.SetParam( firstWidth, 2 );
            kernel.SetParam( secondHandle, 3 );
            kernel.SetParam( secondWidth, 4 );
            kernel.SetParam( resultHandle, 5 );
            ASSERT_EXPR( kernel.Run() );
        }

        int leftOffset = secondWidth - secondWidth % 4;
        int topOffset = firstWidth - firstWidth % 4;
        int count = secondWidth * firstWidth - ( leftOffset * topOffset );
        if( count > 0 ) {
            C1DKernel kernel( *queue, "matrixKernelMultiplyTransposedMatrixByMatrixAndAddThread4x4Borders", 1, count);
            kernel.SetParam( firstHandle, 0 );
            kernel.SetParam( firstHeight, 1 );
            kernel.SetParam( firstWidth, 2 );
            kernel.SetParam( secondHandle, 3 );
            kernel.SetParam( secondWidth, 4 );
            kernel.SetParam( leftOffset, 5 );
            kernel.SetParam( topOffset, 6 );
            kernel.SetParam( resultHandle, 7 );
            ASSERT_EXPR( kernel.Run() );
        }
    } else {
        C2DKernel kernel( *queue, "matrixKernelMultiplyTransposedMatrixByMatrixAndAddThread1x1", 1, 1, firstWidth, secondWidth );
        kernel.SetParam( firstHandle, 0 );
        kernel.SetParam( firstHeight, 1 );
        kernel.SetParam( firstWidth, 2 );
        kernel.SetParam( secondHandle, 3 );
        kernel.SetParam( secondWidth, 4 );
        kernel.SetParam( resultHandle, 5 );
        ASSERT_EXPR( kernel.Run() );
    }
}


void CMetalMathEngine::MultiplyTransposedMatrixByMatrix(int batchSize, const CConstFloatHandle& firstHandle, int firstHeight, int firstWidth,
    const CConstFloatHandle& secondHandle, int secondWidth, const CFloatHandle& resultHandle, int)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C3DKernel kernel( *queue, "cubeKernelBatchMultiplyTransposedMatrixByMatrix", 1, 1, 1, batchSize, firstWidth, secondWidth );
    kernel.SetParam( batchSize, 0 );
    kernel.SetParam( firstHandle, 1 );
    kernel.SetParam( firstHeight, 2 );
    kernel.SetParam( firstWidth, 3 );
    kernel.SetParam( secondHandle, 4 );
    kernel.SetParam( secondWidth, 5 );
    kernel.SetParam( resultHandle, 6 );
    kernel.Run();
}

void CMetalMathEngine::VectorFindMaxValueInSet(const CConstFloatHandle* vectors, int vectorCount, const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    if( vectorCount > 0 ) {
        VectorCopy( resultHandle, vectors[0], vectorSize );
    }
    
    for( int i = 1; i < vectorCount; i++ ) {
        ASSERT_EXPR( vectors[i].GetMathEngine() == this );

        C1DKernel kernel( *queue, "vectorKernelFindMaxValueInSet", 1, vectorSize );
        kernel.SetParam( resultHandle, 0 );
        kernel.SetParam( vectors[i], 1 );
        kernel.SetParam( vectorSize, 2 );
        ASSERT_EXPR( kernel.Run() );
    }
}

void CMetalMathEngine::VectorFindMaxValueInSet(const CConstFloatHandle* vectors, int vectorCount,
    const CFloatHandle& resultHandle, const CIntHandle& indexHandle, int vectorSize)
{
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( indexHandle.GetMathEngine() == this );     

    if( vectorCount > 0 ) {
        VectorCopy( resultHandle, vectors[0], vectorSize );
        VectorFill( indexHandle, (int)0, vectorSize );
    }

    for( int i = 1; i < vectorCount; i++ ) {
        ASSERT_EXPR( vectors[i].GetMathEngine() == this );

        C1DKernel kernel( *queue, "vectorKernelFindMaxValueInSetWithIndices", 1, vectorSize );
        kernel.SetParam( resultHandle, 0 );
        kernel.SetParam( indexHandle, 1 );
        kernel.SetParam( vectors[i], 2 );
        kernel.SetParam( i, 3 );
        kernel.SetParam( vectorSize, 4 );
        ASSERT_EXPR( kernel.Run() );
    }
}

void CMetalMathEngine::VectorSpreadValues(const CConstFloatHandle& sourceHandle, CFloatHandle* vectors, int vectorCount,
	const CConstIntHandle& indexHandle, int vectorSize)
{
    ASSERT_EXPR( sourceHandle.GetMathEngine() == this );
	ASSERT_EXPR( indexHandle.GetMathEngine() == this );

    for( int i = 0; i < vectorCount; i++ ) {
        ASSERT_EXPR( vectors[i].GetMathEngine() == this );

        C1DKernel kernel( *queue, "vectorKernelVectorSpreadValues", 1, vectorSize );
        kernel.SetParam( sourceHandle, 0 );
        kernel.SetParam( indexHandle, 1 );
        kernel.SetParam( vectors[i], 2 );
        kernel.SetParam( i, 3 );
        kernel.SetParam( vectorSize, 4 );
        ASSERT_EXPR( kernel.Run() );
    }
}

void CMetalMathEngine::MultiplyDiagMatrixByMatrix(const CConstFloatHandle& firstHandle, int firstSize,
	const CConstFloatHandle& secondHandle, int secondWidth,
	const CFloatHandle& resultHandle, int)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C2DKernel kernel( *queue, "matrixKernelMultiplyDiagMatrixByMatrix", 1, 1, firstSize, secondWidth );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( firstSize, 1 );
    kernel.SetParam( secondHandle, 2 );
    kernel.SetParam( secondWidth, 3 );
    kernel.SetParam( resultHandle, 4 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::Multiply1DiagMatrixByMatrix(int batchSize, const CConstFloatHandle& firstHandle, int firstSize,
	const CConstFloatHandle& secondHandle, int secondWidth,
	const CFloatHandle& resultHandle, int)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C2DKernel kernel( *queue, "matrixKernelMultiply1DiagMatrixByMatrix", 8, 1, batchSize, firstSize * secondWidth );
    kernel.SetParam( batchSize, 0 );
    kernel.SetParam( firstHandle, 1 );
    kernel.SetParam( firstSize, 2 );
    kernel.SetParam( secondHandle, 3 );
    kernel.SetParam( secondWidth, 4 );
    kernel.SetParam( resultHandle, 5 );
    kernel.SetParam( kernel.GetGridHeight(), 6 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::MultiplyMatrixByDiagMatrix(const CConstFloatHandle& firstHandle, int firstHeight, int firstWidth,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C2DKernel kernel( *queue, "matrixKernelMultiplyMatrixByDiagMatrix", 1, 1, firstHeight, firstWidth );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( firstHeight, 1 );
    kernel.SetParam( firstWidth, 2 );
    kernel.SetParam( secondHandle, 3 );
    kernel.SetParam( resultHandle, 4 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::TransposeMatrix(int batchSize, const CConstFloatHandle& firstHandle,
	int height, int medium, int width, int channels, const CFloatHandle& resultHandle, int resultBufferSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    const int size = batchSize * height * medium * width * channels;
    ASSERT_EXPR( resultBufferSize >= size );
    
    C1DKernel kernel( *queue, "vectorKernelTransposeMatrixFloat", 8, size );
    kernel.SetParam( batchSize, 0 );
    kernel.SetParam( firstHandle, 1 );
    kernel.SetParam( height, 2 );
    kernel.SetParam( medium, 3 );
    kernel.SetParam( width, 4 );
    kernel.SetParam( channels, 5 );
    kernel.SetParam( resultHandle, 6 );
    kernel.SetParam( size, 7 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::TransposeMatrix(int batchSize, const CConstIntHandle& firstHandle,
    int height, int medium, int width, int channels, const CIntHandle& resultHandle, int resultBufferSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    const int size = batchSize * height * medium * width * channels;
    ASSERT_EXPR( resultBufferSize >= size );
    
    C1DKernel kernel( *queue, "vectorKernelTransposeMatrixInt", 8, size );
    kernel.SetParam( batchSize, 0 );
    kernel.SetParam( firstHandle, 1 );
    kernel.SetParam( height, 2 );
    kernel.SetParam( medium, 3 );
    kernel.SetParam( width, 4 );
    kernel.SetParam( channels, 5 );
    kernel.SetParam( resultHandle, 6 );
    kernel.SetParam( size, 7 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::MatrixSpreadRows(const CConstFloatHandle& sourceHandle, int height, int width,
	const CFloatHandle& resultHandle, int resultHeight, const CConstIntHandle& indexHandle,
	const CConstFloatHandle& fillValue)
{
    ASSERT_EXPR( sourceHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( indexHandle.GetMathEngine() == this );
	ASSERT_EXPR( fillValue.GetMathEngine() == this ); 

    if( fillValue.IsNull() ) {
        VectorFill(resultHandle, 0.0f, resultHeight * width);
    } else {
        VectorFill(resultHandle, resultHeight * width, fillValue);
    }
    
    C2DKernel kernel( *queue, "matrixKernelMatrixSpreadRowsFloat", 1, 16, height, width );
    kernel.SetParam( sourceHandle, 0 );
    kernel.SetParam( height, 1 );
    kernel.SetParam( width, 2 );
    kernel.SetParam( resultHandle, 3 );
    kernel.SetParam( indexHandle, 4 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::MatrixSpreadRows(const CConstIntHandle& sourceHandle, int height, int width,
	const CIntHandle& resultHandle, int resultHeight, const CConstIntHandle& indexHandle,
	const CConstIntHandle& fillValue)
{
    ASSERT_EXPR( sourceHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( indexHandle.GetMathEngine() == this );
	ASSERT_EXPR( fillValue.GetMathEngine() == this ); 

    if( fillValue.IsNull() ) {
        VectorFill(resultHandle, 0, resultHeight * width);
    } else {
        VectorFill(resultHandle, resultHeight * width, fillValue);
    }
    
    C2DKernel kernel( *queue, "matrixKernelMatrixSpreadRowsInt", 1, 16, height, width );
    kernel.SetParam( sourceHandle, 0 );
    kernel.SetParam( height, 1 );
    kernel.SetParam( width, 2 );
    kernel.SetParam( resultHandle, 3 );
    kernel.SetParam( indexHandle, 4 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::MatrixSpreadRowsAdd(const CConstFloatHandle& sourceHandle, int height, int width,
	const CFloatHandle& resultHandle, int resultHeight, const CConstIntHandle& indexHandle)
{
    ASSERT_EXPR( sourceHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( indexHandle.GetMathEngine() == this );

    C2DKernel kernel( *queue, "matrixKernelMatrixSpreadRowsAdd", 1, 1, resultHeight, width );
    kernel.SetParam( sourceHandle, 0 );
    kernel.SetParam( height, 1 );
    kernel.SetParam( width, 2 );
    kernel.SetParam( resultHandle, 3 );
    kernel.SetParam( resultHeight, 4 );
    kernel.SetParam( indexHandle, 5 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::MultiplyDiagMatrixByMatrixAndAdd( int batchSize, const CConstFloatHandle& firstHandle,
	int firstSize, const CConstFloatHandle& secondHandle, int secondWidth, const CFloatHandle& resultHandle )
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C2DKernel kernel( *queue, "matrixKernelMultiplyDiagMatrixByMatrixAndAdd", 1, 1, firstSize, secondWidth );
    kernel.SetParam( batchSize, 0 );
    kernel.SetParam( firstHandle, 1 );
    kernel.SetParam( firstSize, 2 );
    kernel.SetParam( secondHandle, 3 );
    kernel.SetParam( secondWidth, 4 );
    kernel.SetParam( resultHandle, 5 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::MatrixLogSumExpByRows(const CConstFloatHandle& matrix, int height, int width,
	const CFloatHandle& result, int resultSize)
{
    ASSERT_EXPR( matrix.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );
	ASSERT_EXPR( resultSize >= height );

    C2DKernel kernel( *queue, "matrixKernelMatrixLogSumExpByRows", 1, 1, height, width );
    kernel.SetParam( matrix, 0 );
    kernel.SetParam( height, 1 );
    kernel.SetParam( width, 2 );
    kernel.SetParam( result, 3 );
    kernel.SetSharedParam( kernel.GetThreadCount() * sizeof(float), 4 );
    
    // threadgroupCount.width = 1;
    ASSERT_EXPR( kernel.Run( 0, 0, 1 ) );
}

void CMetalMathEngine::MatrixLogSumExpByColumns(const CConstFloatHandle& matrix, int height, int width,
	const CFloatHandle& result, int resultSize)
{
    ASSERT_EXPR( matrix.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );
	ASSERT_EXPR( resultSize >= width );

    C2DKernel kernel( *queue, "matrixKernelMatrixLogSumExpByColumns", 1, 1, height, width );
    kernel.SetParam( matrix, 0 );
    kernel.SetParam( height, 1 );
    kernel.SetParam( width, 2 );
    kernel.SetParam( result, 3 );
    kernel.SetSharedParam( kernel.GetThreadCount() * sizeof(float), 4 );
    
    // threadgroupCount.height = 1;
    ASSERT_EXPR( kernel.Run( 0, 1, 0 ) );
}

void CMetalMathEngine::MatrixSoftmaxByRows(const CConstFloatHandle& matrix, int height, int width,
	const CFloatHandle& result)
{
    ASSERT_EXPR( matrix.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );

    C2DKernel kernel( *queue, "matrixKernelMatrixSoftmaxByRows", 1, 1, height, width );
    kernel.SetParam( matrix, 0 );
    kernel.SetParam( height, 1 );
    kernel.SetParam( width, 2 );
    kernel.SetParam( result, 3 );
    kernel.SetSharedParam( kernel.GetThreadCount() * sizeof(float), 4 );
    
    // threadgroupCount.width = 1;
    ASSERT_EXPR( kernel.Run( 0, 0, 1 ) );
}

void CMetalMathEngine::MatrixSoftmaxDiffOpByRows(const CConstFloatHandle& first, const CConstFloatHandle& second,
	int height, int width, const CFloatHandle& result)
{
    ASSERT_EXPR( first.GetMathEngine() == this );
	ASSERT_EXPR( second.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );

    C2DKernel kernel( *queue, "matrixKernelMatrixSoftmaxDiffOpByRows", 1, 1, height, width );
    kernel.SetParam( first, 0 );
    kernel.SetParam( second, 1 );
    kernel.SetParam( height, 2 );
    kernel.SetParam( width, 3 );
    kernel.SetParam( result, 4 );
    kernel.SetSharedParam( kernel.GetThreadCount() * sizeof(float), 5 );
    
    // threadgroupCount.width = 1;
    ASSERT_EXPR( kernel.Run( 0, 0, 1 ) );
}

void CMetalMathEngine::MatrixSoftmaxByColumns(const CConstFloatHandle& matrix, int height, int width,
	const CFloatHandle& result)
{
    ASSERT_EXPR( matrix.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );

    C2DKernel kernel( *queue, "matrixKernelMatrixSoftmaxByColumns", 1, 1, height, width );
    kernel.SetParam( matrix, 0 );
    kernel.SetParam( height, 1 );
    kernel.SetParam( width, 2 );
    kernel.SetParam( result, 3 );
    kernel.SetSharedParam( kernel.GetThreadCount() * sizeof(float), 4 );
    
    // threadgroupCount.height = 1;
    ASSERT_EXPR( kernel.Run( 0, 1, 0 ) );
}

void CMetalMathEngine::MatrixSoftmaxDiffOpByColumns(const CConstFloatHandle& first, const CConstFloatHandle& second,
	int height, int width, const CFloatHandle& result)
{
    ASSERT_EXPR( first.GetMathEngine() == this );
	ASSERT_EXPR( second.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );

    C2DKernel kernel( *queue, "matrixKernelMatrixSoftmaxDiffOpByColumns", 1, 1, height, width );
    kernel.SetParam( first, 0 );
    kernel.SetParam( second, 1 );
    kernel.SetParam( height, 2 );
    kernel.SetParam( width, 3 );
    kernel.SetParam( result, 4 );
    kernel.SetSharedParam( kernel.GetThreadCount() * sizeof(float), 5 );
    
    // threadgroupCount.height = 1;
    ASSERT_EXPR( kernel.Run( 0, 1, 0 ) );
}

void CMetalMathEngine::MultiplyLookupMatrixByLookupVector(int batchSize, const CLookupMatrix& matrix,
	const CLookupVector& vector, const CFloatHandle& resultHandle, int resultSize)
{
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( matrix.Width() == vector.VectorSize() );
	ASSERT_EXPR( resultSize >= batchSize * matrix.Height() );

	VectorFill( resultHandle, 0.0f, batchSize * matrix.Height() );
    
    C2DKernel kernel( *queue, "matrixKernelMultiplyLookupMatrixByLookupVector",
        1, 1, batchSize * matrix.Height(), matrix.Width() );
    kernel.SetParam( batchSize, 0 );
    kernel.SetParam( matrix.Table, 1 );
    kernel.SetParam( matrix.Height(), 2 );
    kernel.SetParam( matrix.Width(), 3 );
    kernel.SetParam( matrix.Rows, 4 );
    kernel.SetParam( vector.Table, 5 );
    kernel.SetParam( vector.Vector, 6 );
    kernel.SetParam( resultHandle, 7 );
    kernel.SetSharedParam( kernel.GetThreadCount() * sizeof(float), 8 );
    
    // threadgroupCount.width = 1;
    ASSERT_EXPR( kernel.Run( 0, 0, 1 ) );
}

void CMetalMathEngine::MultiplyTransposedLookupMatrixByVector(int batchSize, const CLookupMatrix& matrix,
	const CConstFloatHandle& vectorHandle, const CFloatHandle& resultHandle, int resultSize)
{
 	ASSERT_EXPR( vectorHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultSize >= batchSize * matrix.Width() );

    VectorFill( resultHandle, 0.0f, batchSize * matrix.Width() );
    
    MultiplyTransposedLookupMatrixByVectorAndAdd( batchSize, matrix, vectorHandle, resultHandle, resultSize );
}

void CMetalMathEngine::MultiplyTransposedLookupMatrixByVectorAndAdd(int batchSize, const CLookupMatrix& matrix,
	const CConstFloatHandle& vectorHandle, const CFloatHandle& resultHandle, int /*resultSize*/)
{
    ASSERT_EXPR( vectorHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C2DKernel kernel( *queue, "matrixKernelMultiplyTransposedLookupMatrixByVector",
        1, 1, batchSize * matrix.Width(), matrix.Height() );
    kernel.SetParam( batchSize, 0 );
    kernel.SetParam( matrix.Table, 1 );
    kernel.SetParam( matrix.Height(), 2 );
    kernel.SetParam( matrix.Width(), 3 );
    kernel.SetParam( matrix.Rows, 4 );
    kernel.SetParam( vectorHandle, 5 );
    kernel.SetParam( resultHandle, 6 );
    kernel.SetSharedParam( kernel.GetThreadCount() * sizeof(float), 7 );
    
    // threadgroupCount.width = 1;
    ASSERT_EXPR( kernel.Run( 0, 0, 1 ) );
}

void CMetalMathEngine::MultiplyVectorByTransposedLookupVectorAndAddToTable(int batchSize,
	const CFloatHandle& table, int vectorCount, int vectorSize, const CConstIntHandle& indexHandle,
	const CConstFloatHandle& firstHandle, int firstSize, const CLookupVector& second)
{
    ASSERT_EXPR( table.GetMathEngine() == this );
	ASSERT_EXPR( indexHandle.GetMathEngine() == this );
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );

    C2DKernel kernel( *queue, "matrixKernelMultiplyVectorByTransposedLookupVectorAndAddToTable", 1, 1, vectorCount, vectorSize );
    kernel.SetParam( batchSize, 0 );
    kernel.SetParam( table, 1 );
    kernel.SetParam( vectorCount, 2 );
    kernel.SetParam( vectorSize, 3 );
    kernel.SetParam( indexHandle, 4 );
    kernel.SetParam( firstHandle, 5 );
    kernel.SetParam( firstSize, 6 );
    kernel.SetParam( second.Table, 7 );
    kernel.SetParam( second.Vector, 8 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::AddVectorToMatrixRows(int batchSize, const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
	int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle)
{
    ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorHandle.GetMathEngine() == this );

    C2DKernel kernel( *queue, "matrixKernelAddVectorToMatrixRows", 1, 4, batchSize * matrixHeight, matrixWidth );
    kernel.SetParam( batchSize, 0 );
    kernel.SetParam( matrixHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( matrixHeight, 3 );
    kernel.SetParam( matrixWidth, 4 );
    kernel.SetParam( vectorHandle, 5 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::SumMatrixRowsAdd( int batchSize, const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
    int matrixHeight, int matrixWidth )
{
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );

    C2DKernel kernel( *queue, "matrixKernelSumMatrixRowsAdd", 1, 1, batchSize, matrixWidth );
    kernel.SetParam( batchSize, 0 );
    kernel.SetParam( matrixHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( matrixHeight, 3 );
    kernel.SetParam( matrixWidth, 4 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::SumMatrixRows( int batchSize, const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
	int matrixHeight, int matrixWidth )
{
	VectorFill(resultHandle, 0.f, batchSize * matrixWidth);
	SumMatrixRowsAdd(batchSize, resultHandle, matrixHandle, matrixHeight, matrixWidth);
}

} // namespace NeoML

#endif // NEOML_USE_METAL
