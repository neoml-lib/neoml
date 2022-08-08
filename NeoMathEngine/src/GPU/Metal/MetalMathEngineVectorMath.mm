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

namespace NeoML {

// The number of combined values for vector kernels
static const int VectorCombineCount = 8;

void CMetalMathEngine::VectorFill(const CFloatHandle& result, float value, int vectorSize)
{
    ASSERT_EXPR( result.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelFillFloat", VectorCombineCount, vectorSize );
    kernel.SetParam( result, 0 );
    kernel.SetParam( value, 1 );
    kernel.SetParam( vectorSize, 2 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorFill(const CIntHandle& result, int value, int vectorSize)
{
    ASSERT_EXPR( result.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelFillInt", VectorCombineCount, vectorSize );
    kernel.SetParam( result, 0 );
    kernel.SetParam( value, 1 );
    kernel.SetParam( vectorSize, 2 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorFill(const CFloatHandle& result, int vectorSize, const CConstFloatHandle& value)
{
    ASSERT_EXPR( result.GetMathEngine() == this );
    ASSERT_EXPR( value.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelFillFloat", VectorCombineCount, vectorSize );
    kernel.SetParam( result, 0 );
    kernel.SetParam( value, 1 );
    kernel.SetParam( vectorSize, 2 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorFill(const CIntHandle& result, int vectorSize, const CConstIntHandle& value)
{
    ASSERT_EXPR( result.GetMathEngine() == this );
    ASSERT_EXPR( value.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelFillInt", VectorCombineCount, vectorSize );
    kernel.SetParam( result, 0 );
    kernel.SetParam( value, 1 );
    kernel.SetParam( vectorSize, 2 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorConvert(const CConstFloatHandle& from, const CIntHandle& to, int vectorSize)
{
    ASSERT_EXPR( from.GetMathEngine() == this );
    ASSERT_EXPR( to.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelConvertFloatToInt", VectorCombineCount, vectorSize );
    kernel.SetParam( from, 0 );
    kernel.SetParam( to, 1 );
    kernel.SetParam( vectorSize, 2 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorConvert(const CConstIntHandle& from, const CFloatHandle& to, int vectorSize)
{
    ASSERT_EXPR( from.GetMathEngine() == this );
    ASSERT_EXPR( to.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelConvertIntToFloat", VectorCombineCount, vectorSize );
    kernel.SetParam( from, 0 );
    kernel.SetParam( to, 1 );
    kernel.SetParam( vectorSize, 2 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::BroadcastCopy(const CIntHandle&, const CConstIntHandle&,
	const CBlobDesc&, const CBlobDesc&, int)
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::BroadcastCopy(const CFloatHandle&, const CConstFloatHandle&,
	const CBlobDesc&, const CBlobDesc&, int)
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::VectorFillBernoulli(const CFloatHandle& result, float p, int vectorSize, float value, int seed)
{
    ASSERT_EXPR( result.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelVectorFillBernoulli", VectorCombineCount, ( vectorSize + 3 ) / 4 );
    kernel.SetParam( result, 0 );
    kernel.SetParam( p, 1 );
    kernel.SetParam( vectorSize, 2 );
    kernel.SetParam( value, 3 );
    kernel.SetParam( seed, 4 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorDotProduct( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    int vectorSize, const CFloatHandle& resultHandle )
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelVectorDotProduct", 1, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    kernel.SetParam( resultHandle, 3 );
    kernel.SetSharedParam( kernel.GetThreadCount() * sizeof( float ), 4 );
    
    // threadgroupCount.width = 1;
    ASSERT_EXPR( kernel.Run( 0, 0, 1 ) );
}

void CMetalMathEngine::AddVectorToMatrixElements(const CFloatHandle& matrix, int height, int width,
	const CConstIntHandle& indices, const CConstFloatHandle& vector)
{
    ASSERT_EXPR( matrix.GetMathEngine() == this );
    ASSERT_EXPR( indices.GetMathEngine() == this );
    ASSERT_EXPR( vector.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelAddVectorToMatrixElements", VectorCombineCount, height );
    kernel.SetParam( matrix, 0 );
    kernel.SetParam( height, 1 );
    kernel.SetParam( width, 2 );
    kernel.SetParam( indices, 3 );
    kernel.SetParam( vector, 4 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::AddVectorToMatrixElements(const CFloatHandle& matrix, int height, int width,
	const CConstIntHandle& rowIndices, const CConstIntHandle& columnIndices,
	const CConstFloatHandle& vector, int vectorSize)
{
    ASSERT_EXPR( matrix.GetMathEngine() == this );
    ASSERT_EXPR( rowIndices.GetMathEngine() == this );
    ASSERT_EXPR( columnIndices.GetMathEngine() == this );    
    ASSERT_EXPR( vector.GetMathEngine() == this );

    C2DKernel kernel( *queue, "matrixKernelAddVectorToMatrixElementsEx", 1, 1, ( height + 7 ) / 8, ( width + 7 ) / 8 );
    kernel.SetParam( matrix, 0 );
    kernel.SetParam( height, 1 );
    kernel.SetParam( width, 2 );
    kernel.SetParam( rowIndices, 3 );
    kernel.SetParam( columnIndices, 4 );
    kernel.SetParam( vector, 5 );
    kernel.SetParam( vectorSize, 6 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::FilterSmallValues(const CFloatHandle& data, int dataSize, float threshold)
{
    ASSERT_EXPR( data.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorFilterSmallValues", VectorCombineCount, dataSize );
    kernel.SetParam( data, 0 );
    kernel.SetParam( dataSize, 1 );
    kernel.SetParam( threshold, 2 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorSum(const CConstFloatHandle& firstHandle, int vectorSize, const CFloatHandle& resultHandle)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    VectorFill( resultHandle, 0.0, 1 );
    C1DKernel kernel( *queue, "vectorKernelSum", 1, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( vectorSize, 1 );
    kernel.SetParam( 0, 2 );
    kernel.SetParam( resultHandle, 3 );
    kernel.SetSharedParam( kernel.GetThreadCount() * sizeof(float), 4 );

    // threadgroupCount.width = 1
    ASSERT_EXPR( kernel.Run( 0, 0, 1 ) );
}

void CMetalMathEngine::VectorSumAdd(const CConstFloatHandle& firstHandle, int vectorSize, const CFloatHandle& resultHandle)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelSum", 1, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( vectorSize, 1 );
    kernel.SetParam( 0, 2 );
    kernel.SetParam( resultHandle, 3 );
    kernel.SetSharedParam( kernel.GetThreadCount() * sizeof(float), 4 );

    // threadgroupCount.width = 1
    ASSERT_EXPR( kernel.Run( 0, 0, 1 ) );
}

void CMetalMathEngine::VectorSumAlongDimension( const CConstFloatHandle&, int, int, int, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::VectorSumAlongDimensionDiag(const CConstFloatHandle&, int, int,
	int, const CFloatHandle&)
{
	ASSERT_EXPR(false);
}

void CMetalMathEngine::VectorCumSumAlongDimension( const CConstFloatHandle&, int, int, int, const CFloatHandle&, bool )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::VectorCumSumAlongDimension( const CConstIntHandle&, int, int, int, const CIntHandle&, bool )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::VectorCumSumAlongDimensionDiag(const CConstFloatHandle&, int, int,
	int, const CFloatHandle&)
{
	ASSERT_EXPR(false);
}

void CMetalMathEngine::VectorEqual( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize )
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelEqual", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    kernel.SetParam( resultHandle, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorEqualValue( const CConstIntHandle& firstHandle,
    const CFloatHandle& resultHandle, int vectorSize, const CConstIntHandle& valueHandle )
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( valueHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelEqualValue", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( valueHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    kernel.SetParam( resultHandle, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorELU( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
    int vectorSize, const CConstFloatHandle& alpha )
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( alpha.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelELU", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    kernel.SetParam( alpha, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorELUDiff( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& alpha )
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );
    ASSERT_EXPR( alpha.GetMathEngine() == this );    

    C1DKernel kernel( *queue, "vectorKernelELUDiff", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    kernel.SetParam( alpha, 4 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorELUDiffOp( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& alpha )
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );
    ASSERT_EXPR( alpha.GetMathEngine() == this );    

    C1DKernel kernel( *queue, "vectorKernelELUDiffOp", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    kernel.SetParam( alpha, 4 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorReLU(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize,
    const CConstFloatHandle& upperThresholdHandle )
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );
    ASSERT_EXPR( upperThresholdHandle.GetMathEngine() == this );    

    if( vectorSize >= 16 ) {
        C1DKernel kernel( *queue, "vectorKernelReLUFloat4", 16, vectorSize - vectorSize % 16 );
        kernel.SetParam( firstHandle, 0 );
        kernel.SetParam( resultHandle, 1 );
        kernel.SetParam( vectorSize / 16, 2 );
        kernel.SetParam( upperThresholdHandle, 3 );
        ASSERT_EXPR( kernel.Run() );
    }
    if( vectorSize % 16 > 0 ) {
        int offset = vectorSize - vectorSize % 16;
        C1DKernel kernel( *queue, "vectorKernelReLUFloat", 1, vectorSize % 16 );
        kernel.SetParam( firstHandle + offset, 0 );
        kernel.SetParam( resultHandle + offset, 1 );
        kernel.SetParam( vectorSize % 16, 2 );
        kernel.SetParam( upperThresholdHandle, 3 );
        ASSERT_EXPR( kernel.Run() );
    }
}

void CMetalMathEngine::VectorReLUDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& upperThresholdHandle)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );
    ASSERT_EXPR( upperThresholdHandle.GetMathEngine() == this );   

    C1DKernel kernel( *queue, "vectorKernelReLUDiff", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    kernel.SetParam( upperThresholdHandle, 4 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorLeakyReLU( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
    int vectorSize, const CConstFloatHandle& alpha )
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );
    ASSERT_EXPR( alpha.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelLeakyReLU", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    kernel.SetParam( alpha, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorLeakyReLUDiff( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& alpha )
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );
    ASSERT_EXPR( alpha.GetMathEngine() == this );   

    C1DKernel kernel( *queue, "vectorKernelLeakyReLUDiff", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    kernel.SetParam( alpha, 4 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorHSwish( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, 
		int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this ); 

    C1DKernel kernel( *queue, "vectorKernelHSwish", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorHSwishDiff( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize ) 
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelHSwishDiff", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorEltwiseMax(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelEltwiseMax", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorEltwiseMin(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelEltwiseMin", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorAbs(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelAbs", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorAbsDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelAbsDiff", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorHinge(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelHinge", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorHingeDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelHingeDiff", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorSquaredHinge(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelSquaredHinge", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorSquaredHingeDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelSquaredHingeDiff", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorHuber(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelHuber", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorHuberDerivative(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelHuberDiff", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorHardTanh(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelHardTanh", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorHardTanhDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelHardTanhDiff", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorHardSigmoid(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize, 
	const CConstFloatHandle& slopeHandle, const CConstFloatHandle& biasHandle)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelHardSigmoid", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
	kernel.SetParam( slopeHandle, 3 );
	kernel.SetParam( biasHandle, 4 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorHardSigmoidDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& slopeHandle, const CConstFloatHandle& biasHandle)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelHardSigmoidDiff", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
	kernel.SetParam( slopeHandle, 4 );
	kernel.SetParam( biasHandle, 5 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorHardSigmoidDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& slopeHandle, const CConstFloatHandle& /*biasHandle*/)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelHardSigmoidDiffOp", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
	kernel.SetParam( slopeHandle, 4 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorExp(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelExp", 1, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorLog( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize )
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelLog", 1, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorNegLog(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelNegLog", 1, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorErf( const CConstFloatHandle&, const CFloatHandle&, int )
{
    ASSERT_EXPR( false );
}

void CMetalMathEngine::VectorBernulliKLDerivative(const CConstFloatHandle& firstHandle,
    const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& targetHandle)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( targetHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelBernulliKLDerivative", 1, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    kernel.SetParam( targetHandle, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorAdd(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelAddFloat", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorAddValue(const CConstFloatHandle& firstHandle,
    const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& additionHandle)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );
    ASSERT_EXPR( additionHandle.GetMathEngine() == this );   

    C1DKernel kernel( *queue, "vectorKernelAddValueFloat", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    kernel.SetParam( additionHandle, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorAdd(const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
    const CIntHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelAddInt", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorAddValue(const CConstIntHandle& firstHandle,
    const CIntHandle& resultHandle, int vectorSize, const CConstIntHandle& additionHandle)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( additionHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelAddValueInt", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    kernel.SetParam( additionHandle, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorSub(const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
    const CIntHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelSubInt", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorSub(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelSubFloat", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorMultiply(const CConstFloatHandle& firstHandle,
    const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multiplierHandle)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( multiplierHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelMultiply", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    kernel.SetParam( multiplierHandle, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorMultiply(const CConstFloatHandle&, const CFloatHandle&, int, const CConstFloatHandle&)
{
    ASSERT_EXPR( false );
}

void CMetalMathEngine::VectorNegMultiply(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
    int vectorSize, const CConstFloatHandle& multiplierHandle)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( multiplierHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelNegMultiply", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    kernel.SetParam( multiplierHandle, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorEltwiseMultiply(const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
    const CIntHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelEltwiseMultiplyInt", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorEltwiseMultiply(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelEltwiseMultiplyFloat", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorEltwiseMultiplyAdd(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelEltwiseMultiplyAdd", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorEltwiseNegMultiply(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelEltwiseNegMultiply", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorEltwiseDivide(const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
    const CIntHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelEltwiseDivideInt", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorEltwiseDivide(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelEltwiseDivideFloat", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorEltwisePower(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelEltwisePower", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorSqrt(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelSqrt", 1, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorInv(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );
    
    C1DKernel kernel( *queue, "vectorKernelInv", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorMinMax(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize,
    const CConstFloatHandle& minHandle, const CConstFloatHandle& maxHandle)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );
    ASSERT_EXPR( minHandle.GetMathEngine() == this );
    ASSERT_EXPR( maxHandle.GetMathEngine() == this );   

    C1DKernel kernel( *queue, "vectorKernelMinMax", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    kernel.SetParam( minHandle, 3 );
    kernel.SetParam( maxHandle, 4 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorSigmoid(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelSigmoid", 1, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorSigmoidDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelSigmoidDiff", 1, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorSigmoidDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelSigmoidDiffOp", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorTanh(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelTanh", 1, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( resultHandle, 1 );
    kernel.SetParam( vectorSize, 2 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorTanhDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelTanhDiff", 1, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorTanhDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );
 
    C1DKernel kernel( *queue, "vectorKernelTanhDiffOp", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorPower(float exponent, const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelPower", 1, vectorSize );
    kernel.SetParam( exponent, 0 );
    kernel.SetParam( firstHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorPowerDiff(float exponent, const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelPowerDiff", 1, vectorSize );
    kernel.SetParam( exponent, 0 );
    kernel.SetParam( firstHandle, 1 );
    kernel.SetParam( secondHandle, 2 );
    kernel.SetParam( resultHandle, 3 );
    kernel.SetParam( vectorSize, 4 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorPowerDiffOp(float exponent, const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelPowerDiffOp", 1, vectorSize );
    kernel.SetParam( exponent, 0 );
    kernel.SetParam( firstHandle, 1 );
    kernel.SetParam( secondHandle, 2 );
    kernel.SetParam( resultHandle, 3 );
    kernel.SetParam( vectorSize, 4 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorL1DiffAdd(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& hubertThresholdHandle, const CConstFloatHandle& multHandle)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );
    ASSERT_EXPR( hubertThresholdHandle.GetMathEngine() == this );   
    ASSERT_EXPR( multHandle.GetMathEngine() == this );       

    C1DKernel kernel( *queue, "vectorKernelL1DiffAdd", VectorCombineCount, vectorSize );
    kernel.SetParam( firstHandle, 0 );
    kernel.SetParam( secondHandle, 1 );
    kernel.SetParam( resultHandle, 2 );
    kernel.SetParam( vectorSize, 3 );
    kernel.SetParam( hubertThresholdHandle, 4 );
    kernel.SetParam( multHandle, 5 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::VectorMax( const CConstFloatHandle& firstHandle, float secondValue, const CFloatHandle& resultHandle,
	int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorSize > 0 );
	ASSERT_EXPR( false );
	secondValue;	
}

void CMetalMathEngine::VectorMaxDiff( const CConstFloatHandle& firstHandle, float secondValue, const CFloatHandle& gradHandle,
	int gradHeight, int gradWidth )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( gradHandle.GetMathEngine() == this );
	ASSERT_EXPR( gradHeight > 0 );
	ASSERT_EXPR( gradWidth > 0 );
	ASSERT_EXPR( false );
	secondValue;	
}

void CMetalMathEngine::VectorNeg(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorSize >= 0 );
	ASSERT_EXPR( false );
}

void CMetalMathEngine::VectorLogDiff( const CConstFloatHandle& sourceGradHandle, int sourceGradHeight, int sourceGradWidth,
	const CConstFloatHandle& valueHandle, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( sourceGradHandle.GetMathEngine() == this );
	ASSERT_EXPR( valueHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( sourceGradHeight > 0 );
	ASSERT_EXPR( sourceGradWidth > 0 );
	ASSERT_EXPR( false );
}

void CMetalMathEngine::VectorSub(const CConstFloatHandle& firstHandle, float second, const CFloatHandle& resultHandle,
	int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorSize > 0 );
	ASSERT_EXPR( false );
	second;
}

void CMetalMathEngine::VectorSub(float first, const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle,
	int vectorSize)
{
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorSize > 0 );
	ASSERT_EXPR( false );
	first;
}

void CMetalMathEngine::VectorTopK(const CConstFloatHandle& first, int firstSize, int k, const CFloatHandle& result,
	const CIntHandle& indices)
{
	ASSERT_EXPR( first.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );
	ASSERT_EXPR( indices.GetMathEngine() == this );
	ASSERT_EXPR( firstSize > 0 );
	ASSERT_EXPR( k > 0 );
	ASSERT_EXPR( false );
}

void CMetalMathEngine::VectorTopKDiff(const CConstFloatHandle& sourceGrad, int sourceGradHeight, int sourceGradWidth,
	const CConstIntHandle& indices, int k, const CFloatHandle& resultGrad)
{
	ASSERT_EXPR( sourceGrad.GetMathEngine() == this );
	ASSERT_EXPR( resultGrad.GetMathEngine() == this );
	ASSERT_EXPR( indices.GetMathEngine() == this );
	ASSERT_EXPR( sourceGradHeight > 0 );
	ASSERT_EXPR( sourceGradWidth > 0 );
	ASSERT_EXPR( k > 0 );
	ASSERT_EXPR( false );
}

void CMetalMathEngine::VectorAbsDiff(const CConstFloatHandle& sourceGradHandle, int gradHeight, int gradWidth,
	const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle)
{
	ASSERT_EXPR( sourceGradHandle.GetMathEngine() == this );
	ASSERT_EXPR( gradHeight > 0 );
	ASSERT_EXPR( gradWidth > 0 );
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	ASSERT_EXPR( false );
}

void CMetalMathEngine::VectorMinMaxDiff(const CConstFloatHandle& sourceGradHandle, int gradHeight, int gradWidth,
	const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	const CConstFloatHandle& minHandle, const CConstFloatHandle& maxHandle)
{
	ASSERT_EXPR( sourceGradHandle.GetMathEngine() == this );
	ASSERT_EXPR( gradHeight > 0 );
	ASSERT_EXPR( gradWidth > 0 );
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( minHandle.GetMathEngine() == this );
	ASSERT_EXPR( maxHandle.GetMathEngine() == this );

	ASSERT_EXPR( false );
}

void CMetalMathEngine::VectorEltwiseLess( const CConstFloatHandle&, const CConstFloatHandle&,
	const CFloatHandle&, int )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::VectorEltwiseLess( const CConstFloatHandle&, float,
	const CFloatHandle&, int )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::VectorEltwiseLess( float, const CConstFloatHandle&,
	const CFloatHandle&, int )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::VectorEltwiseLess( const CConstFloatHandle&, const CConstFloatHandle&,
	const CIntHandle&, int )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::VectorEltwiseLess( const CConstIntHandle&, const CConstIntHandle&,
	const CIntHandle&, int )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::VectorEltwiseEqual( const CConstFloatHandle&, const CConstFloatHandle&,
	const CIntHandle&, int )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::VectorEltwiseEqual( const CConstIntHandle&, const CConstIntHandle&,
	const CIntHandle&, int )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::VectorEltwiseWhere( const CConstIntHandle&, const CConstFloatHandle&, const CConstFloatHandle&,
	const CFloatHandle&, int )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::VectorEltwiseWhere( const CConstIntHandle&, const CConstIntHandle&, const CConstIntHandle&,
	const CIntHandle&, int )
{
	ASSERT_EXPR( false );
}

} // namespace NeoML

#endif // NEOML_USE_METAL
