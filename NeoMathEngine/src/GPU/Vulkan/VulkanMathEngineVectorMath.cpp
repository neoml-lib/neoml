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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_VULKAN

#include <VulkanMathEngine.h>
#include <MathEngineCommon.h>
#include <MemoryHandleInternal.h>
#include <VulkanCommandQueue.h>
#include <VulkanDll.h>
#include <VulkanShader.h>

namespace NeoML {

// Include the shader code
#include <shaders/generated/VectorFillScalar.h>
#include <shaders/generated/VectorConvertFloatToInt.h>
#include <shaders/generated/VectorConvertIntToFloat.h>
#include <shaders/generated/VectorELU.h>
#include <shaders/generated/VectorELUDiff.h>
#include <shaders/generated/VectorELUDiffOp.h>
#include <shaders/generated/VectorReLU.h>
#include <shaders/generated/VectorReLU4.h>
#include <shaders/generated/VectorReLUDiff.h>
#include <shaders/generated/VectorLeakyReLU.h>
#include <shaders/generated/VectorLeakyReLUDiff.h>
#include <shaders/generated/VectorHSwish.h>
#include <shaders/generated/VectorHSwishDiff.h>
#include <shaders/generated/VectorEltwiseMax.h>
#include <shaders/generated/VectorEltwiseMin.h>
#include <shaders/generated/VectorAbs.h>
#include <shaders/generated/VectorAbsDiff.h>
#include <shaders/generated/VectorHinge.h>
#include <shaders/generated/VectorHingeDiff.h>
#include <shaders/generated/VectorHuber.h>
#include <shaders/generated/VectorHardTanh.h>
#include <shaders/generated/VectorHardTanhDiff.h>
#include <shaders/generated/VectorHardSigmoid.h>
#include <shaders/generated/VectorHardSigmoidDiff.h>
#include <shaders/generated/VectorHardSigmoidDiffOp.h>
#include <shaders/generated/VectorExp.h>
#include <shaders/generated/VectorLog.h>
#include <shaders/generated/VectorBernulliKLDerivative.h>
#include <shaders/generated/VectorSquaredHinge.h>
#include <shaders/generated/VectorSquaredHingeDiff.h>
#include <shaders/generated/VectorAddFloat4.h>
#include <shaders/generated/VectorAddFloat1.h>
#include <shaders/generated/VectorAddValue.h>
#include <shaders/generated/VectorAddInt.h>
#include <shaders/generated/VectorSubInt.h>
#include <shaders/generated/VectorSubFloat.h>
#include <shaders/generated/VectorMultiplyAndAdd.h>
#include <shaders/generated/VectorMultiplyAndSub.h>
#include <shaders/generated/VectorMultiplyInt.h>
#include <shaders/generated/VectorMultiplyFloat.h>
#include <shaders/generated/VectorEltwiseDivideInt.h>
#include <shaders/generated/VectorEltwiseDivideFloat.h>
#include <shaders/generated/VectorEltwisePower.h>
#include <shaders/generated/VectorSqrt.h>
#include <shaders/generated/VectorInv.h>
#include <shaders/generated/VectorMinMax.h>
#include <shaders/generated/VectorSigmoid.h>
#include <shaders/generated/VectorSigmoidDiff.h>
#include <shaders/generated/VectorSigmoidDiffOp.h>
#include <shaders/generated/VectorTanh.h>
#include <shaders/generated/VectorTanhDiff.h>
#include <shaders/generated/VectorTanhDiffOp.h>
#include <shaders/generated/VectorPower.h>
#include <shaders/generated/VectorPowerDiff.h>
#include <shaders/generated/VectorPowerDiffOp.h>
#include <shaders/generated/VectorL1DiffAdd.h>
#include <shaders/generated/VectorDotProduct.h>
#include <shaders/generated/VectorSum.h>
#include <shaders/generated/VectorEqual.h>
#include <shaders/generated/VectorFillBernoulli.h>
#include <shaders/generated/VectorFindMaxValueInSetNoIndices.h>
#include <shaders/generated/VectorFindMaxValueInSet.h>

//------------------------------------------------------------------------------------------------------------

// The number of combined operations
constexpr int VectorCombine = 4;

void CVulkanMathEngine::VectorFill( const CFloatHandle& result, float value, int vectorSize )
{
	static_assert(sizeof(float) == sizeof(uint32_t), "");

	uint32_t data = 0;
	*((float*)(&data)) = value;
	size_t size = vectorSize * sizeof( float );

	CVulkanMemory* vulkanMemory = GetRawAllocation( result );

	std::lock_guard<std::mutex> lock( Mutex );
	commandQueue->RunFillBuffer( vulkanMemory->Buffer(), GetRawOffset( result ), size, data );
}

void CVulkanMathEngine::VectorFill( const CIntHandle& result, int value, int vectorSize )
{
	uint32_t data = value;
	size_t size = vectorSize * sizeof( int );

	CVulkanMemory* vulkanMemory = GetRawAllocation( result );

	std::lock_guard<std::mutex> lock( Mutex );
	commandQueue->RunFillBuffer( vulkanMemory->Buffer(), GetRawOffset(result), size, data );
}

void CVulkanMathEngine::VectorFill(const CFloatHandle& result, int vectorSize, const CConstFloatHandle& value)
{
	CMemoryHandle bufs[2] = { value, result };
	size_t sizes[2] = { sizeof(float), vectorSize * sizeof(float) };

	runVectorShader( shaderLoader->GET_SHADER_DATA(VectorFillScalar, false, 0, 0, 2),
		0, 0, 0, 0, 0, 0, bufs, sizes, 2, Ceil(vectorSize, VectorCombine) );
}

void CVulkanMathEngine::VectorFill(const CIntHandle& result, int vectorSize, const CConstIntHandle& value)
{
	CMemoryHandle bufs[2] = { value, result };
	size_t sizes[2] = { sizeof( int ), vectorSize * sizeof( int ) };

	runVectorShader( shaderLoader->GET_SHADER_DATA(VectorFillScalar, false, 0, 0, 2),
		0, 0, 0, 0, 0, 0, bufs, sizes, 2, Ceil(vectorSize, VectorCombine) );
}

void CVulkanMathEngine::VectorConvert( const CConstFloatHandle& from, const CIntHandle& to, int vectorSize )
{
	CMemoryHandle bufs[2] = { from, to };
	size_t sizes[2] = { vectorSize * sizeof( float ), vectorSize * sizeof( int ) };

	runVectorShader( shaderLoader->GET_SHADER_DATA(VectorConvertFloatToInt, false, 0, 0, 2),
		0, 0, 0, 0, 0, 0, bufs, sizes, 2, Ceil(vectorSize, VectorCombine) );
}

void CVulkanMathEngine::VectorConvert( const CConstIntHandle& from, const CFloatHandle& to, int vectorSize )
{
	CMemoryHandle bufs[2] = { from, to };
	size_t sizes[2] = { vectorSize * sizeof( int ), vectorSize * sizeof( float ) };

	runVectorShader( shaderLoader->GET_SHADER_DATA(VectorConvertIntToFloat, false, 0, 0, 2),
		0, 0, 0, 0, 0, 0, bufs, sizes, 2, Ceil(vectorSize, VectorCombine) );
}

void CVulkanMathEngine::VectorFillBernoulli( const CFloatHandle& result, float p, int vectorSize, float value, int seed )
{
	CMemoryHandle bufs[1] = { result };
	size_t sizes[1] = { vectorSize * sizeof(float) };

	const unsigned int threshold = ( unsigned int ) ( ( double ) p * UINT_MAX );
	PARAM_STRUCT(VectorFillBernoulli) param = { value, p, threshold, seed };

	runVectorShader( shaderLoader->GET_SHADER_DATA(VectorFillBernoulli, true, 0, 0, 1),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 1, Ceil(vectorSize, VectorCombine) );
}

void CVulkanMathEngine::FilterSmallValues( const CFloatHandle&, int, float )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorCopy( const CFloatHandle& to, const CConstFloatHandle& from, int vectorSize )
{
	VkBufferCopy region = {};
	region.srcOffset = GetRawOffset(from);
	region.dstOffset = GetRawOffset(to);
	region.size = vectorSize * sizeof( float );

	CVulkanMemory* vulkanMemoryFrom = GetRawAllocation(from);
	CVulkanMemory* vulkanMemoryTo = GetRawAllocation(to);

	std::lock_guard<std::mutex> lock( Mutex );
	commandQueue->RunCopyBuffer( vulkanMemoryFrom->Buffer(), vulkanMemoryTo->Buffer(), region );
}

void CVulkanMathEngine::VectorCopy(const CIntHandle& to, const CConstIntHandle& from, int vectorSize)
{
	VkBufferCopy region = {};
	region.srcOffset = GetRawOffset(from);
	region.dstOffset = GetRawOffset(to);
	region.size = vectorSize * sizeof(int);

	CVulkanMemory* vulkanMemoryFrom = GetRawAllocation(from);
	CVulkanMemory* vulkanMemoryTo = GetRawAllocation(to);

	std::lock_guard<std::mutex> lock( Mutex );
	commandQueue->RunCopyBuffer( vulkanMemoryFrom->Buffer(), vulkanMemoryTo->Buffer(), region );
}

void CVulkanMathEngine::BroadcastCopy( const CIntHandle& /*toHandle*/, const CConstIntHandle& /*fromHandle*/,
	const CBlobDesc& /*toDesc*/, const CBlobDesc& /*fromDesc*/, int /*additionalWidth*/ )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::BroadcastCopy( const CFloatHandle& /*toHandle*/, const CConstFloatHandle& /*fromHandle*/,
	const CBlobDesc& /*toDesc*/, const CBlobDesc& /*fromDesc*/, int /*additionalWidth*/ )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorELU(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize, CFloatParam alpha)
{
	CFloatHandleStackVar alphaHandle( *this );
	alphaHandle.SetValue( alpha );
	const CMemoryHandle bufs[3]{ firstHandle, resultHandle, alphaHandle.GetHandle() };
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float), sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorELU, false, 0, 0, 3),
		0, 0, 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorELUDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, CFloatParam alpha)
{
	CFloatHandleStackVar alphaHandle( *this );
	alphaHandle.SetValue( alpha );
	const CMemoryHandle bufs[4]{ firstHandle, secondHandle, resultHandle, alphaHandle.GetHandle() };
	const size_t sizes[4]{ vectorSize * sizeof(float), vectorSize * sizeof(float),
		vectorSize * sizeof(float), sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorELUDiff, false, 0, 0, 4),
		0, 0, 0, 0, 0, 0, bufs, sizes, 4, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorELUDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, CFloatParam alpha)
{
	CFloatHandleStackVar alphaHandle( *this );
	alphaHandle.SetValue( alpha );
	const CMemoryHandle bufs[4]{ firstHandle, secondHandle, resultHandle, alphaHandle.GetHandle() };
	const size_t sizes[4]{ vectorSize * sizeof(float), vectorSize * sizeof(float),
		vectorSize * sizeof(float), sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorELUDiffOp, false, 0, 0, 4),
		0, 0, 0, 0, 0, 0, bufs, sizes, 4, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorReLU(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, CFloatParam upperThreshold)
{
	CFloatHandleStackVar var( *this );
	var.SetValue( upperThreshold );

	const int countQuad = ( vectorSize / 16 ) * 4;
	if( countQuad > 0 ) {
		const CMemoryHandle bufs[3]{ firstHandle, resultHandle, var.GetHandle() };
		const size_t sizes[3]{ 4 * countQuad * sizeof( float ), 4 * countQuad * sizeof( float ), sizeof( float ) };

		runVectorShader( shaderLoader->GET_SHADER_DATA( VectorReLU4, false, 0, 0, 3 ),
			0, 0, 0, 0, 0, 0, bufs, sizes, 3, countQuad );
	}

	const int countSingle = vectorSize % 16;
	if( countSingle > 0 ) {
		const int offset = vectorSize - countSingle;
		const CMemoryHandle bufs[3]{ firstHandle + offset, resultHandle + offset, var.GetHandle() };
		const size_t sizes[3]{ countSingle * sizeof( float ), countSingle * sizeof( float ), sizeof( float ) };

		runVectorShader( shaderLoader->GET_SHADER_DATA( VectorReLU, false, 0, 0, 3 ),
			0, 0, 0, 0, 0, 0, bufs, sizes, 3, countSingle );
	}
}

void CVulkanMathEngine::VectorReLUDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, CFloatParam upperThreshold)
{
	CFloatHandleStackVar upperThresholdHandle( *this );
	upperThresholdHandle.SetValue( upperThreshold );
	const CMemoryHandle bufs[4]{ firstHandle, secondHandle, resultHandle, upperThresholdHandle.GetHandle() };
	const size_t sizes[4]{ vectorSize * sizeof(float), vectorSize * sizeof(float),
		vectorSize * sizeof(float), sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorReLUDiff, false, 0, 0, 4),
		0, 0, 0, 0, 0, 0, bufs, sizes, 4, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorLeakyReLU(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, CFloatParam alpha)
{
	CFloatHandleStackVar alphaHandle( *this );
	alphaHandle.SetValue( alpha );
	const CMemoryHandle bufs[3]{ firstHandle, resultHandle, alphaHandle.GetHandle() };
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float), sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorLeakyReLU, false, 0, 0, 3),
		0, 0, 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorLeakyReLUDiff(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize,
	CFloatParam alpha)
{
	CFloatHandleStackVar alphaHandle( *this );
	alphaHandle.SetValue( alpha );
	const CMemoryHandle bufs[4]{ firstHandle, secondHandle, resultHandle, alphaHandle.GetHandle() };
	const size_t sizes[4]{ vectorSize * sizeof(float), vectorSize * sizeof(float),
		vectorSize * sizeof(float), sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorLeakyReLUDiff, false, 0, 0, 4),
		0, 0, 0, 0, 0, 0, bufs, sizes, 4, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorHSwish( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize )
{
	const CMemoryHandle bufs[2]{ firstHandle, resultHandle };
	const size_t sizes[2]{ vectorSize * sizeof( float ), vectorSize * sizeof( float ) };

	runVectorShader( shaderLoader->GET_SHADER_DATA( VectorHSwish, false, 0, 0, 2 ),
		0, 0, 0, 0, 0, 0, bufs, sizes, 2, Ceil( vectorSize, VectorCombine ) );
}

void CVulkanMathEngine::VectorHSwishDiff( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize )
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof( float ), vectorSize * sizeof( float ), vectorSize * sizeof( float ) };

	runVectorShader( shaderLoader->GET_SHADER_DATA( VectorHSwishDiff, false, 0, 0, 3 ),
		0, 0, 0, 0, 0, 0, bufs, sizes, 3, Ceil( vectorSize, VectorCombine ) );
}

void CVulkanMathEngine::VectorEltwiseMax(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float), vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorEltwiseMax, false, 0, 0, 3),
		0, 0, 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorEltwiseMin(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float), vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorEltwiseMin, false, 0, 0, 3),
		0, 0, 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorAbs(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[2]{ firstHandle, resultHandle };
	const size_t sizes[2]{ vectorSize * sizeof(float), vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorAbs, false, 0, 0, 2),
		0, 0, 0, 0, 0, 0, bufs, sizes, 2, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorAbsDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float),
		vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorAbsDiff, false, 0, 0, 3),
		0, 0, 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorHinge(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[2]{ firstHandle, resultHandle };
	const size_t sizes[2]{ vectorSize * sizeof(float), vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorHinge, false, 0, 0, 2),
		0, 0, 0, 0, 0, 0, bufs, sizes, 2, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorHingeDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float),
		vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorHingeDiff, false, 0, 0, 3),
		0, 0, 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorSquaredHinge(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[2]{ firstHandle, resultHandle };
	const size_t sizes[2]{ vectorSize * sizeof(float), vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorSquaredHinge, false, 0, 0, 2),
		0, 0, 0, 0, 0, 0, bufs, sizes, 2, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorSquaredHingeDiff(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float),
		vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorSquaredHingeDiff, false, 0, 0, 3),
		0, 0, 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorHuber(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[2]{ firstHandle, resultHandle };
	const size_t sizes[2]{ vectorSize * sizeof(float), vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorHuber, false, 0, 0, 2),
		0, 0, 0, 0, 0, 0, bufs, sizes, 2, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorHuberDerivative( const CConstFloatHandle&, const CFloatHandle&, int )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorHardTanh(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[2]{ firstHandle, resultHandle };
	const size_t sizes[2]{ vectorSize * sizeof(float), vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorHardTanh, false, 0, 0, 2),
		0, 0, 0, 0, 0, 0, bufs, sizes, 2, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorHardTanhDiff(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float),
		vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorHardTanhDiff, false, 0, 0, 3),
		0, 0, 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorHardSigmoid(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, CFloatParam slope, CFloatParam bias)
{
	CFloatHandleStackVar var( *this, 2 );
	var.SetValueAt( 0, slope );
	var.SetValueAt( 1, bias );
	const CMemoryHandle bufs[4]{ firstHandle, resultHandle, var.GetHandle(), var.GetHandle() + 1 };
	const size_t sizes[4]{ vectorSize * sizeof( float ), vectorSize * sizeof( float ),
		sizeof( float ), sizeof( float ) };

	runVectorShader( shaderLoader->GET_SHADER_DATA(VectorHardSigmoid, false, 0, 0, 4),
		0, 0, 0, 0, 0, 0, bufs, sizes, 4, Ceil( vectorSize, VectorCombine ) );
}

void CVulkanMathEngine::VectorHardSigmoidDiff(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize,
	CFloatParam slope, CFloatParam bias )
{
	CFloatHandleStackVar var( *this, 2 );
	var.SetValueAt( 0, slope );
	var.SetValueAt( 1, bias );
	const CMemoryHandle bufs[5]{ firstHandle, secondHandle, resultHandle, var.GetHandle(), var.GetHandle() + 1 };
	const size_t sizes[5]{ vectorSize * sizeof(float), vectorSize * sizeof(float),
		vectorSize * sizeof(float), sizeof( float ), sizeof( float ) };

	runVectorShader( shaderLoader->GET_SHADER_DATA( VectorHardSigmoidDiff, false, 0, 0, 5 ),
		0, 0, 0, 0, 0, 0, bufs, sizes, 5, Ceil( vectorSize, VectorCombine ) );
}

void CVulkanMathEngine::VectorHardSigmoidDiffOp( const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize,
	CFloatParam slope, CFloatParam /*bias*/ )
{
	CFloatHandleStackVar slopeHandle( *this );
	slopeHandle.SetValue( slope );
	const CMemoryHandle bufs[4]{ firstHandle, secondHandle, resultHandle, slopeHandle.GetHandle() };
	const size_t sizes[4]{ vectorSize * sizeof(float), vectorSize * sizeof(float),
		vectorSize * sizeof(float), sizeof( float ) };

	runVectorShader( shaderLoader->GET_SHADER_DATA( VectorHardSigmoidDiffOp, false, 0, 0, 4 ),
		0, 0, 0, 0, 0, 0, bufs, sizes, 4, Ceil( vectorSize, VectorCombine ) );
}

void CVulkanMathEngine::VectorExp( const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize )
{
	const CMemoryHandle bufs[2]{ firstHandle, resultHandle };
	const size_t sizes[2]{ vectorSize * sizeof(float), vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorExp, false, 0, 0, 2),
		0, 0, 0, 0, 0, 0, bufs, sizes, 2, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorLog(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[2]{ firstHandle, resultHandle };
	const size_t sizes[2]{ vectorSize * sizeof(float), vectorSize * sizeof(float) };

	PARAM_STRUCT(VectorLog) param = { 0 };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorLog, true, 0, 0, 2),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorNegLog(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[2]{ firstHandle, resultHandle };
	const size_t sizes[2]{ vectorSize * sizeof(float), vectorSize * sizeof(float) };

	PARAM_STRUCT(VectorLog) param = { 1 };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorLog, true, 0, 0, 2),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorErf( const CConstFloatHandle&, const CFloatHandle&, int )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorBernulliKLDerivative(const CConstFloatHandle& estimationHandle,
	const CFloatHandle& resultHandle, int vectorSize, CFloatParam target)
{
	CFloatHandleStackVar targetHandle( *this );
	targetHandle.SetValue( target );
	const CMemoryHandle bufs[3]{ estimationHandle, resultHandle, targetHandle.GetHandle() };
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float), sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorBernulliKLDerivative, false, 0, 0, 3),
		0, 0, 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorAdd(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	const int countQuad = ( vectorSize / 16 ) * 4;
	if( countQuad > 0 ) {
		const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
		const size_t sizes[3]{ 4 * countQuad * sizeof( float ), 4 * countQuad * sizeof( float ),
			4 * countQuad * sizeof( float ) };

		runVectorShader( shaderLoader->GET_SHADER_DATA( VectorAddFloat4, false, 0, 0, 3 ),
			0, 0, 0, 0, 0, 0, bufs, sizes, 3, countQuad );
	}

	const int countSingle = vectorSize % 16;
	if( countSingle > 0 ) {
		const int offset = vectorSize - countSingle;
		const CMemoryHandle bufs[3]{ firstHandle + offset, secondHandle + offset, resultHandle + offset };
		const size_t sizes[3]{ countSingle * sizeof( float ), countSingle * sizeof( float ),
			countSingle * sizeof( float ) };

		runVectorShader( shaderLoader->GET_SHADER_DATA( VectorAddFloat1, false, 0, 0, 3 ),
			0, 0, 0, 0, 0, 0, bufs, sizes, 3, countSingle );
	}
}

void CVulkanMathEngine::VectorAdd(const CConstIntHandle& firstHandle,
	const CConstIntHandle& secondHandle, const CIntHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(int), vectorSize * sizeof(int), vectorSize * sizeof(int) };

	PARAM_STRUCT(VectorAddInt) param = { 0 };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorAddInt, true, 0, 0, 3),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorAddValue(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, CFloatParam addition)
{
	CFloatHandleStackVar additionHandle( *this );
	additionHandle.SetValue( addition );
	const CMemoryHandle bufs[3]{ firstHandle, additionHandle.GetHandle(), resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), sizeof(float), vectorSize * sizeof(float) };

	runVectorShader( shaderLoader->GET_SHADER_DATA( VectorAddValue, false, 0, 0, 3),
		0, 0, 0, 0, 0, 0, bufs, sizes, 3, Ceil( vectorSize, VectorCombine ) );
}

void CVulkanMathEngine::VectorAddValue(const CConstIntHandle& firstHandle,
	const CIntHandle& resultHandle, int vectorSize, CIntParam addition)
{
	CIntHandleStackVar additionHandle( *this );
	additionHandle.SetValue( addition );
	const CMemoryHandle bufs[3]{ firstHandle, additionHandle.GetHandle(), resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(int), sizeof(int), vectorSize * sizeof(int) };

	PARAM_STRUCT(VectorAddInt) param = { 1 };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorAddInt, true, 0, 0, 3),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorSub(const CConstIntHandle& firstHandle,
	const CConstIntHandle& secondHandle, const CIntHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float), vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorSubInt, false, 0, 0, 3),
		0, 0, 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorSub(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float), vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorSubFloat, false, 0, 0, 3),
		0, 0, 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorMultiplyAndAdd(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle,
	int vectorSize, CFloatParam mult)
{
	CFloatHandleStackVar multHandle( *this );
	multHandle.SetValue( mult );
	const CMemoryHandle bufs[4]{ firstHandle, secondHandle, resultHandle, multHandle.GetHandle() };
	const size_t sizes[4]{ vectorSize * sizeof(float), vectorSize * sizeof(float),
		vectorSize * sizeof(float), sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorMultiplyAndAdd, false, 0, 0, 4),
		0, 0, 0, 0, 0, 0, bufs, sizes, 4, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorMultiplyAndSub(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle,
	int vectorSize, CFloatParam mult)
{
	CFloatHandleStackVar multHandle( *this );
	multHandle.SetValue( mult );
	const CMemoryHandle bufs[4]{ firstHandle, secondHandle, resultHandle, multHandle.GetHandle() };
	const size_t sizes[4]{ vectorSize * sizeof(float), vectorSize * sizeof(float),
		vectorSize * sizeof(float), sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorMultiplyAndSub, false, 0, 0, 4),
		0, 0, 0, 0, 0, 0, bufs, sizes, 4, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorMultiply(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, CFloatParam mult)
{
	CFloatHandleStackVar multiplierHandle( *this );
	multiplierHandle.SetValue( mult );
	const CMemoryHandle bufs[3]{ firstHandle, multiplierHandle.GetHandle(), resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), sizeof(float), vectorSize * sizeof(float) };

	PARAM_STRUCT(VectorMultiplyFloat) param = { 1, 0, 0 };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorMultiplyFloat, true, 0, 0, 3),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorMultiply(const CConstIntHandle&, const CIntHandle&, int, CIntParam )
{
    ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorNegMultiply(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, CFloatParam mult)
{
	CFloatHandleStackVar multiplierHandle( *this );
	multiplierHandle.SetValue( mult );
	const CMemoryHandle bufs[3]{ firstHandle, multiplierHandle.GetHandle(), resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), sizeof(float), vectorSize * sizeof(float) };

	PARAM_STRUCT(VectorMultiplyFloat) param = { 1, 1, 0 };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorMultiplyFloat, true, 0, 0, 3),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorEltwiseMultiply(const CConstIntHandle& firstHandle,
	const CConstIntHandle& secondHandle, const CIntHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(int), vectorSize * sizeof(int), vectorSize * sizeof(int) };

	PARAM_STRUCT(VectorMultiplyInt) param = { 0, 0, 0 };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorMultiplyInt, true, 0, 0, 3),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorEltwiseMultiply(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float), vectorSize * sizeof(float) };

	PARAM_STRUCT(VectorMultiplyFloat) param = { 0, 0, 0 };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorMultiplyFloat, true, 0, 0, 3),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorEltwiseMultiplyAdd(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float), vectorSize * sizeof(float) };

	PARAM_STRUCT(VectorMultiplyFloat) param = { 0, 0, 1 };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorMultiplyFloat, true, 0, 0, 3),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorEltwiseNegMultiply(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float), vectorSize * sizeof(float) };

	PARAM_STRUCT(VectorMultiplyFloat) param = { 0, 1, 0 };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorMultiplyFloat, true, 0, 0, 3),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorEltwiseDivide(const CConstIntHandle& firstHandle,
	const CConstIntHandle& secondHandle, const CIntHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float), vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorEltwiseDivideInt, false, 0, 0, 3),
		0, 0, 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorEltwiseDivide(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float), vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorEltwiseDivideFloat, false, 0, 0, 3),
		0, 0, 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorEltwisePower(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float), vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorEltwisePower, false, 0, 0, 3),
		0, 0, 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorSqrt(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[2]{ firstHandle, resultHandle };
	const size_t sizes[2]{ vectorSize * sizeof(float), vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorSqrt, false, 0, 0, 2),
		0, 0, 0, 0, 0, 0, bufs, sizes, 2, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorInv(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[2]{ firstHandle, resultHandle };
	const size_t sizes[2]{ vectorSize * sizeof(float), vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorInv, false, 0, 0, 2),
		0, 0, 0, 0, 0, 0, bufs, sizes, 2, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorMinMax(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize, CFloatParam min, CFloatParam max)
{
	CFloatHandleStackVar var( *this, 2 );
	var.SetValueAt( 0, min );
	var.SetValueAt( 1, max );
	const CMemoryHandle bufs[4]{ firstHandle, resultHandle, var.GetHandle(), var.GetHandle() + 1 };
	const size_t sizes[4]{ vectorSize * sizeof(float), vectorSize * sizeof(float), sizeof(float), sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorMinMax, false, 0, 0, 4),
		0, 0, 0, 0, 0, 0, bufs, sizes, 4, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorSigmoid(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[2]{ firstHandle, resultHandle };
	const size_t sizes[2]{ vectorSize * sizeof(float), vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorSigmoid, false, 0, 0, 2),
		0, 0, 0, 0, 0, 0, bufs, sizes, 2, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorSigmoidDiff(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float),
		vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorSigmoidDiff, false, 0, 0, 3),
		0, 0, 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorSigmoidDiffOp(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float),
		vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorSigmoidDiffOp, false, 0, 0, 3),
		0, 0, 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorTanh(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[2]{ firstHandle, resultHandle };
	const size_t sizes[2]{ vectorSize * sizeof(float), vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorTanh, false, 0, 0, 2), 0, 0, 0, 0, 0, 0, bufs, sizes, 2,
		Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorTanhDiff(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float),
		vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorTanhDiff, false, 0, 0, 3),
		0, 0, 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorTanhDiffOp(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float),
		vectorSize * sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorTanhDiffOp, false, 0, 0, 3),
		0, 0, 0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorPower(float exponent, const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[2]{ firstHandle, resultHandle };
	const size_t sizes[2]{ vectorSize * sizeof(float), vectorSize * sizeof(float) };

	PARAM_STRUCT(VectorPower) param = { exponent };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorPower, true, 0, 0, 2), &param, sizeof(param),
		0, 0, 0, 0, bufs, sizes, 2, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorPowerDiff(float exponent, const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float),
		vectorSize * sizeof(float) };

	PARAM_STRUCT(VectorPowerDiff) param = { exponent };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorPowerDiff, true, 0, 0, 3), &param, sizeof(param),
		0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorPowerDiffOp(float exponent, const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float),
		vectorSize * sizeof(float) };

	PARAM_STRUCT(VectorPowerDiffOp) param = { exponent };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorPowerDiffOp, true, 0, 0, 3), &param, sizeof(param),
		0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorL1DiffAdd(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, CFloatParam hubertThreshold, CFloatParam mult)
{
	CFloatHandleStackVar var( *this, 2 );
	var.SetValueAt( 0, hubertThreshold );
	var.SetValueAt( 1, mult );
	const CMemoryHandle bufs[5]{ firstHandle, secondHandle, resultHandle, var.GetHandle(), var.GetHandle() + 1 };
	const size_t sizes[5]{ vectorSize * sizeof(float), vectorSize * sizeof(float), vectorSize * sizeof(float),
		sizeof(float), sizeof(float) };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorL1DiffAdd, false, 0, 0, 5),
		0, 0, 0, 0, 0, 0, bufs, sizes, 5, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorDotProduct(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, int vectorSize, const CFloatHandle& resultHandle)
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle};
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float), sizeof(float) };

	const CVulkanShaderData& shaderData = shaderLoader->GET_SHADER_DATA(VectorDotProduct, false, 0, 0, 3);

	runVectorShader(shaderData, 0, 0, 0, 0, 0, 0, bufs, sizes, 3, shaderData.GetGroupSize());
}

void CVulkanMathEngine::VectorEltwiseNot( const CConstIntHandle&, const CIntHandle&, int )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorEltwiseNotNegative( const CConstIntHandle&, const CFloatHandle&, int )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorEltwiseLess( const CConstFloatHandle&, const CConstFloatHandle&,
	const CFloatHandle&, int )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorEltwiseLess( const CConstFloatHandle&, float,
	const CFloatHandle&, int )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorEltwiseLess( float, const CConstFloatHandle&,
	const CFloatHandle&, int )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorEltwiseLess( const CConstFloatHandle&, const CConstFloatHandle&,
	const CIntHandle&, int )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorEltwiseLess( const CConstIntHandle&, const CConstIntHandle&,
	const CIntHandle&, int )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorEltwiseEqual( const CConstFloatHandle&, const CConstFloatHandle&,
	const CIntHandle&, int )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorEltwiseEqual( const CConstIntHandle&, const CConstIntHandle&,
	const CIntHandle&, int )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorEltwiseWhere( const CConstIntHandle&, const CConstFloatHandle&, const CConstFloatHandle&,
	const CFloatHandle&, int )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorEltwiseWhere( const CConstIntHandle&, const CConstIntHandle&, const CConstIntHandle&,
	const CIntHandle&, int )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorFindMaxValueInSet( const CConstFloatHandle* vectors, int vectorCount, const CFloatHandle& resultHandle,
	int vectorSize )
{
	if( vectorCount > 0 ) {
		VectorCopy( resultHandle, vectors[0], vectorSize );
	}

	PARAM_STRUCT( VectorFindMaxValueInSetNoIndices ) param = { vectorSize };

	for( int i = 1; i < vectorCount; i++ ) {
		const CMemoryHandle bufs[2]{ vectors[i], resultHandle };
		const size_t sizes[2]{ vectorSize * sizeof(float), vectorSize * sizeof(float) };

		runVectorShader( shaderLoader->GET_SHADER_DATA( VectorFindMaxValueInSetNoIndices, true, 0, 0, 2 ), &param, sizeof( param ),
			0, 0, 0, 0, bufs, sizes, 2, vectorSize );
	}
}

void CVulkanMathEngine::VectorFindMaxValueInSet( const CConstFloatHandle* vectors, int vectorCount, const CFloatHandle& resultHandle,
	const CIntHandle& indexHandle, int vectorSize )
{
	if( vectorCount > 0 ) {
		VectorCopy( resultHandle, vectors[0], vectorSize );
		VectorFill( indexHandle, 0, vectorSize );
	}

	for( int i = 1; i < vectorCount; i++ ) {
		const CMemoryHandle bufs[3]{ vectors[i], resultHandle, indexHandle };
		const size_t sizes[3]{ vectorSize * sizeof( float ), vectorSize * sizeof( float ), vectorSize * sizeof( int ) };

		PARAM_STRUCT( VectorFindMaxValueInSet ) param = { vectorSize, i };

		runVectorShader( shaderLoader->GET_SHADER_DATA( VectorFindMaxValueInSet, true, 0, 0, 3 ), &param, sizeof( param ),
			0, 0, 0, 0, bufs, sizes, 3, vectorSize );
	}
}

void CVulkanMathEngine::VectorSpreadValues( const CConstFloatHandle&, CFloatHandle*, int, const CConstIntHandle&, int )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorSum( const CConstFloatHandle& firstHandle, int vectorSize, const CFloatHandle& resultHandle )
{
	const CMemoryHandle bufs[2]{ firstHandle, resultHandle };
	const size_t sizes[2]{ vectorSize * sizeof(float), sizeof(float) };

	PARAM_STRUCT(VectorSum) param = { 0, 0 };

	const CVulkanShaderData& shaderData = shaderLoader->GET_SHADER_DATA(VectorSum, true, 0, 0, 2);

	runVectorShader(shaderData, &param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2, shaderData.GetGroupSize());
}

void CVulkanMathEngine::VectorSumAdd( const CConstFloatHandle& firstHandle, int vectorSize, const CFloatHandle& resultHandle )
{
	const CMemoryHandle bufs[2]{ firstHandle, resultHandle };
	const size_t sizes[2]{ vectorSize * sizeof(float), sizeof(float) };

	PARAM_STRUCT(VectorSum) param = { 1, 0 };

	const CVulkanShaderData& shaderData = shaderLoader->GET_SHADER_DATA(VectorSum, true, 0, 0, 2);

	runVectorShader(shaderData, &param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2, shaderData.GetGroupSize());
}

void CVulkanMathEngine::VectorSumAlongDimension( const CConstFloatHandle&, int, int, int, const CFloatHandle& )
{
	ASSERT_EXPR(false);
}

void CVulkanMathEngine::VectorCumSumAlongDimension( const CConstFloatHandle& /*firstHandle*/, int /*precedingDimension*/, int /*dimension*/,
	int /*followingDimension*/, const CFloatHandle& /*resultHandle*/, bool /*reverse*/ )
{
	ASSERT_EXPR(false);
}

void CVulkanMathEngine::VectorCumSumAlongDimension( const CConstIntHandle& /*firstHandle*/, int /*precedingDimension*/, int /*dimension*/,
	int /*followingDimension*/, const CIntHandle& /*resultHandle*/, bool /*reverse*/ )
{
	ASSERT_EXPR(false);
}

void CVulkanMathEngine::VectorSumAlongDimensionDiag( const CConstFloatHandle& /*firstHandle*/, int /*precedingDimension*/, int /*dimension*/,
	int /*followingDimension*/, const CFloatHandle& /*resultHandle*/ )
{
	ASSERT_EXPR(false);
}

void CVulkanMathEngine::VectorCumSumAlongDimensionDiag( const CConstFloatHandle& /*firstHandle*/, int /*precedingDimension*/, int /*dimension*/,
	int /*followingDimension*/, const CFloatHandle& /*resultHandle*/ )
{
	ASSERT_EXPR(false);
}

void CVulkanMathEngine::VectorEqual(const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	const CMemoryHandle bufs[3]{ firstHandle, secondHandle, resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), vectorSize * sizeof(float), vectorSize * sizeof(float) };

	PARAM_STRUCT(VectorEqual) param = { 0 };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorEqual, true, 0, 0, 3), &param, sizeof(param),
		0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorEqualValue(const CConstIntHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, CIntParam value)
{
	CIntHandleStackVar valueHandle( *this );
	valueHandle.SetValue( value );
	const CMemoryHandle bufs[3]{ firstHandle, valueHandle.GetHandle(), resultHandle };
	const size_t sizes[3]{ vectorSize * sizeof(float), sizeof(float), vectorSize * sizeof(float) };

	PARAM_STRUCT(VectorEqual) param = { 1 };

	runVectorShader(shaderLoader->GET_SHADER_DATA(VectorEqual, true, 0, 0, 3), &param, sizeof(param),
		0, 0, 0, 0, bufs, sizes, 3, Ceil(vectorSize, VectorCombine));
}

void CVulkanMathEngine::VectorMax( const CConstFloatHandle& /*firstHandle*/, CFloatParam /*secondValue*/, const CFloatHandle& /*resultHandle*/,
	int /*vectorSize*/ )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorMaxDiff( const CConstFloatHandle& /*firstHandle*/, CFloatParam /*secondValue*/, const CFloatHandle& /*gradHandle*/,
	int /*gradHeight*/, int /*gradWidth*/ )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorNeg(const CConstFloatHandle& /*firstHandle*/, const CFloatHandle& /*resultHandle*/, int /*vectorSize*/)
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorLogDiff( const CConstFloatHandle& /*sourceGradHandle*/, int /*sourceGradHeight*/, int /*sourceGradWidth*/,
	const CConstFloatHandle& /*valueHandle*/, const CFloatHandle& /*resultHandle*/ )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorSub(const CConstFloatHandle& /*firstHandle*/, float /*second*/, const CFloatHandle& /*resultHandle*/,
	int /*vectorSize*/)
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorSub(float /*first*/, const CConstFloatHandle& /*secondHandle*/, const CFloatHandle& /*resultHandle*/,
	int /*vectorSize*/)
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorTopK(const CConstFloatHandle& /*first*/, int /*firstSize*/, int /*k*/, const CFloatHandle& /*result*/,
	const CIntHandle& /*indices*/)
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorTopKDiff(const CConstFloatHandle& /*sourceGrad*/, int /*sourceGradHeight*/, int /*sourceGradWidth*/,
	const CConstIntHandle& /*indices*/, int /*k*/, const CFloatHandle& /*resultGrad*/)
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorAbsDiff(const CConstFloatHandle& /*sourceGradHandle*/, int /*gradHeight*/, int /*gradWidth*/,
	const CConstFloatHandle& /*firstHandle*/, const CFloatHandle& /*resultHandle*/)
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorMinMaxDiff(const CConstFloatHandle& /*sourceGradHandle*/, int /*gradHeight*/, int /*gradWidth*/,
	const CConstFloatHandle& /*firstHandle*/, const CFloatHandle& /*resultHandle*/, CFloatParam /*min*/, CFloatParam /*max*/)
{
	ASSERT_EXPR( false );
}

} // namespace NeoML

#endif // NEOML_USE_VULKAN
