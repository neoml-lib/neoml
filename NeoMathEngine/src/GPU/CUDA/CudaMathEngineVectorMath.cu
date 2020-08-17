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
#include <CudaAssert.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>

#include <Kernels/CudaVectorMathKernels.h>

namespace NeoML {

void CCudaMathEngine::VectorCopy(const CFloatHandle& first, const CConstFloatHandle& second, int vectorSize)
{
	ASSERT_EXPR(first.GetMathEngine() == this);
	ASSERT_EXPR(second.GetMathEngine() == this);

	ASSERT_CUDA( cudaMemcpy(GetRaw(first), GetRaw(second), vectorSize * sizeof(float), cudaMemcpyDeviceToDevice));
}

void CCudaMathEngine::VectorCopy(const CIntHandle& first, const CConstIntHandle& second, int vectorSize)
{
	ASSERT_EXPR(first.GetMathEngine() == this);
	ASSERT_EXPR(second.GetMathEngine() == this);

	ASSERT_CUDA( cudaMemcpy(GetRaw(first), GetRaw(second), vectorSize * sizeof(int), cudaMemcpyDeviceToDevice));
}

void CCudaMathEngine::VectorFill(const CFloatHandle& result, float value, int vectorSize)
{
	ASSERT_EXPR(result.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorFillCombineCount);

	VectorFillKernel<<<blockCount, threadCount, 0, cudaStream>>>(GetRaw(result), value, vectorSize);
}

void CCudaMathEngine::VectorFill(const CIntHandle& result, int value, int vectorSize)
{
	ASSERT_EXPR(result.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorFillCombineCount);

	VectorFillKernel<<<blockCount, threadCount, 0, cudaStream>>>(GetRaw(result), value, vectorSize);
}

void CCudaMathEngine::VectorFill(const CFloatHandle& result, int vectorSize, const CConstFloatHandle& value)
{
	ASSERT_EXPR(result.GetMathEngine() == this);
	ASSERT_EXPR(value.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorFillHandleCombineCount);

	VectorFillHandleKernel<<<blockCount, threadCount, 0, cudaStream>>>(GetRaw(result), vectorSize, GetRaw(value));
}

void CCudaMathEngine::VectorFill(const CIntHandle& result, int vectorSize, const CConstIntHandle& value)
{
	ASSERT_EXPR(result.GetMathEngine() == this);
	ASSERT_EXPR(value.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorFillHandleCombineCount);

	VectorFillHandleKernel<<<blockCount, threadCount, 0, cudaStream>>>(GetRaw(result), vectorSize, GetRaw(value));
}

void CCudaMathEngine::VectorFillBernoulli( const CFloatHandle& result, float p, int vectorSize, float valueHandle, int seed )
{
	ASSERT_EXPR(result.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, ( vectorSize + 3 ) / 4, VectorFillBernoulliCombine );

	VectorFillBernoulliKernel<<<blockCount, threadCount, 0, cudaStream>>>( GetRaw( result ), p, vectorSize,
		valueHandle, seed );
}

void CCudaMathEngine::FilterSmallValues( const CFloatHandle& data, int dataSize, float threshold )
{
	ASSERT_EXPR(data.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, dataSize, VectorFillCombineCount );

	FilterSmallValuesKernel<<<blockCount, threadCount, 0, cudaStream>>>( GetRaw( data ), threshold, dataSize );
}

void CCudaMathEngine::VectorSum(const CConstFloatHandle& firstHandle, int vectorSize, const CFloatHandle& resultHandle)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorSumCombineCount);

	bool setZero = true;
	if(blockCount > 1) {
		setZero = false;
		resultHandle.SetValue(0);
	}

	const int sharedSize = threadCount * sizeof(float);
	VectorSumKernel<<<blockCount, threadCount, sharedSize, cudaStream>>>(GetRaw(firstHandle), vectorSize,
		GetRaw(resultHandle), false, setZero);
}

void CCudaMathEngine::VectorSumAdd(const CConstFloatHandle& firstHandle, int vectorSize, const CFloatHandle& resultHandle)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorSumCombineCount);

	const int sharedSize = threadCount * sizeof(float);
	VectorSumKernel<<<blockCount, threadCount, sharedSize, cudaStream>>>(GetRaw(firstHandle), vectorSize,
		GetRaw(resultHandle), false, false);
}

void CCudaMathEngine::VectorNegSum(const CConstFloatHandle& firstHandle, int vectorSize, const CFloatHandle& resultHandle)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorSumCombineCount);

	bool setZero = true;
	if(blockCount > 1) {
		setZero = false;
		resultHandle.SetValue(0);
	}

	const int sharedSize = threadCount * sizeof(float);
	VectorSumKernel<<<blockCount, threadCount, sharedSize, cudaStream>>>
		(GetRaw(firstHandle), vectorSize, GetRaw(resultHandle), true, setZero);
}

void CCudaMathEngine::VectorEqual( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorEqualCombineCount );

	VectorEqualKernel <<<blockCount, threadCount, 0, cudaStream>>>
		(GetRaw( firstHandle ), GetRaw( secondHandle ), GetRaw( resultHandle ), vectorSize);
}

void CCudaMathEngine::VectorEqualValue( const CConstIntHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstIntHandle& valueHandle )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(valueHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorEqualCombineCount );

	VectorEqualValueKernel<<<blockCount, threadCount, 0, cudaStream>>>
		(GetRaw( firstHandle ), GetRaw( resultHandle ), vectorSize, GetRaw( valueHandle ));
}

void CCudaMathEngine::VectorELU( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize, const CConstFloatHandle& alpha )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(alpha.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorActivationCombineCount );

	VectorELUKernel<<<blockCount, threadCount, 0, cudaStream>>>( GetRaw( firstHandle ),
		GetRaw( resultHandle ), vectorSize, GetRaw( alpha ) );
}

void CCudaMathEngine::VectorELUDiff( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& alpha )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	ASSERT_EXPR(alpha.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorActivationCombineCount );

	VectorELUDiffKernel<<<blockCount, threadCount, 0, cudaStream>>>( GetRaw( firstHandle ), GetRaw( secondHandle ),
		GetRaw( resultHandle ), vectorSize, GetRaw( alpha ) );
}

void CCudaMathEngine::VectorELUDiffOp( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& alpha )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	ASSERT_EXPR(alpha.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorActivationCombineCount );

	VectorELUDiffOpKernel<<<blockCount, threadCount, 0, cudaStream>>>( GetRaw( firstHandle ), GetRaw( secondHandle ),
		GetRaw( resultHandle ), vectorSize,	GetRaw( alpha ) );
}

void CCudaMathEngine::VectorReLU(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize, const CConstFloatHandle& upperThresholdHandle)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	ASSERT_EXPR(upperThresholdHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorReLUKernel<<<blockCount, threadCount, 0, cudaStream>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize,
		GetRaw(upperThresholdHandle));
}

void CCudaMathEngine::VectorReLUDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& upperThresholdHandle)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	ASSERT_EXPR(upperThresholdHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorReLUDiffKernel<<<blockCount, threadCount, 0, cudaStream>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize, GetRaw(upperThresholdHandle));
}

void CCudaMathEngine::VectorLeakyReLU( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize, const CConstFloatHandle& alpha )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	ASSERT_EXPR(alpha.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorActivationCombineCount );

	VectorLeakyReLUKernel<<<blockCount, threadCount, 0, cudaStream>>>( GetRaw( firstHandle ),
		GetRaw( resultHandle ), vectorSize, GetRaw( alpha ) );
}

void CCudaMathEngine::VectorLeakyReLUDiff( const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize,
	const CConstFloatHandle& alpha )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	ASSERT_EXPR(alpha.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorActivationCombineCount );

	VectorLeakyReLUDiffKernel<<<blockCount, threadCount, 0, cudaStream>>>
		( GetRaw( firstHandle ), GetRaw( secondHandle ), GetRaw( resultHandle ), vectorSize,
			GetRaw( alpha ) );
}

void CCudaMathEngine::VectorHSwish( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorActivationCombineCount );

	VectorHSwishKernel<<<blockCount, threadCount, 0, cudaStream>>>
		( GetRaw( firstHandle ), GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorHSwishDiff( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorActivationCombineCount );

	VectorHSwishDiffKernel<<<blockCount, threadCount, 0, cudaStream>>>
		( GetRaw( firstHandle ), GetRaw( secondHandle ), GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorEltwiseMax(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorEltwiseMaxCombineCount);

	VectorEltwiseMaxKernel<<<blockCount, threadCount, 0, cudaStream>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorEltwiseMin(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorEltwiseMinCombineCount);

	VectorEltwiseMinKernel<<<blockCount, threadCount, 0, cudaStream>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorAbs(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorAbsKernel<<<blockCount, threadCount, 0, cudaStream>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorAbsDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorAbsDiffKernel<<<blockCount, threadCount, 0, cudaStream>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorHinge(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorHingeKernel<<<blockCount, threadCount, 0, cudaStream>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorHingeDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorHingeDiffKernel<<<blockCount, threadCount, 0, cudaStream>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}
void CCudaMathEngine::VectorSquaredHinge(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorSquaredHingeKernel<<<blockCount, threadCount, 0, cudaStream>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorSquaredHingeDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorSquaredHingeDiffKernel<<<blockCount, threadCount, 0, cudaStream>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorHuber(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorHuberKernel<<<blockCount, threadCount, 0, cudaStream>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorHuberDerivative(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorHuberDiffKernel<<<blockCount, threadCount, 0, cudaStream>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorHardTanh(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorHardTanhKernel<<<blockCount, threadCount, 0, cudaStream>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorHardTanhDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorHardTanhDiffKernel<<<blockCount, threadCount, 0, cudaStream>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorHardSigmoid( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize, 
	const CConstFloatHandle& slopeHandle, const CConstFloatHandle& biasHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorActivationCombineCount );

	VectorHardSigmoidKernel<<<blockCount, threadCount, 0, cudaStream>>>
		( GetRaw( firstHandle ), GetRaw( resultHandle ), vectorSize, GetRaw( slopeHandle ), GetRaw( biasHandle ) );
}

void CCudaMathEngine::VectorHardSigmoidDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& slopeHandle, const CConstFloatHandle& biasHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorHardSigmoidDiffKernel<<<blockCount, threadCount, 0, cudaStream>>>
		( GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize, GetRaw( slopeHandle ), GetRaw( biasHandle ) );
}

void CCudaMathEngine::VectorHardSigmoidDiffOp( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& slopeHandle, const CConstFloatHandle& /*biasHandle*/ )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorActivationCombineCount );

	VectorHardSigmoidDiffOpKernel<<<blockCount, threadCount, 0, cudaStream>>>
		( GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize, GetRaw( slopeHandle ) );
}

void CCudaMathEngine::VectorExp(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorExpKernel<<<blockCount, threadCount, 0, cudaStream>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorLog( const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize );

	VectorLogKernel<<<blockCount, threadCount, 0, cudaStream>>>( GetRaw( firstHandle ),
		GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorNegLog(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorNegLogKernel<<<blockCount, threadCount, 0, cudaStream>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorBernulliKLDerivative(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& targetHandle)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(targetHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorBernulliKLDerivativeKernel<<<blockCount, threadCount, 0, cudaStream>>>
		(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize, GetRaw(targetHandle));
}

void CCudaMathEngine::VectorAdd(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorAddCombineCount);

	VectorAddKernel<<<blockCount, threadCount, 0, cudaStream>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorAdd( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
	const CIntHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorAddCombineCount );

	VectorAddKernel<<<blockCount, threadCount, 0, cudaStream>>>
		( GetRaw( firstHandle ), GetRaw( secondHandle ), GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorAddValue(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& additionHandle)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(additionHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorAddValueCombineCount);

	VectorAddValueKernel<<<blockCount, threadCount, 0, cudaStream>>>
		(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize, GetRaw(additionHandle));
}

void CCudaMathEngine::VectorAddValue(
	const CConstIntHandle& firstHandle, const CIntHandle& resultHandle,
	int vectorSize, const CConstIntHandle& additionHandle )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(additionHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorAddValueCombineCount );

	VectorAddValueKernel<<<blockCount, threadCount, 0, cudaStream>>>(
		GetRaw( firstHandle ), GetRaw( resultHandle ), vectorSize,
		GetRaw( additionHandle ) );
}

void CCudaMathEngine::VectorSub(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorSubCombineCount);

	VectorSubKernel<<<blockCount, threadCount, 0, cudaStream>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorMultiplyAndSub(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( multHandle.GetMathEngine() == this );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorMultiplyAndSubKernel<<<blockCount, threadCount, 0, cudaStream>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize, GetRaw(multHandle));
}

void CCudaMathEngine::VectorMultiply(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multiplierHandle)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(multiplierHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorMultiplyCombineCount);

	VectorMultiplyKernel<<<blockCount, threadCount, 0, cudaStream>>>
		(GetRaw(firstHandle),GetRaw(resultHandle), vectorSize, GetRaw(multiplierHandle));
}

void CCudaMathEngine::VectorNegMultiply(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multiplierHandle)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(multiplierHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorMultiplyCombineCount);

	VectorNegMultiplyKernel<<<blockCount, threadCount, 0, cudaStream>>>
		(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize, GetRaw(multiplierHandle));
}

void CCudaMathEngine::VectorEltwiseMultiply(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorEltwiseMultiplyCombineCount);

	VectorEltwiseMultiplyKernel<<<blockCount, threadCount, 0, cudaStream>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorEltwiseMultiplyAdd(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorEltwiseMultiplyAddCombineCount);

	VectorEltwiseMultiplyAddKernel<<<blockCount, threadCount, 0, cudaStream>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorEltwiseNegMultiply(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorEltwiseNegMultiplyCombineCount);

	VectorEltwiseNegMultiplyKernel<<<blockCount, threadCount, 0, cudaStream>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorEltwiseDivide(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorEltwiseDivideCombineCount);

	VectorEltwiseDivideKernel<<<blockCount, threadCount, 0, cudaStream>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorEltwisePower(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorEltwisePowerKernel<<<blockCount, threadCount, 0, cudaStream>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorSqrt(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorSqrtKernel<<<blockCount, threadCount, 0, cudaStream>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorInv(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorInvCombineCount);

	VectorInvKernel<<<blockCount, threadCount, 0, cudaStream>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorMinMax(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize,
	const CConstFloatHandle& minHandle, const CConstFloatHandle& maxHandle)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(minHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	ASSERT_EXPR(maxHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorMinMaxCombineCount);

	VectorMinMaxKernel<<<blockCount, threadCount, 0, cudaStream>>>(GetRaw(firstHandle), GetRaw(resultHandle),
		vectorSize, GetRaw(minHandle), GetRaw(maxHandle));
}

void CCudaMathEngine::VectorSigmoid(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorSigmoidKernel<<<blockCount, threadCount, 0, cudaStream>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorSigmoidDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorSigmoidDiffKernel<<<blockCount, threadCount, 0, cudaStream>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorSigmoidDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorSigmoidDiffOpCombineCount);

	VectorSigmoidDiffOpKernel<<<blockCount, threadCount, 0, cudaStream>>>(GetRaw(firstHandle), GetRaw(secondHandle),
		GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorTanh(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorTanhKernel<<<blockCount, threadCount, 0, cudaStream>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorTanhDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorTanhDiffKernel<<<blockCount, threadCount, 0, cudaStream>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorTanhDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorTanhDiffOpCombineCount);

	VectorTanhDiffOpKernel<<<blockCount, threadCount, 0, cudaStream>>>(GetRaw(firstHandle), GetRaw(secondHandle),
		GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorPower(float exponent, const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorPowerKernel<<<blockCount, threadCount, 0, cudaStream>>>(exponent, GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorPowerDiff(float exponent, const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorPowerDiffKernel<<<blockCount, threadCount, 0, cudaStream>>>
		(exponent, GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorPowerDiffOp(float exponent, const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorPowerDiffOpKernel<<<blockCount, threadCount, 0, cudaStream>>>
		(exponent, GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorL1DiffAdd(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize,
	const CConstFloatHandle& hubertThresholdHandle, const CConstFloatHandle& multHandle)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	ASSERT_EXPR(hubertThresholdHandle.GetMathEngine() == this);
	ASSERT_EXPR(multHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorL1DiffAddKernel<<<blockCount, threadCount, 0, cudaStream>>>(GetRaw(firstHandle), GetRaw(secondHandle),
		GetRaw(resultHandle), vectorSize, GetRaw(hubertThresholdHandle), GetRaw(multHandle));
}

void CCudaMathEngine::VectorEltwiseNotNegative( const CConstIntHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize );

	vectorGreaterEqualToZeroKernel<<<blockCount, threadCount, 0, cudaStream>>>( GetRaw( firstHandle ),
		GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorFindMaxValueInSet(const CConstFloatHandle* vectors, int vectorCount,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectors[0].GetMathEngine() == this );

	VectorCopy( resultHandle, vectors[0], vectorSize );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	CCudaConstVectorArray cudaVectors;
	cudaVectors.VectorCount = 0;
	for( int i = 1; i < vectorCount; i++ ) {
		ASSERT_EXPR( vectors[i].GetMathEngine() == this );
		cudaVectors.Vectors[cudaVectors.VectorCount] = GetRaw( vectors[i] );
		cudaVectors.VectorCount++;

		if( cudaVectors.VectorCount == CCudaConstVectorArray::MaxSize || i + 1 == vectorCount ) {
			VectorFindMaxValueInSetKernel<<<blockCount, threadCount, 0, cudaStream>>>(cudaVectors,
				GetRaw(resultHandle), vectorSize);
			cudaVectors.VectorCount = 0;
		}
	}
}

void CCudaMathEngine::VectorFindMaxValueInSet(const CConstFloatHandle* vectors, int vectorCount, const CFloatHandle& resultHandle,
	const CIntHandle& indexHandle, int vectorSize)
{
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( indexHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectors[0].GetMathEngine() == this );

	VectorFill( indexHandle, 0, vectorSize );
	VectorCopy( resultHandle, vectors[0], vectorSize );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	CCudaConstVectorArray cudaVectors;
	cudaVectors.VectorCount = 0;
	for( int i = 1; i < vectorCount; i++ ) {
		ASSERT_EXPR( vectors[i].GetMathEngine() == this );
		cudaVectors.Vectors[cudaVectors.VectorCount] = GetRaw( vectors[i] );
		cudaVectors.VectorCount++;

		if( cudaVectors.VectorCount == CCudaConstVectorArray::MaxSize || i + 1 == vectorCount ) {
			VectorFindMaxValueInSetWithIndicesKernel<<<blockCount, threadCount, 0, cudaStream>>>(cudaVectors,
				GetRaw(resultHandle), GetRaw(indexHandle), vectorSize, i - cudaVectors.VectorCount + 1);
			cudaVectors.VectorCount = 0;
		}
	}
}

void CCudaMathEngine::VectorSpreadValues(const CConstFloatHandle& sourceHandle, CFloatHandle* vectors, int vectorCount,
	const CConstIntHandle& indexHandle, int vectorSize)
{
	ASSERT_EXPR( sourceHandle.GetMathEngine() == this );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	CCudaVectorArray cudaVectors;
	cudaVectors.VectorCount = 0;
	for( int i = 0; i < vectorCount; i++ ) {
		ASSERT_EXPR( vectors[i].GetMathEngine() == this );
		cudaVectors.Vectors[cudaVectors.VectorCount] = GetRaw( vectors[i] );
		cudaVectors.VectorCount++;

		if( cudaVectors.VectorCount == CCudaVectorArray::MaxSize || i + 1 == vectorCount ) {
			VectorSpreadValuesKernel<<<blockCount, threadCount, 0, cudaStream>>>(GetRaw(sourceHandle), cudaVectors,
				GetRaw(indexHandle), vectorSize, i - cudaVectors.VectorCount + 1);
			cudaVectors.VectorCount = 0;
		}
	}
}

void CCudaMathEngine::VectorEltwiseLogSumExp(const CConstFloatHandle& first, const CConstFloatHandle& second,
	const CFloatHandle& result, int vectorSize)
{
	ASSERT_EXPR(first.GetMathEngine() == this);
	ASSERT_EXPR(second.GetMathEngine() == this);
	ASSERT_EXPR(result.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorEltwiseLogSumExpKernel<<<blockCount, threadCount, 0, cudaStream>>>(GetRaw(first),
		GetRaw(second), GetRaw(result), vectorSize);
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
