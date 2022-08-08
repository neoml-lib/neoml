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
#include <CudaDevice.h>
#include <CudaAssert.h>
#include <CudaCommon.h>
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

void CCudaMathEngine::BroadcastCopy(const CIntHandle& toHandle, const CConstIntHandle& fromHandle,
	const CBlobDesc& toDesc, const CBlobDesc& fromDesc, int additionalWidth)
{
	ASSERT_EXPR(toHandle.GetMathEngine() == this);
	ASSERT_EXPR(fromHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	int resultSize = toDesc.BlobSize();
	getCudaTaskGrid(blockCount, threadCount, resultSize);

	VectorBroadcastCopyKernel<<<blockCount, threadCount>>>(
		GetRaw( toHandle ), GetRaw( fromHandle ), toDesc, fromDesc, additionalWidth, resultSize );
}

void CCudaMathEngine::BroadcastCopy(const CFloatHandle& toHandle, const CConstFloatHandle& fromHandle,
	const CBlobDesc& toDesc, const CBlobDesc& fromDesc, int additionalWidth)
{
	ASSERT_EXPR(toHandle.GetMathEngine() == this);
	ASSERT_EXPR(fromHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	int resultSize = toDesc.BlobSize();
	getCudaTaskGrid(blockCount, threadCount, resultSize);

	VectorBroadcastCopyKernel<<<blockCount, threadCount>>>(
		GetRaw( toHandle ), GetRaw( fromHandle ), toDesc, fromDesc, additionalWidth, resultSize );
}

void CCudaMathEngine::VectorFill(const CFloatHandle& result, float value, int vectorSize)
{
	ASSERT_EXPR(result.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorFillCombineCount);

	VectorFillKernel<<<blockCount, threadCount>>>(GetRaw(result), value, vectorSize);
}

void CCudaMathEngine::VectorFill(const CIntHandle& result, int value, int vectorSize)
{
	ASSERT_EXPR(result.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorFillCombineCount);

	VectorFillKernel<<<blockCount, threadCount>>>(GetRaw(result), value, vectorSize);
}

void CCudaMathEngine::VectorFill(const CFloatHandle& result, int vectorSize, const CConstFloatHandle& value)
{
	ASSERT_EXPR(result.GetMathEngine() == this);
	ASSERT_EXPR(value.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorFillHandleCombineCount);

	VectorFillHandleKernel<<<blockCount, threadCount>>>(GetRaw(result), vectorSize, GetRaw(value));
}

void CCudaMathEngine::VectorFill(const CIntHandle& result, int vectorSize, const CConstIntHandle& value)
{
	ASSERT_EXPR(result.GetMathEngine() == this);
	ASSERT_EXPR(value.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorFillHandleCombineCount);

	VectorFillHandleKernel<<<blockCount, threadCount>>>(GetRaw(result), vectorSize, GetRaw(value));
}

void CCudaMathEngine::VectorConvert(const CConstFloatHandle& from, const CIntHandle& to, int vectorSize)
{
	ASSERT_EXPR(from.GetMathEngine() == this);
	ASSERT_EXPR(to.GetMathEngine() == this);
	ASSERT_EXPR(vectorSize >= 0);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorConvertCombineCount);

	VectorConvertKernel<<<blockCount, threadCount>>>(GetRaw(from), GetRaw(to), vectorSize);
}

void CCudaMathEngine::VectorConvert(const CConstIntHandle& from, const CFloatHandle& to, int vectorSize)
{
	ASSERT_EXPR(from.GetMathEngine() == this);
	ASSERT_EXPR(to.GetMathEngine() == this);
	ASSERT_EXPR(vectorSize >= 0);

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorConvertCombineCount);

	VectorConvertKernel<<<blockCount, threadCount>>>(GetRaw(from), GetRaw(to), vectorSize);
}

void CCudaMathEngine::VectorFillBernoulli( const CFloatHandle& result, float p, int vectorSize, float valueHandle, int seed )
{
	ASSERT_EXPR(result.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, ( vectorSize + 3 ) / 4, VectorFillBernoulliCombine );

	VectorFillBernoulliKernel<<<blockCount, threadCount>>>( GetRaw( result ), p, vectorSize,
		valueHandle, seed );
}

void CCudaMathEngine::FilterSmallValues( const CFloatHandle& data, int dataSize, float threshold )
{
	ASSERT_EXPR(data.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, dataSize, VectorFillCombineCount );

	FilterSmallValuesKernel<<<blockCount, threadCount>>>( GetRaw( data ), threshold, dataSize );
}

void CCudaMathEngine::VectorSum(const CConstFloatHandle& firstHandle, int vectorSize, const CFloatHandle& resultHandle)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorSumCombineCount);

	bool setZero = true;
	if(blockCount > 1) {
		setZero = false;
		resultHandle.SetValue(0);
	}

	const int sharedSize = threadCount * sizeof(float);
	VectorSumKernel<<<blockCount, threadCount, sharedSize>>>(GetRaw(firstHandle), vectorSize,
		GetRaw(resultHandle), false, setZero);
}

void CCudaMathEngine::VectorSumAdd(const CConstFloatHandle& firstHandle, int vectorSize, const CFloatHandle& resultHandle)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorSumCombineCount);

	const int sharedSize = threadCount * sizeof(float);
	VectorSumKernel<<<blockCount, threadCount, sharedSize>>>(GetRaw(firstHandle), vectorSize,
		GetRaw(resultHandle), false, false);
}

void CCudaMathEngine::VectorSumAlongDimension( const CConstFloatHandle& firstHandle, int precedingDimension, int dimension,
	int followingDimension, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2D( blockCount, threadCount, precedingDimension, followingDimension );

	VectorSumAlongDimensionKernel<<<blockCount, threadCount>>>
		( GetRaw( firstHandle ), precedingDimension, dimension, followingDimension, GetRaw( resultHandle ) );
}

void CCudaMathEngine::VectorCumSumAlongDimension( const CConstFloatHandle& firstHandle, int precedingDimension, int dimension,
	int followingDimension, const CFloatHandle& resultHandle, bool reverse )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2D( blockCount, threadCount, precedingDimension, followingDimension );

	VectorCumSumAlongDimensionKernel<<<blockCount, threadCount>>>
		( GetRaw( firstHandle ), precedingDimension, dimension, followingDimension, GetRaw( resultHandle ), reverse );
}

void CCudaMathEngine::VectorCumSumAlongDimension( const CConstIntHandle& firstHandle, int precedingDimension, int dimension,
	int followingDimension, const CIntHandle& resultHandle, bool reverse )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2D( blockCount, threadCount, precedingDimension, followingDimension );

	VectorCumSumAlongDimensionKernel<<<blockCount, threadCount>>>
		( GetRaw( firstHandle ), precedingDimension, dimension, followingDimension, GetRaw( resultHandle ), reverse );
}

void CCudaMathEngine::VectorSumAlongDimensionDiag( const CConstFloatHandle& firstHandle, int precedingDimension, int dimension,
	int followingDimension, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2D( blockCount, threadCount, precedingDimension, followingDimension );

	VectorFill( resultHandle, 0.0, precedingDimension * precedingDimension * dimension
		* followingDimension * followingDimension );
	VectorSumAlongDimensionDiagKernel<<<blockCount, threadCount>>>
		( GetRaw( firstHandle ), precedingDimension, dimension, followingDimension, GetRaw( resultHandle ) );
}

void CCudaMathEngine::VectorCumSumAlongDimensionDiag( const CConstFloatHandle& firstHandle, int precedingDimension, int dimension,
	int followingDimension, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2D( blockCount, threadCount, precedingDimension, dimension * followingDimension );

	VectorFill( resultHandle, 0.0, precedingDimension * precedingDimension * dimension
		* dimension * followingDimension * followingDimension );
	VectorCumSumAlongDimensionDiagKernel<<<blockCount, threadCount>>>
		( GetRaw( firstHandle ), precedingDimension, dimension, followingDimension, GetRaw( resultHandle ) );
}

void CCudaMathEngine::VectorEqual( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorEqualCombineCount );

	VectorEqualKernel <<<blockCount, threadCount>>>
		(GetRaw( firstHandle ), GetRaw( secondHandle ), GetRaw( resultHandle ), vectorSize);
}

void CCudaMathEngine::VectorEqualValue( const CConstIntHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstIntHandle& valueHandle )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(valueHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorEqualCombineCount );

	VectorEqualValueKernel<<<blockCount, threadCount>>>
		(GetRaw( firstHandle ), GetRaw( resultHandle ), vectorSize, GetRaw( valueHandle ));
}

void CCudaMathEngine::VectorMax( const CConstFloatHandle& firstHandle, float secondValue, const CFloatHandle& resultHandle,
	int vectorSize )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	ASSERT_EXPR(vectorSize >= 0);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorMaxCombineCount );

	VectorMaxKernel<<<blockCount, threadCount>>>
		( GetRaw( firstHandle ), secondValue, GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorMaxDiff( const CConstFloatHandle& firstHandle, float secondValue, const CFloatHandle& gradHandle,
	int gradHeight, int gradWidth )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(gradHandle.GetMathEngine() == this);
	ASSERT_EXPR(gradHeight >= 0);
	ASSERT_EXPR(gradWidth > 0);
	SetCudaDevice( device->DeviceNumber );

	const int firstSize = gradHeight == 1 ? gradWidth : gradHeight;
	const int gradSize = gradHeight == 1 ? 1 : gradWidth;
	const int gradNorm = ( gradSize + VectorMaxDiffCombineCount - 1 ) / VectorMaxDiffCombineCount;

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2D( blockCount, threadCount, firstSize, gradNorm );

	VectorMaxDiffKernel<<<blockCount, threadCount>>>( GetRaw( gradHandle ),
		firstSize, gradSize, gradNorm, GetRaw( firstHandle ), secondValue );
}

void CCudaMathEngine::VectorELU( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize, const CConstFloatHandle& alpha )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(alpha.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorActivationCombineCount );

	VectorELUKernel<<<blockCount, threadCount>>>( GetRaw( firstHandle ),
		GetRaw( resultHandle ), vectorSize, GetRaw( alpha ) );
}

void CCudaMathEngine::VectorELUDiff( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& alpha )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	ASSERT_EXPR(alpha.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorActivationCombineCount );

	VectorELUDiffKernel<<<blockCount, threadCount>>>( GetRaw( firstHandle ), GetRaw( secondHandle ),
		GetRaw( resultHandle ), vectorSize, GetRaw( alpha ) );
}

void CCudaMathEngine::VectorELUDiffOp( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& alpha )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	ASSERT_EXPR(alpha.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorActivationCombineCount );

	VectorELUDiffOpKernel<<<blockCount, threadCount>>>( GetRaw( firstHandle ), GetRaw( secondHandle ),
		GetRaw( resultHandle ), vectorSize,	GetRaw( alpha ) );
}

void CCudaMathEngine::VectorReLU(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize, const CConstFloatHandle& upperThresholdHandle)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	ASSERT_EXPR(upperThresholdHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorReLUKernel<<<blockCount, threadCount>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize,
		GetRaw(upperThresholdHandle));
}

void CCudaMathEngine::VectorReLUDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& upperThresholdHandle)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	ASSERT_EXPR(upperThresholdHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorReLUDiffKernel<<<blockCount, threadCount>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize, GetRaw(upperThresholdHandle));
}

void CCudaMathEngine::VectorLeakyReLU( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize, const CConstFloatHandle& alpha )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	ASSERT_EXPR(alpha.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorActivationCombineCount );

	VectorLeakyReLUKernel<<<blockCount, threadCount>>>( GetRaw( firstHandle ),
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
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorActivationCombineCount );

	VectorLeakyReLUDiffKernel<<<blockCount, threadCount>>>
		( GetRaw( firstHandle ), GetRaw( secondHandle ), GetRaw( resultHandle ), vectorSize,
			GetRaw( alpha ) );
}

void CCudaMathEngine::VectorHSwish( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorActivationCombineCount );

	VectorHSwishKernel<<<blockCount, threadCount>>>
		( GetRaw( firstHandle ), GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorHSwishDiff( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorActivationCombineCount );

	VectorHSwishDiffKernel<<<blockCount, threadCount>>>
		( GetRaw( firstHandle ), GetRaw( secondHandle ), GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorEltwiseMax(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorEltwiseMaxCombineCount);

	VectorEltwiseMaxKernel<<<blockCount, threadCount>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorEltwiseMin(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorEltwiseMinCombineCount);

	VectorEltwiseMinKernel<<<blockCount, threadCount>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorAbs(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorAbsKernel<<<blockCount, threadCount>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorAbsDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorAbsDiffKernel<<<blockCount, threadCount>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorHinge(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorHingeKernel<<<blockCount, threadCount>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorHingeDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorHingeDiffKernel<<<blockCount, threadCount>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}
void CCudaMathEngine::VectorSquaredHinge(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorSquaredHingeKernel<<<blockCount, threadCount>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorSquaredHingeDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorSquaredHingeDiffKernel<<<blockCount, threadCount>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorHuber(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorHuberKernel<<<blockCount, threadCount>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorHuberDerivative(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorHuberDiffKernel<<<blockCount, threadCount>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorHardTanh(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorHardTanhKernel<<<blockCount, threadCount>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorHardTanhDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorHardTanhDiffKernel<<<blockCount, threadCount>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorHardSigmoid( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize, 
	const CConstFloatHandle& slopeHandle, const CConstFloatHandle& biasHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorActivationCombineCount );

	VectorHardSigmoidKernel<<<blockCount, threadCount>>>
		( GetRaw( firstHandle ), GetRaw( resultHandle ), vectorSize, GetRaw( slopeHandle ), GetRaw( biasHandle ) );
}

void CCudaMathEngine::VectorHardSigmoidDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& slopeHandle, const CConstFloatHandle& biasHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorHardSigmoidDiffKernel<<<blockCount, threadCount>>>
		( GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize, GetRaw( slopeHandle ), GetRaw( biasHandle ) );
}

void CCudaMathEngine::VectorHardSigmoidDiffOp( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& slopeHandle, const CConstFloatHandle& /*biasHandle*/ )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorActivationCombineCount );

	VectorHardSigmoidDiffOpKernel<<<blockCount, threadCount>>>
		( GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize, GetRaw( slopeHandle ) );
}

void CCudaMathEngine::VectorNeg(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize );

	VectorNegKernel<<<blockCount, threadCount>>>( GetRaw( firstHandle ), GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorExp(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorExpKernel<<<blockCount, threadCount>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorLog( const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize );

	VectorLogKernel<<<blockCount, threadCount>>>( GetRaw( firstHandle ),
		GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorLogDiff( const CConstFloatHandle& sourceGradHandle, int sourceGradHeight, int sourceGradWidth,
	const CConstFloatHandle& valueHandle, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR(sourceGradHandle.GetMathEngine() == this);
	ASSERT_EXPR(valueHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	const int firstSize = sourceGradHeight == 1 ? sourceGradWidth : sourceGradHeight;
	const int gradSize = sourceGradHeight == 1 ? 1 : sourceGradWidth;
	const int gradNorm = ( gradSize + VectorLogDiffCombine - 1 ) / VectorLogDiffCombine;

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2D( blockCount, threadCount, firstSize, gradNorm );

	VectorLogDiffKernel<<<blockCount, threadCount>>>( GetRaw( sourceGradHandle ), firstSize, gradSize, gradNorm,
		GetRaw( valueHandle ), GetRaw( resultHandle ) );
}

void CCudaMathEngine::VectorNegLog(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorNegLogKernel<<<blockCount, threadCount>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorErf(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorErfKernel<<<blockCount, threadCount>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorBernulliKLDerivative(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& targetHandle)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(targetHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorBernulliKLDerivativeKernel<<<blockCount, threadCount>>>
		(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize, GetRaw(targetHandle));
}

void CCudaMathEngine::VectorAdd(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorAddCombineCount);

	VectorAddKernel<<<blockCount, threadCount>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorAdd( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
	const CIntHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorAddCombineCount );

	VectorAddKernel<<<blockCount, threadCount>>>
		( GetRaw( firstHandle ), GetRaw( secondHandle ), GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorAddValue(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& additionHandle)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(additionHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorAddValueCombineCount);

	VectorAddValueKernel<<<blockCount, threadCount>>>
		(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize, GetRaw(additionHandle));
}

void CCudaMathEngine::VectorAddValue(
	const CConstIntHandle& firstHandle, const CIntHandle& resultHandle,
	int vectorSize, const CConstIntHandle& additionHandle )
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(additionHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorAddValueCombineCount );

	VectorAddValueKernel<<<blockCount, threadCount>>>(
		GetRaw( firstHandle ), GetRaw( resultHandle ), vectorSize,
		GetRaw( additionHandle ) );
}

void CCudaMathEngine::VectorSub(const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
	const CIntHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorSubCombineCount );

	VectorSubKernel<<<blockCount, threadCount>>>
		( GetRaw( firstHandle ), GetRaw( secondHandle ), GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorSub(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorSubCombineCount );

	VectorSubKernel<<<blockCount, threadCount>>>
		( GetRaw( firstHandle ), GetRaw( secondHandle ), GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorSub(const CConstFloatHandle& firstHandle, float second, const CFloatHandle& resultHandle,
	int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorSubCombineCount );

	VectorSubKernel<<<blockCount, threadCount>>>
		( GetRaw( firstHandle ), second, GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorSub(float first, const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle,
	int vectorSize)
{
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorSubCombineCount );

	VectorSubKernel<<<blockCount, threadCount>>>
		( first, GetRaw( secondHandle ), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorMultiplyAndSub(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( multHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorMultiplyAndSubKernel<<<blockCount, threadCount>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize, GetRaw(multHandle));
}

void CCudaMathEngine::VectorMultiply(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multiplierHandle)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(multiplierHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorMultiplyCombineCount);

	VectorMultiplyKernel<<<blockCount, threadCount>>>
		(GetRaw(firstHandle),GetRaw(resultHandle), vectorSize, GetRaw(multiplierHandle));
}

void CCudaMathEngine::VectorMultiply(const CConstIntHandle& firstHandle,
	const CIntHandle& resultHandle, int vectorSize, const CConstIntHandle& multiplierHandle)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(multiplierHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorMultiplyCombineCount);

	VectorMultiplyKernel<<<blockCount, threadCount>>>
		(GetRaw(firstHandle),GetRaw(resultHandle), vectorSize, GetRaw(multiplierHandle));
}

void CCudaMathEngine::VectorNegMultiply(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multiplierHandle)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(multiplierHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorMultiplyCombineCount);

	VectorNegMultiplyKernel<<<blockCount, threadCount>>>
		(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize, GetRaw(multiplierHandle));
}

void CCudaMathEngine::VectorEltwiseMultiply(const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
	const CIntHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorEltwiseMultiplyCombineCount);

	VectorEltwiseMultiplyKernel<<<blockCount, threadCount>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorEltwiseMultiply(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorEltwiseMultiplyCombineCount);

	VectorEltwiseMultiplyKernel<<<blockCount, threadCount>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorEltwiseMultiplyAdd(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorEltwiseMultiplyAddCombineCount);

	VectorEltwiseMultiplyAddKernel<<<blockCount, threadCount>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorEltwiseNegMultiply(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorEltwiseNegMultiplyCombineCount);

	VectorEltwiseNegMultiplyKernel<<<blockCount, threadCount>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorEltwiseDivide(const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
	const CIntHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorEltwiseDivideCombineCount);

	VectorEltwiseDivideKernel<<<blockCount, threadCount>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorEltwiseDivide(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorEltwiseDivideCombineCount);

	VectorEltwiseDivideKernel<<<blockCount, threadCount>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorEltwisePower(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorEltwisePowerKernel<<<blockCount, threadCount>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorSqrt(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorSqrtKernel<<<blockCount, threadCount>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorInv(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorInvCombineCount);

	VectorInvKernel<<<blockCount, threadCount>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorMinMax(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize,
	const CConstFloatHandle& minHandle, const CConstFloatHandle& maxHandle)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(minHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	ASSERT_EXPR(maxHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorMinMaxCombineCount);

	VectorMinMaxKernel<<<blockCount, threadCount>>>(GetRaw(firstHandle), GetRaw(resultHandle),
		vectorSize, GetRaw(minHandle), GetRaw(maxHandle));
}

void CCudaMathEngine::VectorSigmoid(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorSigmoidKernel<<<blockCount, threadCount>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorSigmoidDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorSigmoidDiffKernel<<<blockCount, threadCount>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorSigmoidDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorSigmoidDiffOpCombineCount);

	VectorSigmoidDiffOpKernel<<<blockCount, threadCount>>>(GetRaw(firstHandle), GetRaw(secondHandle),
		GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorTanh(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorTanhKernel<<<blockCount, threadCount>>>(GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorTanhDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorTanhDiffKernel<<<blockCount, threadCount>>>
		(GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorTanhDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorTanhDiffOpCombineCount);

	VectorTanhDiffOpKernel<<<blockCount, threadCount>>>(GetRaw(firstHandle), GetRaw(secondHandle),
		GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorPower(float exponent, const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorPowerKernel<<<blockCount, threadCount>>>(exponent, GetRaw(firstHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorPowerDiff(float exponent, const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorPowerDiffKernel<<<blockCount, threadCount>>>
		(exponent, GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize);
}

void CCudaMathEngine::VectorPowerDiffOp(float exponent, const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(firstHandle.GetMathEngine() == this);
	ASSERT_EXPR(secondHandle.GetMathEngine() == this);
	ASSERT_EXPR(resultHandle.GetMathEngine() == this);
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize);

	VectorPowerDiffOpKernel<<<blockCount, threadCount>>>
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
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, vectorSize, VectorActivationCombineCount);

	VectorL1DiffAddKernel<<<blockCount, threadCount>>>(GetRaw(firstHandle), GetRaw(secondHandle),
		GetRaw(resultHandle), vectorSize, GetRaw(hubertThresholdHandle), GetRaw(multHandle));
}

void CCudaMathEngine::VectorEltwiseNot( const CConstIntHandle& firstHandle, const CIntHandle& resultHandle,
	int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize );
	vectorNotKernel<<<blockCount, threadCount>>>( GetRaw( firstHandle ), GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorEltwiseNotNegative( const CConstIntHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize );

	vectorGreaterEqualToZeroKernel<<<blockCount, threadCount>>>( GetRaw( firstHandle ),
		GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorEltwiseLess( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize );

	vectorLessKernel<<<blockCount, threadCount>>>( GetRaw( firstHandle ), GetRaw( secondHandle ),
		GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorEltwiseLess( const CConstFloatHandle& firstHandle, float second,
	const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize );

	vectorLessKernel<<<blockCount, threadCount>>>( GetRaw( firstHandle ), second,
		GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorEltwiseLess( float first, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize );

	vectorLessKernel<<<blockCount, threadCount>>>( first, GetRaw( secondHandle ),
		GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorEltwiseLess( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CIntHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize );

	vectorLessKernel<<<blockCount, threadCount>>>( GetRaw( firstHandle ), GetRaw( secondHandle ),
		GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorEltwiseLess( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
	const CIntHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize );

	vectorLessKernel<<<blockCount, threadCount>>>( GetRaw( firstHandle ), GetRaw( secondHandle ),
		GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorEltwiseEqual( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CIntHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize );

	vectorEqualKernel<<<blockCount, threadCount>>>( GetRaw( firstHandle ), GetRaw( secondHandle ),
		GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorEltwiseEqual( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
	const CIntHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize );

	vectorEqualKernel<<<blockCount, threadCount>>>( GetRaw( firstHandle ), GetRaw( secondHandle ),
		GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorEltwiseWhere( const CConstIntHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CConstFloatHandle& thirdHandle, const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( thirdHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize );

	vectorWhereKernel<<<blockCount, threadCount>>>( GetRaw( firstHandle ), GetRaw( secondHandle ),
		GetRaw( thirdHandle ), GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorEltwiseWhere( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
	const CConstIntHandle& thirdHandle, const CIntHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( thirdHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize );

	vectorWhereKernel<<<blockCount, threadCount>>>( GetRaw( firstHandle ), GetRaw( secondHandle ),
		GetRaw( thirdHandle ), GetRaw( resultHandle ), vectorSize );
}

void CCudaMathEngine::VectorFindMaxValueInSet(const CConstFloatHandle* vectors, int vectorCount,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectors[0].GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

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
			VectorFindMaxValueInSetKernel<<<blockCount, threadCount>>>(cudaVectors,
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
	SetCudaDevice( device->DeviceNumber );

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
			VectorFindMaxValueInSetWithIndicesKernel<<<blockCount, threadCount>>>(cudaVectors,
				GetRaw(resultHandle), GetRaw(indexHandle), vectorSize, i - cudaVectors.VectorCount + 1);
			cudaVectors.VectorCount = 0;
		}
	}
}

void CCudaMathEngine::VectorSpreadValues(const CConstFloatHandle& sourceHandle, CFloatHandle* vectors, int vectorCount,
	const CConstIntHandle& indexHandle, int vectorSize)
{
	ASSERT_EXPR( sourceHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

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
			VectorSpreadValuesKernel<<<blockCount, threadCount>>>(GetRaw(sourceHandle), cudaVectors,
				GetRaw(indexHandle), vectorSize, i - cudaVectors.VectorCount + 1);
			cudaVectors.VectorCount = 0;
		}
	}
}

void CCudaMathEngine::VectorTopK(const CConstFloatHandle& firstHandle, int firstSize, int k, const CFloatHandle& resultHandle,
	const CIntHandle& indicesHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( firstSize >= 0 );
	ASSERT_EXPR( k > 0 );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( indicesHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	CGlobalMaxPoolingDesc* desc = InitGlobalMaxPooling( { firstSize, 1 }, { k, 1 }, { k, 1 } );
	BlobGlobalMaxPooling( *desc, firstHandle, indicesHandle, resultHandle );
	delete desc;
}

void CCudaMathEngine::VectorTopKDiff( const CConstFloatHandle& sourceGrad, int sourceGradHeight, int sourceGradWidth,
		const CConstIntHandle& indices, int k, const CFloatHandle& resultGrad )
{
	ASSERT_EXPR( sourceGrad.GetMathEngine() == this );
	ASSERT_EXPR( sourceGradHeight > 0 );
	ASSERT_EXPR( sourceGradWidth > 0 );
	ASSERT_EXPR( indices.GetMathEngine() == this );
	ASSERT_EXPR( k > 0 );
	ASSERT_EXPR( resultGrad.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	if( sourceGradHeight == 1 ){
		VectorFill( resultGrad, 0, k * sourceGradWidth );

		int blockCount;
		int threadCount;
		getCudaTaskGrid( blockCount, threadCount, k );

		VectorTopKDiffKernel<<<blockCount, threadCount>>>( GetRaw( sourceGrad ), GetRaw( indices ),
			GetRaw( resultGrad ), k, sourceGradWidth );
	} else {
		CLookupDimension dimension( sourceGradHeight, sourceGradWidth );
		VectorMultichannelLookupAndCopy( k, 1, indices,
			&sourceGrad, &dimension, 1, resultGrad, sourceGradWidth );
	}
}

void CCudaMathEngine::VectorAbsDiff(const CConstFloatHandle& sourceGradHandle, int gradHeight, int gradWidth,
	const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle)
{
	ASSERT_EXPR( sourceGradHandle.GetMathEngine() == this );
	ASSERT_EXPR( gradHeight > 0 );
	ASSERT_EXPR( gradWidth > 0 );
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const int firstSize = gradHeight == 1 ? gradWidth : gradHeight;
	const int gradSize = gradHeight == 1 ? 1 : gradWidth;
	const int gradNorm = ( gradSize + VectorAbsDiffCombine - 1 ) / VectorAbsDiffCombine;

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2D( blockCount, threadCount, firstSize, gradNorm );

	VectorAbsDiffKernel<<<blockCount, threadCount>>>( GetRaw( sourceGradHandle ), firstSize, gradSize, gradNorm,
		GetRaw( firstHandle ), GetRaw( resultHandle ) );
}

void CCudaMathEngine::VectorMinMaxDiff(const CConstFloatHandle& sourceGradHandle, int gradHeight, int gradWidth,
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
	SetCudaDevice( device->DeviceNumber );

	const int firstSize = gradHeight == 1 ? gradWidth : gradHeight;
	const int gradSize = gradHeight == 1 ? 1 : gradWidth;
	const int gradNorm = ( gradSize + VectorMinMaxDiffCombine - 1 ) / VectorMinMaxDiffCombine;

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2D( blockCount, threadCount, firstSize, gradNorm );

	VectorMinMaxDiffKernel<<<blockCount, threadCount>>>( GetRaw( sourceGradHandle ), firstSize, gradSize, gradNorm,
		GetRaw( firstHandle ), GetRaw( resultHandle ), GetRaw( minHandle ), GetRaw( maxHandle ) );
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
