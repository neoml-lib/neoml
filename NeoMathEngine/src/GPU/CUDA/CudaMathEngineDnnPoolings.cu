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
#include <CudaMathEngineDnnPoolings.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>

#include <Kernels/CudaDnnPoolingKernels.h>
#include <Kernels/CudaDnn3dPoolingKernels.h>
#include <Kernels/CudaDnnGlobalPoolingKernels.h>
#include <Kernels/CudaDnnTimePoolingKernels.h>
#include <Kernels/CudaDnnGlobalTimePoolingKernels.h>

namespace NeoML {

CMaxPoolingDesc* CCudaMathEngine::InitMaxPooling( const CBlobDesc& source,
	int filterHeight, int filterWidth, int strideHeight, int strideWidth, const CBlobDesc& result )
{
	CCudaMaxPoolingDesc* desc = new CCudaMaxPoolingDesc();
	desc->Internal.Source = source;
	desc->Internal.Result = result;
	desc->Internal.FilterWidth = filterWidth;
	desc->Internal.FilterHeight = filterHeight;
	desc->Internal.StrideHeight = strideHeight;
	desc->Internal.StrideWidth = strideWidth;
	return desc;
}

void CCudaMathEngine::BlobMaxPooling(const CMaxPoolingDesc& poolingDesc, const CFloatHandle& sourceData,
	const CIntHandle* maxIndicesData, const CFloatHandle& resultData)
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData == 0 || maxIndicesData->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCudaMaxPoolingDescInternal& desc = static_cast<const CCudaMaxPoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& result = desc.Result;

	dim3 blockCount;
	dim3 threadCount;

	int* maxIndexPtr = 0;
	if(maxIndicesData != 0) {
		maxIndexPtr = GetRaw( *maxIndicesData );
	}
	getCudaTaskGrid3DMinZYX(1, 1, 32, blockCount, threadCount,
		result.ObjectCount(), result.Height() * result.Width(), result.Depth() * result.Channels());

	BlobMaxPoolingKernel<<<blockCount, threadCount, 0, cudaStream>>>( desc, GetRaw(sourceData), maxIndexPtr, GetRaw(resultData));
}

void CCudaMathEngine::BlobMaxPoolingBackward( const CMaxPoolingDesc& poolingDesc, const CFloatHandle& outputDiffData,
	const CIntHandle& maxIndicesData, const CFloatHandle& inputDiffData )
{
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData.GetMathEngine() == this );
	ASSERT_EXPR( inputDiffData.GetMathEngine() == this );

	const CCudaMaxPoolingDescInternal& desc = static_cast<const CCudaMaxPoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& inputDiff = desc.Source;
	const CCudaBlobDesc& outputDiff = desc.Result;

	VectorFill(inputDiffData, 0, inputDiff.BlobSize());

	bool isAtomic = desc.FilterHeight > desc.StrideHeight || desc.FilterWidth > desc.StrideWidth;
	int batchNorm = (outputDiff.ObjectCount() + BlobMaxPoolingBackwardCombine - 1) / BlobMaxPoolingBackwardCombine;

	dim3 blockCount;
	dim3 threadCount;

	int totalChannels = outputDiff.Depth() * outputDiff.Channels();

	getCudaTaskGrid3DMinZYX(1, 1, 32, blockCount, threadCount, batchNorm, outputDiff.Height() * outputDiff.Width(), totalChannels);

	BlobMaxPoolingBackwardKernel<<<blockCount, threadCount, 0, cudaStream>>>( desc, isAtomic, 
		GetRaw(outputDiffData), GetRaw(maxIndicesData), GetRaw(inputDiffData), batchNorm );
}

CMeanPoolingDesc* CCudaMathEngine::InitMeanPooling( const CBlobDesc& source,
	int filterHeight, int filterWidth, int strideHeight, int strideWidth, const CBlobDesc& result )
{
	CCudaMeanPoolingDesc* desc = new CCudaMeanPoolingDesc();
	desc->Internal.Source = source;
	desc->Internal.Result = result;
	desc->Internal.FilterHeight = filterHeight;
	desc->Internal.FilterWidth = filterWidth;
	desc->Internal.StrideHeight = strideHeight;
	desc->Internal.StrideWidth = strideWidth;
	return desc;
}

void CCudaMathEngine::BlobMeanPooling( const CMeanPoolingDesc& poolingDesc,
	const CFloatHandle& sourceData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCudaMeanPoolingDescInternal& desc = static_cast<const CCudaMeanPoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& result = desc.Result;

	dim3 blockCount;
	dim3 threadCount;

	int totalChannels = result.Depth() * result.Channels();

	getCudaTaskGrid3DMinZYX(1, 1, 32, blockCount, threadCount,
		result.ObjectCount(), result.Height() * result.Width(), totalChannels);
	BlobMeanPoolingKernel<<<blockCount, threadCount, 0, cudaStream>>>( desc, GetRaw(sourceData), GetRaw(resultData) );
}

void CCudaMathEngine::BlobMeanPoolingBackward( const CMeanPoolingDesc& poolingDesc, const CFloatHandle& outputDiffData,
	const CFloatHandle& inputDiffData )
{
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( inputDiffData.GetMathEngine() == this );

	const CCudaMeanPoolingDescInternal& desc = static_cast<const CCudaMeanPoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& outputDiff = desc.Result;
	const bool isAtomic = desc.FilterHeight > desc.StrideHeight || desc.FilterWidth > desc.StrideWidth;

	dim3 blockCount;
	dim3 threadCount;

	VectorFill( inputDiffData, 0, desc.Source.BlobSize() );

	getCudaTaskGrid3D( blockCount, threadCount, outputDiff.ObjectCount(), outputDiff.Height() * outputDiff.Width(),
		outputDiff.Depth() * outputDiff.Channels() );

	BlobMeanPoolingBackwardKernel<<<blockCount, threadCount, 0, cudaStream>>>( desc, GetRaw(outputDiffData),
		GetRaw(inputDiffData), isAtomic );
}

CGlobalMaxOverTimePoolingDesc* CCudaMathEngine::InitGlobalMaxOverTimePooling( const CBlobDesc& source, const CBlobDesc& result )
{
	CCudaGlobalMaxOverTimePoolingDesc* desc = new CCudaGlobalMaxOverTimePoolingDesc();
	desc->Internal.Source = source;
	desc->Internal.Result = result;
	return desc;
}

void CCudaMathEngine::BlobGlobalMaxOverTimePooling( const CGlobalMaxOverTimePoolingDesc& poolingDesc,
	const CFloatHandle& sourceData, const CIntHandle* maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData == 0 || maxIndicesData->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCudaGlobalMaxOverTimePoolingDescInternal& desc = static_cast<const CCudaGlobalMaxOverTimePoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& source = desc.Source;

	int objectCount = source.BatchLength();
	int objectSize = source.BlobSize() / objectCount;

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, objectSize);

	if( maxIndicesData == 0 ) {
		BlobGlobalMaxOverTimePoolingKernel<<<blockCount, threadCount, 0, cudaStream>>>( desc, GetRaw(sourceData), GetRaw(resultData) );
	} else {
		BlobGlobalMaxOverTimePoolingWithIndexKernel<<<blockCount, threadCount, 0, cudaStream>>>( desc, GetRaw(sourceData), GetRaw(*maxIndicesData), GetRaw(resultData) );
	}
}

void CCudaMathEngine::BlobGlobalMaxOverTimePoolingBackward( const CGlobalMaxOverTimePoolingDesc& poolingDesc,
	const CFloatHandle& sourceData, const CIntHandle& maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCudaGlobalMaxOverTimePoolingDescInternal& desc = static_cast<const CCudaGlobalMaxOverTimePoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& source = desc.Source;
	const CCudaBlobDesc& result = desc.Result;

	VectorFill(resultData, 0, result.BlobSize());

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, source.BlobSize());

	BlobGlobalMaxOverTimePoolingBackwardKernel<<<blockCount, threadCount, 0, cudaStream>>>( desc, GetRaw(sourceData), GetRaw(maxIndicesData), GetRaw(resultData) );
}

CGlobalMaxPoolingDesc* CCudaMathEngine::InitGlobalMaxPooling( const CBlobDesc& source, const CBlobDesc& maxIndices, const CBlobDesc& result )
{
	CCudaGlobalMaxPoolingDesc* desc = new CCudaGlobalMaxPoolingDesc();
	desc->Internal.Source = source;
	desc->Internal.MaxIndices = maxIndices;
	desc->Internal.Result = result;
	return desc;
}

void CCudaMathEngine::BlobGlobalMaxPooling( const CGlobalMaxPoolingDesc& poolingDesc, const CFloatHandle& sourceData,
	const CIntHandle& maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCudaGlobalMaxPoolingDescInternal& desc = static_cast<const CCudaGlobalMaxPoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& source = desc.Source;
	const CCudaBlobDesc& maxIndices = desc.MaxIndices;
	const CCudaBlobDesc& result = desc.Result;

	ASSERT_EXPR(source.ObjectCount() == result.ObjectCount() && maxIndices.ObjectCount() == result.ObjectCount());
	ASSERT_EXPR(maxIndices.ObjectSize() == result.ObjectSize());

	int poolSize = source.Depth() * source.Height() * source.Width();
	int maxCount = result.Depth() * result.Height() * result.Width();

	int poolSizeNorm = (poolSize + BlobGlobalMaxPoolingCombine - 1) / BlobGlobalMaxPoolingCombine;

	// As the shared memory size depends on maxCount, we may need to limit the number of threads
	int sharedMemoryPerThread = 4 * maxCount * sizeof(float);
	int maxThreadCount = device->SharedMemoryLimit / sharedMemoryPerThread;

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2DMinYX(1, device->ThreadMaxCount, blockCount, threadCount,
		source.ObjectCount() * source.Channels(), poolSizeNorm, maxThreadCount);
	blockCount.x = 1;

	int sharedSize = threadCount.y * threadCount.x * sharedMemoryPerThread;
	BlobGlobalMaxPoolingKernel<<<blockCount, threadCount, sharedSize, cudaStream>>>( desc, GetRaw( sourceData ),
		GetRaw( maxIndicesData ), GetRaw( resultData ), poolSize, maxCount, poolSizeNorm );
}

void CCudaMathEngine::BlobGlobalMaxPoolingBackward( const CGlobalMaxPoolingDesc& poolingDesc,
	const CFloatHandle& outputDiffData, const CIntHandle& maxIndicesData, const CFloatHandle& inputDiffData )
{
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData.GetMathEngine() == this );
	ASSERT_EXPR( inputDiffData.GetMathEngine() == this );

	const CCudaGlobalMaxPoolingDescInternal& desc = static_cast<const CCudaGlobalMaxPoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& inputDiff = desc.Source;
	const CCudaBlobDesc& outputDiff = desc.Result;

	VectorFill( inputDiffData, 0, inputDiff.BlobSize() );

	int poolSize = inputDiff.Depth() * inputDiff.Height() * inputDiff.Width();
	int maxCount = outputDiff.Depth() * outputDiff.Height() * outputDiff.Width();
	int fullSize = outputDiff.ObjectCount() * maxCount * outputDiff.Channels();

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, fullSize, BlobGlobalMaxPoolingBackwardCombine);

	BlobGlobalMaxPoolingBackwardKernel<<<blockCount, threadCount, 0, cudaStream>>>( desc, GetRaw( outputDiffData ),
		GetRaw( maxIndicesData ), GetRaw( inputDiffData ), poolSize, maxCount, fullSize );
}

C3dMaxPoolingDesc* CCudaMathEngine::Init3dMaxPooling( const CBlobDesc& source,
	int filterHeight, int filterWidth, int filterDepth,
	int strideHeight, int strideWidth, int strideDepth,
	const CBlobDesc& result )
{
	CCuda3dMaxPoolingDesc* desc = new CCuda3dMaxPoolingDesc();
	desc->Internal.Source = source;
	desc->Internal.Result = result;
	desc->Internal.FilterHeight = filterHeight;
	desc->Internal.FilterWidth = filterWidth;
	desc->Internal.FilterDepth = filterDepth;
	desc->Internal.StrideHeight = strideHeight;
	desc->Internal.StrideWidth = strideWidth;
	desc->Internal.StrideDepth = strideDepth;
	return desc;
}

void CCudaMathEngine::Blob3dMaxPooling( const C3dMaxPoolingDesc& poolingDesc, const CFloatHandle& sourceData,
	const CIntHandle* maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData == 0 || maxIndicesData->IsNull() || maxIndicesData->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCuda3dMaxPoolingDescInternal& desc = static_cast<const CCuda3dMaxPoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& result = desc.Result;

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid3DMinZYX(1, 1, 32, blockCount, threadCount, result.ObjectCount(),
		result.Depth() * result.Height() * result.Width(), result.Channels());

	Blob3dMaxPoolingKernel<<<blockCount, threadCount, 0, cudaStream>>>( desc, GetRaw( sourceData ),
		maxIndicesData == 0 ? 0 : GetRaw( *maxIndicesData ), GetRaw( resultData ) );
}

void CCudaMathEngine::Blob3dMaxPoolingBackward( const C3dMaxPoolingDesc& poolingDesc, const CFloatHandle& outputDiffData,
	const CIntHandle& maxIndicesData, const CFloatHandle& inputDiffData )
{
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData.GetMathEngine() == this );
	ASSERT_EXPR( inputDiffData.GetMathEngine() == this );

	const CCuda3dMaxPoolingDescInternal& desc = static_cast<const CCuda3dMaxPoolingDesc&>( poolingDesc ).Internal;
	VectorFill( inputDiffData, 0, desc.Source.BlobSize() );

	bool isAtomic = desc.FilterHeight > desc.StrideHeight || desc.FilterWidth > desc.StrideWidth || desc.FilterDepth > desc.StrideDepth;

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid3DMinZYX(1, 1, 32, blockCount, threadCount, desc.Result.ObjectCount(),
		desc.Result.Depth() * desc.Result.Height() * desc.Result.Width(), desc.Result.Channels());

	Blob3dMaxPoolingBackwardKernel<<<blockCount, threadCount, 0, cudaStream>>>( desc, GetRaw( outputDiffData ),
		GetRaw( maxIndicesData ), GetRaw( inputDiffData ), isAtomic );
}

C3dMeanPoolingDesc* CCudaMathEngine::Init3dMeanPooling( const CBlobDesc& source,
	int filterHeight, int filterWidth, int filterDepth,
	int strideHeight, int strideWidth, int strideDepth,
	const CBlobDesc& result )
{
	CCuda3dMeanPoolingDesc* desc = new CCuda3dMeanPoolingDesc();
	desc->Internal.Source = source;
	desc->Internal.Result = result;
	desc->Internal.FilterHeight = filterHeight;
	desc->Internal.FilterWidth = filterWidth;
	desc->Internal.FilterDepth = filterDepth;
	desc->Internal.StrideHeight = strideHeight;
	desc->Internal.StrideWidth = strideWidth;
	desc->Internal.StrideDepth = strideDepth;
	return desc;
}

void CCudaMathEngine::Blob3dMeanPooling( const C3dMeanPoolingDesc& poolingDesc, const CFloatHandle& sourceData,
	const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCuda3dMeanPoolingDescInternal& desc = static_cast<const CCuda3dMeanPoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& result = desc.Result;

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid3DMinZYX(1, 1, 32, blockCount, threadCount, result.ObjectCount(),
		result.Depth() * result.Height() * result.Width(), result.Channels());

	Blob3dMeanPoolingKernel<<<blockCount, threadCount, 0, cudaStream>>>( desc, GetRaw( sourceData ), GetRaw( resultData ) );
}

void CCudaMathEngine::Blob3dMeanPoolingBackward( const C3dMeanPoolingDesc& poolingDesc,
	const CFloatHandle& outputDiffData, const CFloatHandle& inputDiffData )
{
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( inputDiffData.GetMathEngine() == this );

	const CCuda3dMeanPoolingDescInternal& desc = static_cast<const CCuda3dMeanPoolingDesc&>( poolingDesc ).Internal;

	if( desc.FilterHeight != desc.StrideHeight || desc.FilterWidth != desc.StrideWidth || desc.FilterDepth != desc.StrideDepth ) {
		// Either the cube blocks used for pooling have nonzero intersections, and we need to add up several diffs,
		// or some of the data is skipped when pooling and we need to set diff = 0 for it
		VectorFill( inputDiffData, 0, desc.Source.BlobSize() );
	}

	// Indicates that the cube blocks used for pooling have nonzero intersections, and the diffs should be added up (atomicAdd)
	bool isAtomic = desc.FilterHeight > desc.StrideHeight || desc.FilterWidth > desc.StrideWidth || desc.FilterDepth > desc.StrideDepth;
	const CCudaBlobDesc& outputDiff = desc.Result;

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid3DMinZYX(1, 1, 32, blockCount, threadCount, outputDiff.ObjectCount(),
		outputDiff.Depth() * outputDiff.Height() * outputDiff.Width(), outputDiff.Channels());

	Blob3dMeanPoolingBackwardKernel<<<blockCount, threadCount, 0, cudaStream>>>( desc, GetRaw( outputDiffData ),
		GetRaw( inputDiffData ), isAtomic );
}

CMaxOverTimePoolingDesc* CCudaMathEngine::InitMaxOverTimePooling( const CBlobDesc& source,
	int filterLen, int strideLen, const CBlobDesc& result )
{
	CCudaMaxOverTimePoolingDesc* desc = new CCudaMaxOverTimePoolingDesc();
	desc->Internal.Source = source;
	desc->Internal.Result = result;
	desc->Internal.FilterLen = filterLen;
	desc->Internal.StrideLen = strideLen;
	return desc;
}

void CCudaMathEngine::BlobMaxOverTimePooling( const CMaxOverTimePoolingDesc& poolingDesc, const CFloatHandle& sourceData,
	const CIntHandle* maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData == 0 || maxIndicesData->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCudaMaxOverTimePoolingDescInternal& desc = static_cast<const CCudaMaxOverTimePoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& result = desc.Result;

	int xSize = (desc.FilterLen + BlobMaxOverTimePoolingCombine - 1) / BlobMaxOverTimePoolingCombine;
	xSize = alignXSizeForWarp(xSize);

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2DMinYX(1, device->ThreadMaxCount, blockCount, threadCount, result.BlobSize(), xSize);
	blockCount.x = 1; // in any case there may only one block along the X coordinate so that we can calculate the maximum inside one block

	int sharedSize = threadCount.x * threadCount.y * threadCount.z;

	if( maxIndicesData != 0 ) {
		BlobMaxOverTimePoolingKernel<<<blockCount, threadCount, sharedSize * sizeof(CValueWithIndex), cudaStream>>>( desc,
			GetRaw( sourceData ), GetRaw( *maxIndicesData ), GetRaw( resultData ) );
	} else {
		BlobMaxOverTimePoolingKernel<<<blockCount, threadCount, sharedSize * sizeof(float), cudaStream>>>( desc,
			GetRaw( sourceData ), GetRaw( resultData ) );
	}
}

void CCudaMathEngine::BlobMaxOverTimePoolingBackward( const CMaxOverTimePoolingDesc& poolingDesc, const CFloatHandle& outputDiffData,
	const CIntHandle& maxIndicesData, const CFloatHandle& inputDiffData )
{
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData.GetMathEngine() == this );
	ASSERT_EXPR( inputDiffData.GetMathEngine() == this );

	const CCudaMaxOverTimePoolingDescInternal& desc = static_cast<const CCudaMaxOverTimePoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& inputDiff = desc.Source;
	const CCudaBlobDesc& outputDiff = desc.Result;

	// Set diff to 0
	CCudaMathEngine::VectorFill( inputDiffData, 0, inputDiff.BlobSize() );

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, outputDiff.BlobSize(), BlobMaxOverTimePoolingBackwardCombine);

	if( desc.StrideLen >= desc.FilterLen ) {
		// The pooling areas do not intersect, no need to add
		CStoreSet store;
		BlobMaxOverTimePoolingBackwardKernel<<<blockCount, threadCount, 0, cudaStream>>>(store, desc, GetRaw( outputDiffData ),
			GetRaw( maxIndicesData ), GetRaw( inputDiffData ) );
	} else {
		CStoreAtomicAdd store;
		BlobMaxOverTimePoolingBackwardKernel<<<blockCount, threadCount, 0, cudaStream>>>(store, desc, GetRaw( outputDiffData ),
			GetRaw( maxIndicesData ), GetRaw( inputDiffData ));
	}
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
