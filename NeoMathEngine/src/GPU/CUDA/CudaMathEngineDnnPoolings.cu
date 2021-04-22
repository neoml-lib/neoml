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
#include <CudaCommon.h>
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
	SetCudaDevice( device->DeviceNumber );

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

	BlobMaxPoolingKernel<<<blockCount, threadCount>>>( desc, GetRaw(sourceData), maxIndexPtr, GetRaw(resultData));
}

void CCudaMathEngine::BlobMaxPoolingBackward( const CMaxPoolingDesc& poolingDesc, const CFloatHandle& outputDiffData,
	const CIntHandle& maxIndicesData, const CFloatHandle& inputDiffData )
{
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData.GetMathEngine() == this );
	ASSERT_EXPR( inputDiffData.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

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

	BlobMaxPoolingBackwardKernel<<<blockCount, threadCount>>>( desc, isAtomic, 
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
	SetCudaDevice( device->DeviceNumber );

	const CCudaMeanPoolingDescInternal& desc = static_cast<const CCudaMeanPoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& result = desc.Result;

	dim3 blockCount;
	dim3 threadCount;

	int totalChannels = result.Depth() * result.Channels();

	getCudaTaskGrid3DMinZYX(1, 1, 32, blockCount, threadCount,
		result.ObjectCount(), result.Height() * result.Width(), totalChannels);
	BlobMeanPoolingKernel<<<blockCount, threadCount>>>( desc, GetRaw(sourceData), GetRaw(resultData) );
}

void CCudaMathEngine::BlobMeanPoolingBackward( const CMeanPoolingDesc& poolingDesc, const CFloatHandle& outputDiffData,
	const CFloatHandle& inputDiffData )
{
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( inputDiffData.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const CCudaMeanPoolingDescInternal& desc = static_cast<const CCudaMeanPoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& outputDiff = desc.Result;
	const bool isAtomic = desc.FilterHeight > desc.StrideHeight || desc.FilterWidth > desc.StrideWidth;

	dim3 blockCount;
	dim3 threadCount;

	VectorFill( inputDiffData, 0, desc.Source.BlobSize() );

	getCudaTaskGrid3D( blockCount, threadCount, outputDiff.ObjectCount(), outputDiff.Height() * outputDiff.Width(),
		outputDiff.Depth() * outputDiff.Channels() );

	BlobMeanPoolingBackwardKernel<<<blockCount, threadCount>>>( desc, GetRaw(outputDiffData),
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
	SetCudaDevice( device->DeviceNumber );

	const CCudaGlobalMaxOverTimePoolingDescInternal& desc = static_cast<const CCudaGlobalMaxOverTimePoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& source = desc.Source;

	int objectCount = source.BatchLength();
	int objectSize = source.BlobSize() / objectCount;

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, objectSize);

	if( maxIndicesData == 0 ) {
		BlobGlobalMaxOverTimePoolingKernel<<<blockCount, threadCount>>>( desc, GetRaw(sourceData), GetRaw(resultData) );
	} else {
		BlobGlobalMaxOverTimePoolingWithIndexKernel<<<blockCount, threadCount>>>( desc, GetRaw(sourceData), GetRaw(*maxIndicesData), GetRaw(resultData) );
	}
}

void CCudaMathEngine::BlobGlobalMaxOverTimePoolingBackward( const CGlobalMaxOverTimePoolingDesc& poolingDesc,
	const CFloatHandle& sourceData, const CIntHandle& maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const CCudaGlobalMaxOverTimePoolingDescInternal& desc = static_cast<const CCudaGlobalMaxOverTimePoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& source = desc.Source;
	const CCudaBlobDesc& result = desc.Result;

	VectorFill(resultData, 0, result.BlobSize());

	int blockCount;
	int threadCount;
	getCudaTaskGrid(blockCount, threadCount, source.BlobSize());

	BlobGlobalMaxOverTimePoolingBackwardKernel<<<blockCount, threadCount>>>( desc, GetRaw(sourceData), GetRaw(maxIndicesData), GetRaw(resultData) );
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
	SetCudaDevice( device->DeviceNumber );

	const CCudaGlobalMaxPoolingDescInternal& desc = static_cast<const CCudaGlobalMaxPoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& source = desc.Source;
	const CCudaBlobDesc& maxIndices = desc.MaxIndices;
	const CCudaBlobDesc& result = desc.Result;

	ASSERT_EXPR(source.ObjectCount() == result.ObjectCount() && maxIndices.ObjectCount() == result.ObjectCount());
	ASSERT_EXPR(maxIndices.ObjectSize() == result.ObjectSize());

	int poolSize = source.Depth() * source.Height() * source.Width();
	int maxCount = result.Depth() * result.Height() * result.Width();

	if( maxCount < 100 ){
		// As the shared memory size depends on maxCount, we may need to limit the number of thread
		int sharedMemoryPerThread = 2 * maxCount * sizeof( float );
		int maxThreadCount = device->SharedMemoryLimit / sharedMemoryPerThread;
		dim3 blockCount;
		dim3 threadCount;

		getCudaTaskGrid2DMinYX(device->ThreadMax3DCountY, 1, blockCount, threadCount,
			( poolSize + maxCount - 1 ) / maxCount + 1, source.ObjectCount() * source.Channels(), maxThreadCount);
		blockCount.y = 1;
		threadCount.y--;

		int sharedSize = threadCount.x * ( threadCount.y + 1 ) * sharedMemoryPerThread;
		BlobGlobalMaxPoolingHeapKernel<<<blockCount, threadCount, sharedSize>>>( desc, GetRaw( sourceData ),
			GetRaw( maxIndicesData ), GetRaw( resultData ), poolSize, maxCount );
	} else {
		dim3 blockCount;
		dim3 threadCount;
		dim3 scanBlockCount;
		dim3 scanThreadCount;

		int bitsPerBin = 8;
		int histSize = ( 1 << bitsPerBin );
		int memoryPerThread = histSize * sizeof( int );
		int maxThreadCount = device->SharedMemoryLimit / memoryPerThread;

		int height = 1;
		int poolSizeNorm = max( 1, poolSize / ( 2 * histSize ) );
		while( height < poolSizeNorm ) {
			height *= 2;
		}

		getCudaTaskGrid2DMinYX(device->ThreadMax3DCountY, 1, blockCount, threadCount,
			height, source.ObjectCount() * source.Channels(), maxThreadCount);

		getCudaTaskGrid2DMinYX(device->ThreadMax3DCountY, 1, scanBlockCount, scanThreadCount,
			blockCount.y, source.ObjectCount() * source.Channels(), maxThreadCount);
		scanBlockCount.y = 1;

		CIntHandleVar indicesSorted1( mathEngine(), source.BlobSize() );
		CIntHandleVar indicesSorted2( mathEngine(), source.BlobSize() );
		CIntHandleVar localSums( mathEngine(), blockCount.x * threadCount.x * blockCount.y * histSize );
		CIntHandleVar globalSums( mathEngine(), blockCount.x * threadCount.x * ( blockCount.y + 1 ) * histSize );
		int localSortSharedSize = threadCount.y * threadCount.x * memoryPerThread;
		int scanSharedSize = scanThreadCount.x * scanThreadCount.y  * memoryPerThread;

		int bitCount = sizeof(float) * 8;
		for( int bin = 0; bin < bitCount; bin += bitsPerBin ) {
			// local sort inside block
			BlobGlobalMaxPoolingLocalSortKernel<<<blockCount, threadCount, localSortSharedSize>>>( desc, GetRaw( sourceData ),
				GetRaw( indicesSorted1.GetHandle() ), GetRaw( indicesSorted2.GetHandle() ), poolSize, bin, histSize, GetRaw( localSums.GetHandle() ), GetRaw( globalSums.GetHandle() ) );

			// prefix scan for blocks data
			BlobGlobalMaxPoolingGlobalScanKernel<<<scanBlockCount, scanThreadCount, scanSharedSize>>>( desc, 
				histSize, GetRaw( globalSums.GetHandle() ), blockCount.y );

			// global sort
			BlobGlobalMaxPoolingGlobalShuffleKernel<<<blockCount, threadCount, 1>>>( desc, GetRaw( sourceData ),
			 	GetRaw( indicesSorted2.GetHandle() ), GetRaw( indicesSorted1.GetHandle() ), bin, histSize, poolSize, GetRaw( localSums.GetHandle() ), GetRaw( globalSums.GetHandle() ),
				GetRaw( resultData ), GetRaw( maxIndicesData ), maxCount, bin == 0, bin >= bitCount - bitsPerBin );
		}
	}
}

void CCudaMathEngine::BlobGlobalMaxPoolingBackward( const CGlobalMaxPoolingDesc& poolingDesc,
	const CFloatHandle& outputDiffData, const CIntHandle& maxIndicesData, const CFloatHandle& inputDiffData )
{
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData.GetMathEngine() == this );
	ASSERT_EXPR( inputDiffData.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

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

	BlobGlobalMaxPoolingBackwardKernel<<<blockCount, threadCount>>>( desc, GetRaw( outputDiffData ),
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
	SetCudaDevice( device->DeviceNumber );

	const CCuda3dMaxPoolingDescInternal& desc = static_cast<const CCuda3dMaxPoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& result = desc.Result;

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid3DMinZYX(1, 1, 32, blockCount, threadCount, result.ObjectCount(),
		result.Depth() * result.Height() * result.Width(), result.Channels());

	Blob3dMaxPoolingKernel<<<blockCount, threadCount>>>( desc, GetRaw( sourceData ),
		maxIndicesData == 0 ? 0 : GetRaw( *maxIndicesData ), GetRaw( resultData ) );
}

void CCudaMathEngine::Blob3dMaxPoolingBackward( const C3dMaxPoolingDesc& poolingDesc, const CFloatHandle& outputDiffData,
	const CIntHandle& maxIndicesData, const CFloatHandle& inputDiffData )
{
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData.GetMathEngine() == this );
	ASSERT_EXPR( inputDiffData.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const CCuda3dMaxPoolingDescInternal& desc = static_cast<const CCuda3dMaxPoolingDesc&>( poolingDesc ).Internal;
	VectorFill( inputDiffData, 0, desc.Source.BlobSize() );

	bool isAtomic = desc.FilterHeight > desc.StrideHeight || desc.FilterWidth > desc.StrideWidth || desc.FilterDepth > desc.StrideDepth;

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid3DMinZYX(1, 1, 32, blockCount, threadCount, desc.Result.ObjectCount(),
		desc.Result.Depth() * desc.Result.Height() * desc.Result.Width(), desc.Result.Channels());

	Blob3dMaxPoolingBackwardKernel<<<blockCount, threadCount>>>( desc, GetRaw( outputDiffData ),
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
	SetCudaDevice( device->DeviceNumber );

	const CCuda3dMeanPoolingDescInternal& desc = static_cast<const CCuda3dMeanPoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& result = desc.Result;

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid3DMinZYX(1, 1, 32, blockCount, threadCount, result.ObjectCount(),
		result.Depth() * result.Height() * result.Width(), result.Channels());

	Blob3dMeanPoolingKernel<<<blockCount, threadCount>>>( desc, GetRaw( sourceData ), GetRaw( resultData ) );
}

void CCudaMathEngine::Blob3dMeanPoolingBackward( const C3dMeanPoolingDesc& poolingDesc,
	const CFloatHandle& outputDiffData, const CFloatHandle& inputDiffData )
{
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( inputDiffData.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

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

	Blob3dMeanPoolingBackwardKernel<<<blockCount, threadCount>>>( desc, GetRaw( outputDiffData ),
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
	SetCudaDevice( device->DeviceNumber );

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
		BlobMaxOverTimePoolingKernel<<<blockCount, threadCount, sharedSize * sizeof(CValueWithIndex)>>>( desc,
			GetRaw( sourceData ), GetRaw( *maxIndicesData ), GetRaw( resultData ) );
	} else {
		BlobMaxOverTimePoolingKernel<<<blockCount, threadCount, sharedSize * sizeof(float)>>>( desc,
			GetRaw( sourceData ), GetRaw( resultData ) );
	}
}

void CCudaMathEngine::BlobMaxOverTimePoolingBackward( const CMaxOverTimePoolingDesc& poolingDesc, const CFloatHandle& outputDiffData,
	const CIntHandle& maxIndicesData, const CFloatHandle& inputDiffData )
{
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData.GetMathEngine() == this );
	ASSERT_EXPR( inputDiffData.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

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
		BlobMaxOverTimePoolingBackwardKernel<<<blockCount, threadCount>>>(store, desc, GetRaw( outputDiffData ),
			GetRaw( maxIndicesData ), GetRaw( inputDiffData ) );
	} else {
		CStoreAtomicAdd store;
		BlobMaxOverTimePoolingBackwardKernel<<<blockCount, threadCount>>>(store, desc, GetRaw( outputDiffData ),
			GetRaw( maxIndicesData ), GetRaw( inputDiffData ));
	}
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
