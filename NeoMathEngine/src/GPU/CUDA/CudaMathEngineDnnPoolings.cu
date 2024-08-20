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

void CCudaMathEngine::BlobMaxPooling( const CMaxPoolingDesc& poolingDesc, const CConstFloatHandle& sourceData,
	const CIntHandle* maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData == 0 || maxIndicesData->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const CCudaMaxPoolingDescInternal& desc = static_cast<const CCudaMaxPoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& result = desc.Result;

	int* maxIndexPtr = 0;
	if( maxIndicesData != 0 ) {
		maxIndexPtr = GetRaw( *maxIndicesData );
	}

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid3DMinZYX( 1, 1, 32, blockCount, threadCount,
		result.ObjectCount(), result.Height() * result.Width(), result.Depth() * result.Channels() );

	BlobMaxPoolingKernel<<<blockCount, threadCount>>>( desc, GetRaw( sourceData ), maxIndexPtr, GetRaw( resultData ) );
}

void CCudaMathEngine::BlobMaxPoolingBackward( const CMaxPoolingDesc& poolingDesc, const CConstFloatHandle& resultDiff,
	const CConstIntHandle& maxIndices, const CFloatHandle& sourceDiff )
{
	ASSERT_EXPR( resultDiff.GetMathEngine() == this );
	ASSERT_EXPR( maxIndices.GetMathEngine() == this );
	ASSERT_EXPR( sourceDiff.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const CCudaMaxPoolingDescInternal& desc = static_cast<const CCudaMaxPoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& source = desc.Source;
	const CCudaBlobDesc& result = desc.Result;

	VectorFill( sourceDiff, 0, source.BlobSize() );

	const bool isAtomic = desc.FilterHeight > desc.StrideHeight || desc.FilterWidth > desc.StrideWidth;
	const int batchNorm = ( result.ObjectCount() + BlobMaxPoolingBackwardCombine - 1 ) / BlobMaxPoolingBackwardCombine;
	const int totalChannels = result.Depth() * result.Channels();

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid3DMinZYX( 1, 1, 32, blockCount, threadCount, batchNorm, result.Height() * result.Width(), totalChannels );

	BlobMaxPoolingBackwardKernel<<<blockCount, threadCount>>>( desc, isAtomic,
		GetRaw( resultDiff ), GetRaw( maxIndices ), GetRaw( sourceDiff ), batchNorm );
}

//-----------------------------------------------------------------------------------------------------------------------

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
	const CConstFloatHandle& sourceData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const CCudaMeanPoolingDescInternal& desc = static_cast<const CCudaMeanPoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& result = desc.Result;

	dim3 blockCount;
	dim3 threadCount;

	const int totalChannels = result.Depth() * result.Channels();

	getCudaTaskGrid3DMinZYX( 1, 1, 32, blockCount, threadCount,
		result.ObjectCount(), result.Height() * result.Width(), totalChannels );
	BlobMeanPoolingKernel<<<blockCount, threadCount>>>( desc, GetRaw( sourceData ), GetRaw( resultData ) );
}

void CCudaMathEngine::BlobMeanPoolingBackward( const CMeanPoolingDesc& poolingDesc, const CConstFloatHandle& resultDiff,
	const CFloatHandle& sourceDiff )
{
	ASSERT_EXPR( resultDiff.GetMathEngine() == this );
	ASSERT_EXPR( sourceDiff.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const CCudaMeanPoolingDescInternal& desc = static_cast<const CCudaMeanPoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& source = desc.Source;
	const CCudaBlobDesc& result = desc.Result;

	const bool isAtomic = desc.FilterHeight > desc.StrideHeight || desc.FilterWidth > desc.StrideWidth;

	VectorFill( sourceDiff, 0, source.BlobSize() );

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid3D( blockCount, threadCount, result.ObjectCount(), result.Height() * result.Width(), result.Depth() * result.Channels() );

	BlobMeanPoolingBackwardKernel<<<blockCount, threadCount>>>( desc, GetRaw( resultDiff ), GetRaw( sourceDiff ), isAtomic );
}

//-----------------------------------------------------------------------------------------------------------------------

CGlobalMaxOverTimePoolingDesc* CCudaMathEngine::InitGlobalMaxOverTimePooling( const CBlobDesc& source, const CBlobDesc& result )
{
	CCudaGlobalMaxOverTimePoolingDesc* desc = new CCudaGlobalMaxOverTimePoolingDesc();
	desc->Internal.Source = source;
	desc->Internal.Result = result;
	return desc;
}

void CCudaMathEngine::BlobGlobalMaxOverTimePooling( const CGlobalMaxOverTimePoolingDesc& poolingDesc,
	const CConstFloatHandle& sourceData, const CIntHandle* maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData == 0 || maxIndicesData->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const CCudaGlobalMaxOverTimePoolingDescInternal& desc = static_cast<const CCudaGlobalMaxOverTimePoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& source = desc.Source;

	const int objectCount = source.BatchLength();
	const int objectSize = source.BlobSize() / objectCount;

	int blockCount = 0;
	int threadCount = 0;
	getCudaTaskGrid( blockCount, threadCount, objectSize );

	if( maxIndicesData == 0 ) {
		BlobGlobalMaxOverTimePoolingKernel<<<blockCount, threadCount>>>( desc, GetRaw( sourceData ), GetRaw( resultData ) );
	} else {
		BlobGlobalMaxOverTimePoolingWithIndexKernel<<<blockCount, threadCount>>>( desc, GetRaw( sourceData ), GetRaw( *maxIndicesData ), GetRaw( resultData ) );
	}
}

void CCudaMathEngine::BlobGlobalMaxOverTimePoolingBackward( const CGlobalMaxOverTimePoolingDesc& poolingDesc,
	const CConstFloatHandle& resultDiff, const CConstIntHandle& maxIndices, const CFloatHandle& sourceDiff )
{
	ASSERT_EXPR( sourceDiff.GetMathEngine() == this );
	ASSERT_EXPR( maxIndices.GetMathEngine() == this );
	ASSERT_EXPR( resultDiff.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const CCudaGlobalMaxOverTimePoolingDescInternal& desc = static_cast<const CCudaGlobalMaxOverTimePoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& source = desc.Source;
	const CCudaBlobDesc& result = desc.Result;

	VectorFill( sourceDiff, 0, source.BlobSize() );

	int blockCount = 0;
	int threadCount = 0;
	getCudaTaskGrid( blockCount, threadCount, result.BlobSize() );

	BlobGlobalMaxOverTimePoolingBackwardKernel<<<blockCount, threadCount>>>( desc, GetRaw( resultDiff ), GetRaw( maxIndices ), GetRaw( sourceDiff ) );
}

//-----------------------------------------------------------------------------------------------------------------------

CGlobalMaxPoolingDesc* CCudaMathEngine::InitGlobalMaxPooling( const CBlobDesc& source, const CBlobDesc& maxIndices, const CBlobDesc& result )
{
	CCudaGlobalMaxPoolingDesc* desc = new CCudaGlobalMaxPoolingDesc();
	desc->Internal.Source = source;
	desc->Internal.MaxIndices = maxIndices;
	desc->Internal.Result = result;
	return desc;
}

void CCudaMathEngine::BlobGlobalMaxPooling( const CGlobalMaxPoolingDesc& poolingDesc, const CConstFloatHandle& sourceData,
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

	ASSERT_EXPR( source.ObjectCount() == result.ObjectCount() && maxIndices.ObjectCount() == result.ObjectCount() );
	ASSERT_EXPR( maxIndices.ObjectSize() == result.ObjectSize() );

	const int poolSize = source.Depth() * source.Height() * source.Width();
	const int maxCount = result.Depth() * result.Height() * result.Width();

	const int heapSharedMemoryPerThread = 2 * maxCount * sizeof( float );
	const int heapMaxThreadCount = device->SharedMemoryLimit / heapSharedMemoryPerThread;
	const int deviceMaxMemoryLimit = 4 * source.BlobSize() * sizeof( float );
	if( heapMaxThreadCount > 32 || device->MemoryLimit < deviceMaxMemoryLimit ) {
		dim3 blockCount;
		dim3 threadCount;
		getCudaTaskGrid2DMinYX( 1, device->ThreadMax3DCountX, blockCount, threadCount,
			source.ObjectCount() * source.Channels(), ( poolSize + maxCount - 1 ) / maxCount + 1, heapMaxThreadCount );
		blockCount.x = 1;
		threadCount.x--;
		ASSERT_EXPR( threadCount.x > 0 );

		const int sharedSize = threadCount.y * ( threadCount.x + 1 ) * heapSharedMemoryPerThread;
		BlobGlobalMaxPoolingHeapKernel<<<blockCount, threadCount, sharedSize>>>( desc, GetRaw( sourceData ),
			GetRaw( maxIndicesData ), GetRaw( resultData ), poolSize, maxCount );
	} else {
		constexpr int bitsPerBin = 8;
		constexpr int histSize = ( 1 << bitsPerBin );
		constexpr int memoryPerThread = histSize * sizeof( int );
		const int maxThreadCount = device->SharedMemoryLimit / memoryPerThread;

		int height = 1;
		const int poolSizeNorm = max( 1, poolSize / ( 2 * histSize ) );
		while( height < poolSizeNorm ) {
			height *= 2;
		}

		dim3 blockCount;
		dim3 threadCount;
		getCudaTaskGrid2DMinYX( device->ThreadMax3DCountY, 1, blockCount, threadCount,
			height, source.ObjectCount() * source.Channels(), maxThreadCount );

		dim3 scanBlockCount;
		dim3 scanThreadCount;
		getCudaTaskGrid2DMinYX( device->ThreadMax3DCountY, 1, scanBlockCount, scanThreadCount,
			blockCount.y, source.ObjectCount() * source.Channels(), maxThreadCount );
		scanBlockCount.y = 1;

		CIntHandleStackVar indicesSorted1( mathEngine(), source.BlobSize() );
		CIntHandleStackVar indicesSorted2( mathEngine(), source.BlobSize() );
		CIntHandleStackVar localSums( mathEngine(), blockCount.x * threadCount.x * blockCount.y * histSize );
		CIntHandleStackVar globalSums( mathEngine(), blockCount.x * threadCount.x * ( blockCount.y + 1 ) * histSize );
		const int localSortSharedSize = threadCount.y * threadCount.x * memoryPerThread;
		const int scanSharedSize = scanThreadCount.x * scanThreadCount.y * memoryPerThread;

		constexpr int bitCount = sizeof( float ) * 8;
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
	const CConstFloatHandle& resultDiff, const CConstIntHandle& maxIndices, const CFloatHandle& sourceDiff )
{
	ASSERT_EXPR( resultDiff.GetMathEngine() == this );
	ASSERT_EXPR( maxIndices.GetMathEngine() == this );
	ASSERT_EXPR( sourceDiff.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const CCudaGlobalMaxPoolingDescInternal& desc = static_cast<const CCudaGlobalMaxPoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& source = desc.Source;
	const CCudaBlobDesc& result = desc.Result;

	VectorFill( sourceDiff, 0, source.BlobSize() );

	const int poolSize = source.Depth() * source.Height() * source.Width();
	const int maxCount = result.Depth() * result.Height() * result.Width();
	const int fullSize = result.ObjectCount() * maxCount * result.Channels();

	int blockCount = 0;
	int threadCount = 0;
	getCudaTaskGrid( blockCount, threadCount, fullSize, BlobGlobalMaxPoolingBackwardCombine );

	BlobGlobalMaxPoolingBackwardKernel<<<blockCount, threadCount>>>( desc, GetRaw( resultDiff ),
		GetRaw( maxIndices ), GetRaw( sourceDiff ), poolSize, maxCount, fullSize );
}

//-----------------------------------------------------------------------------------------------------------------------

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

void CCudaMathEngine::Blob3dMaxPooling( const C3dMaxPoolingDesc& poolingDesc, const CConstFloatHandle& sourceData,
	const CIntHandle* maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData == 0 || maxIndicesData->IsNull() || maxIndicesData->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const CCuda3dMaxPoolingDescInternal& desc = static_cast<const CCuda3dMaxPoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& result = desc.Result;

	dim3 blockCount = 0;
	dim3 threadCount = 0;
	getCudaTaskGrid3DMinZYX( 1, 1, 32, blockCount, threadCount, result.ObjectCount(),
		result.Depth() * result.Height() * result.Width(), result.Channels() );

	Blob3dMaxPoolingKernel<<<blockCount, threadCount>>>( desc, GetRaw( sourceData ),
		maxIndicesData == 0 ? 0 : GetRaw( *maxIndicesData ), GetRaw( resultData ) );
}

void CCudaMathEngine::Blob3dMaxPoolingBackward( const C3dMaxPoolingDesc& poolingDesc, const CConstFloatHandle& resultDiff,
	const CConstIntHandle& maxIndices, const CFloatHandle& sourceDiff )
{
	ASSERT_EXPR( resultDiff.GetMathEngine() == this );
	ASSERT_EXPR( maxIndices.GetMathEngine() == this );
	ASSERT_EXPR( sourceDiff.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const CCuda3dMaxPoolingDescInternal& desc = static_cast<const CCuda3dMaxPoolingDesc&>( poolingDesc ).Internal;
	VectorFill( sourceDiff, 0, desc.Source.BlobSize() );

	const bool isAtomic = desc.FilterHeight > desc.StrideHeight || desc.FilterWidth > desc.StrideWidth || desc.FilterDepth > desc.StrideDepth;

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid3DMinZYX( 1, 1, 32, blockCount, threadCount, desc.Result.ObjectCount(),
		desc.Result.Depth() * desc.Result.Height() * desc.Result.Width(), desc.Result.Channels() );

	Blob3dMaxPoolingBackwardKernel<<<blockCount, threadCount>>>( desc, GetRaw( resultDiff ),
		GetRaw( maxIndices ), GetRaw( sourceDiff ), isAtomic );
}

//-----------------------------------------------------------------------------------------------------------------------

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

void CCudaMathEngine::Blob3dMeanPooling( const C3dMeanPoolingDesc& poolingDesc, const CConstFloatHandle& sourceData,
	const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const CCuda3dMeanPoolingDescInternal& desc = static_cast<const CCuda3dMeanPoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& result = desc.Result;

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid3DMinZYX( 1, 1, 32, blockCount, threadCount, result.ObjectCount(),
		result.Depth() * result.Height() * result.Width(), result.Channels() );

	Blob3dMeanPoolingKernel<<<blockCount, threadCount>>>( desc, GetRaw( sourceData ), GetRaw( resultData ) );
}

void CCudaMathEngine::Blob3dMeanPoolingBackward( const C3dMeanPoolingDesc& poolingDesc,
	const CConstFloatHandle& resultDiff, const CFloatHandle& sourceDiff )
{
	ASSERT_EXPR( resultDiff.GetMathEngine() == this );
	ASSERT_EXPR( sourceDiff.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const CCuda3dMeanPoolingDescInternal& desc = static_cast<const CCuda3dMeanPoolingDesc&>( poolingDesc ).Internal;

	if( desc.FilterHeight != desc.StrideHeight || desc.FilterWidth != desc.StrideWidth || desc.FilterDepth != desc.StrideDepth ) {
		// Either the cube blocks used for pooling have nonzero intersections, and we need to add up several diffs,
		// or some of the data is skipped when pooling and we need to set diff = 0 for it
		VectorFill( sourceDiff, 0, desc.Source.BlobSize() );
	}

	// Indicates that the cube blocks used for pooling have nonzero intersections, and the diffs should be added up (atomicAdd)
	const bool isAtomic = desc.FilterHeight > desc.StrideHeight || desc.FilterWidth > desc.StrideWidth || desc.FilterDepth > desc.StrideDepth;
	const CCudaBlobDesc& result = desc.Result;

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid3DMinZYX( 1, 1, 32, blockCount, threadCount, result.ObjectCount(),
		result.Depth() * result.Height() * result.Width(), result.Channels() );

	Blob3dMeanPoolingBackwardKernel<<<blockCount, threadCount>>>( desc, GetRaw( resultDiff ), GetRaw( sourceDiff ), isAtomic );
}

//-----------------------------------------------------------------------------------------------------------------------

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

void CCudaMathEngine::BlobMaxOverTimePooling( const CMaxOverTimePoolingDesc& poolingDesc, const CConstFloatHandle& sourceData,
	const CIntHandle* maxIndices, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndices == 0 || maxIndices->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const CCudaMaxOverTimePoolingDescInternal& desc = static_cast<const CCudaMaxOverTimePoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& result = desc.Result;

	int xSize = ( desc.FilterLen + BlobMaxOverTimePoolingCombine - 1 ) / BlobMaxOverTimePoolingCombine;
	xSize = alignXSizeForWarp( xSize );

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2DMinYX( 1, device->ThreadMaxCount, blockCount, threadCount, result.BlobSize(), xSize );
	blockCount.x = 1; // in any case there may only one block along the X coordinate so that we can calculate the maximum inside one block

	const int sharedSize = threadCount.x * threadCount.y * threadCount.z;
	if( maxIndices != 0 ) {
		BlobMaxOverTimePoolingKernel<<<blockCount, threadCount, sharedSize * sizeof( CValueWithIndex )>>>( desc,
			GetRaw( sourceData ), GetRaw( *maxIndices ), GetRaw( resultData ) );
	} else {
		BlobMaxOverTimePoolingKernel<<<blockCount, threadCount, sharedSize * sizeof( float )>>>( desc,
			GetRaw( sourceData ), GetRaw( resultData ) );
	}
}

void CCudaMathEngine::BlobMaxOverTimePoolingBackward( const CMaxOverTimePoolingDesc& poolingDesc, const CConstFloatHandle& resultDiff,
	const CConstIntHandle& maxIndices, const CFloatHandle& sourceDiff )
{
	ASSERT_EXPR( resultDiff.GetMathEngine() == this );
	ASSERT_EXPR( maxIndices.GetMathEngine() == this );
	ASSERT_EXPR( sourceDiff.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const CCudaMaxOverTimePoolingDescInternal& desc = static_cast<const CCudaMaxOverTimePoolingDesc&>( poolingDesc ).Internal;
	const CCudaBlobDesc& source = desc.Source;
	const CCudaBlobDesc& result = desc.Result;

	// Set diff to 0
	VectorFill( sourceDiff, 0, source.BlobSize() );

	int blockCount = 0;
	int threadCount = 0;
	getCudaTaskGrid( blockCount, threadCount, result.BlobSize(), BlobMaxOverTimePoolingBackwardCombine );

	if( desc.StrideLen >= desc.FilterLen ) {
		// The pooling areas do not intersect, no need to add
		CStoreSet store;
		BlobMaxOverTimePoolingBackwardKernel<<<blockCount, threadCount>>>( store, desc, GetRaw( resultDiff ),
			GetRaw( maxIndices ), GetRaw( sourceDiff ) );
	} else {
		CStoreAtomicAdd store;
		BlobMaxOverTimePoolingBackwardKernel<<<blockCount, threadCount>>>( store, desc, GetRaw( resultDiff ),
			GetRaw( maxIndices ), GetRaw( sourceDiff ) );
	}
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
