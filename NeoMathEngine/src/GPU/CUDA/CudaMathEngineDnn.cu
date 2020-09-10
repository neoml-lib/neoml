﻿/* Copyright © 2017-2020 ABBYY Production LLC

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
#include <CudaCommon.h>
#include <CudaDevice.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>

#include <Kernels/CudaDnnKernels.h>

namespace NeoML {

template<class T>
void CCudaMathEngine::blobMergeByDimCuda( int dimNum, const CBlobDesc* from, const CTypedMemoryHandle<T>* fromData, int fromCount, const CBlobDesc& to, const CTypedMemoryHandle<T>& toData )
{
	ASSERT_EXPR(fromCount <= MaxBlobDescs);
	ASSERT_EXPR(0 <= dimNum && dimNum < CBlobDesc::MaxDimensions);
	SetCudaDevice( device->DeviceNumber );

	int s[CBlobDesc::MaxDimensions];
	CCudaBlobDescArray<T> fromArr;
	fromArr.Count = fromCount;
	for(int i = 0; i < fromCount; ++i) {
		fromArr.Descs[i] = from[i];
		fromArr.Data[i] = GetRaw(fromData[i]);
		from[i].GetDimSizes(s);
 		fromArr.Widths[i] = 1;
		for(int z = dimNum; z < CBlobDesc::MaxDimensions; z++) {
			fromArr.Widths[i] *= s[z];
		}
	}
	to.GetDimSizes(s);
	int height = 1;
	for(int z  = 0; z < dimNum; z++) {
		height *= s[z];
	}
 	int width = to.BlobSize() / height;
	int heightNorm = (height + BlobMergeByDimCombine - 1) / BlobMergeByDimCombine;
	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2D(blockCount, threadCount, heightNorm, width);

	BlobMergeByDimKernel<<<blockCount, threadCount>>>(height, width, fromArr, to, GetRaw(toData), heightNorm);
}

template<class T>
void CCudaMathEngine::blobMergeByDim0(const CBlobDesc* from, const CTypedMemoryHandle<T>* fromData, int fromCount, const CTypedMemoryHandle<T>& toData)
{
	CTypedMemoryHandle<T> output = toData;
	for(int i = 0; i < fromCount; ++i) {
		int blobSize = from[i].BlobSize();
		VectorCopy(output, fromData[i], blobSize);
		output += blobSize;
	}
}

template<class T>
void CCudaMathEngine::blobMergeByDim( int dim, const CBlobDesc* from, const CTypedMemoryHandle<T>* fromData, int fromCount, const CBlobDesc& to, const CTypedMemoryHandle<T>& toData )
{
	if(dim == 0) {
		return blobMergeByDim0(from, fromData, fromCount, toData);
	} else {
		return blobMergeByDimCuda(dim, from, fromData, fromCount, to, toData);
	}
}

template<class T>
void CCudaMathEngine::blobSplitByDimCuda(int dimNum, const CBlobDesc& from, const CTypedMemoryHandle<T>& fromData, const CBlobDesc* to, const CTypedMemoryHandle<T>* toData, int toCount)
{
	ASSERT_EXPR(toCount <= MaxBlobDescs);
	ASSERT_EXPR(0 <= dimNum && dimNum < CBlobDesc::MaxDimensions);
	SetCudaDevice( device->DeviceNumber );

	CCudaBlobDescArray<T> toArr;
	toArr.Count = toCount;
	int s[CBlobDesc::MaxDimensions];
	for(int i = 0; i < toCount; ++i) {
		toArr.Descs[i] = to[i];
		toArr.Data[i] = GetRaw(toData[i]);

		to[i].GetDimSizes(s);
 		toArr.Widths[i] = 1;
		for(int z = dimNum; z < CBlobDesc::MaxDimensions; z++) {
			toArr.Widths[i] *= s[z];
		}
	}
	from.GetDimSizes(s);
	int height = 1;
	for(int z  = 0; z < dimNum; z++) {
		height *= s[z];
	}
 	int width = from.BlobSize() / height;
	int heightNorm = (height + BlobSplitByDimCombine - 1) / BlobSplitByDimCombine;

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2D(blockCount, threadCount, heightNorm, width);

	BlobSplitByDimKernel<<<blockCount, threadCount>>>(height, width, from, GetRaw(fromData), toArr, heightNorm);
}

template<class T>
void CCudaMathEngine::blobSplitByDim0(const CBlobDesc& from, const CTypedMemoryHandle<T>& fromData, const CBlobDesc* to, const CTypedMemoryHandle<T>* toData, int toCount)
{
	CTypedMemoryHandle<const T> input = fromData;
	for( int i = 0; i < toCount; ++i ) {
		int blobSize = to[i].BlobSize();
		VectorCopy( toData[i], input, blobSize );
		input += blobSize;
	}
}

template<class T>
void CCudaMathEngine::blobSplitByDim( int dim, const CBlobDesc& from, const CTypedMemoryHandle<T>& fromData, const CBlobDesc* to, const CTypedMemoryHandle<T>* toData, int toCount )
{
	if(dim == 0) {
		return blobSplitByDim0(from, fromData, to, toData, toCount);
	} else {
		return blobSplitByDimCuda(dim, from, fromData, to, toData, toCount);
	}
}

void CCudaMathEngine::BlobMergeByDim(TBlobDim dim, const CBlobDesc* from, const CFloatHandle* fromData, int fromCount, const CBlobDesc& to, const CFloatHandle& toData)
{
	ASSERT_EXPR(dim < BD_Count && fromCount <= MaxBlobDescs);
	blobMergeByDim(dim, from, fromData, fromCount, to, toData);
}

void CCudaMathEngine::BlobMergeByDim(TBlobDim dim, const CBlobDesc* from, const CIntHandle* fromData, int fromCount, const CBlobDesc& to, const CIntHandle& toData)
{
	ASSERT_EXPR(dim < BD_Count && fromCount <= MaxBlobDescs);
	blobMergeByDim(dim, from, fromData, fromCount, to, toData);
}

void CCudaMathEngine::BlobSplitByDim(TBlobDim dim, const CBlobDesc& from, const CFloatHandle& fromData, const CBlobDesc* to, const CFloatHandle* toData, int toCount)
{
	ASSERT_EXPR(dim < BD_Count && toCount <= MaxBlobDescs);
	blobSplitByDim(dim, from, fromData, to, toData, toCount);
}

void CCudaMathEngine::BlobSplitByDim(TBlobDim dim, const CBlobDesc& from, const CIntHandle& fromData, const CBlobDesc* to, const CIntHandle* toData, int toCount)
{
	ASSERT_EXPR(dim < BD_Count && toCount <= MaxBlobDescs);
	blobSplitByDim(dim, from, fromData, to, toData, toCount);
}

void CCudaMathEngine::BlobResizeImage( const CBlobDesc& from, const CFloatHandle& fromData, int deltaLeft, int deltaRight,
	int deltaTop, int deltaBottom, float defaultValue, const CBlobDesc& to, const CFloatHandle& toData )
{
	ASSERT_EXPR( fromData.GetMathEngine() == this );
	ASSERT_EXPR( toData.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid3D( blockCount, threadCount, to.ObjectCount(), to.Height() * to.Width(), to.Channels() * to.Depth() );
	BlobResizeImageKernel<<<blockCount, threadCount>>>( from, GetRaw(fromData), deltaLeft, deltaTop, defaultValue, to,
		GetRaw(toData) );
}

void CCudaMathEngine::BlobGetSubSequence( const CBlobDesc& from, const CFloatHandle& fromData, const CIntHandle& indexHandle, const CBlobDesc& to,
	const CFloatHandle& toData, int startPos, bool isRev )
{
	ASSERT_EXPR( fromData.GetMathEngine() == this );
	ASSERT_EXPR( indexHandle.IsNull() || indexHandle.GetMathEngine() == this );
	ASSERT_EXPR( toData.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int objectSize = from.ObjectSize() * from.ListSize();
	int objectSizeNorm = (objectSize + BlobGetSubSequenceCombine - 1) / BlobGetSubSequenceCombine;

	dim3 blockCount;
	dim3 threadCount;

	getCudaTaskGrid3D(blockCount, threadCount, to.BatchLength(), to.BatchWidth(), objectSizeNorm);

	BlobGetSubSequenceKernel<<<blockCount, threadCount>>>(from, GetRaw(fromData), GetRaw(indexHandle),
		to, GetRaw( toData ), startPos, isRev, objectSizeNorm);
}

void CCudaMathEngine::Upsampling2DForward( const CBlobDesc& input, const CFloatHandle& inputData, int heightCopyCount,
	int widthCopyCount, const CBlobDesc& result, const CFloatHandle& resultData )
{
	ASSERT_EXPR( inputData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	ASSERT_EXPR(heightCopyCount > 0);
	ASSERT_EXPR(widthCopyCount > 0);
	ASSERT_EXPR(input.BatchLength() == result.BatchLength());
	ASSERT_EXPR(input.BatchWidth() == result.BatchWidth());
	ASSERT_EXPR(input.Channels() == result.Channels());
	ASSERT_EXPR(input.Depth() == result.Depth());
	ASSERT_EXPR(input.Height() * heightCopyCount == result.Height());
	ASSERT_EXPR(input.Width() * widthCopyCount == result.Width());
	SetCudaDevice( device->DeviceNumber );

	// This is how the algorithm works
	// The input blob can be considered as batchSize matrices of inputHeight x inputRowSize size each
	// The output blob can be considered as batchSize matrices of resultHeight x resultRowSize size each
	// To calculate the (i,j) element of the output matrix create a separate thread
	const int inputHeight = input.Height();
	const int inputRowSize = input.Width() * input.Depth() * input.Channels();
	const int pixelSize = input.Depth() * input.Channels();
	const int resultHeight = result.Height();
	const int resultRowSize = result.Width() * result.Depth() * result.Channels();
	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2D( blockCount, threadCount, resultHeight, resultRowSize );
	Upsampling2DForwardKernel<<<blockCount, threadCount>>>(
		heightCopyCount, widthCopyCount, pixelSize,
		input.ObjectCount(), inputHeight, inputRowSize, GetRaw( inputData ),
		resultHeight, resultRowSize, GetRaw( resultData ) );
}

void CCudaMathEngine::Upsampling2DBackward( const CBlobDesc& input, const CFloatHandle& inputData, int heightCopyCount,
	int widthCopyCount, const CBlobDesc& result, const CFloatHandle& resultData )
{
	ASSERT_EXPR( inputData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	ASSERT_EXPR(heightCopyCount > 0);
	ASSERT_EXPR(widthCopyCount > 0);
	ASSERT_EXPR(input.BatchLength() == result.BatchLength());
	ASSERT_EXPR(input.BatchWidth() == result.BatchWidth());
	ASSERT_EXPR(input.Channels() == result.Channels());
	ASSERT_EXPR(input.Depth() == result.Depth());
	ASSERT_EXPR(result.Height() * heightCopyCount == input.Height());
	ASSERT_EXPR(result.Width() * widthCopyCount == input.Width());
	SetCudaDevice( device->DeviceNumber );

	// Fill the resulting blob with zeros
	VectorFill( resultData, 0, result.BlobSize() );

	// This is how the algorithm works
	// The input blob can be considered as batchSize matrices of inputHeight x inputRowSize size each
	// The output blob can be considered as batchSize matrices of resultHeight x resultRowSize size each
	// Create a lattice of threads over the input matrix
	// Each thread processes the (i,j) element of the input matrix
	const int inputHeight = input.Height();
	const int inputRowSize = input.Width() * input.Depth() * input.Channels();
	const int pixelSize = input.Depth() * input.Channels();
	const int resultHeight = result.Height();
	const int resultRowSize = result.Width() * result.Depth() * result.Channels();
	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2D( blockCount, threadCount, inputHeight, inputRowSize );
	Upsampling2DBackwardKernel<<<blockCount, threadCount>>>(
		heightCopyCount, widthCopyCount, pixelSize,
		input.ObjectCount(), inputHeight, inputRowSize, GetRaw( inputData ),
		resultHeight, resultRowSize, GetRaw( resultData ) );
}

void CCudaMathEngine::BuildIntegerHist( const CConstIntHandle& numbersHandle, int numbersCount,
	const CIntHandle& resultHandle, int maxNumber )
{
	ASSERT_EXPR( numbersHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	VectorFill( resultHandle, 0, maxNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, numbersCount );

	BuildIntegerHistKernel<<<blockCount, threadCount>>>( GetRaw( numbersHandle ),
		numbersCount, GetRaw( resultHandle ) );
}


void CCudaMathEngine::MatrixRowsToVectorSquaredL2Distance( const CConstFloatHandle& matrixHandle,
	const int matrixHeight, const int matrixWidth, const CConstFloatHandle& vectorHandle,
	const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	VectorFill( resultHandle, 0.f, matrixHeight );

	const int normalizedWidth = ( matrixWidth + MatrixRowsToVectorSquaredL2DistanceCombineCount - 1 )
		/ MatrixRowsToVectorSquaredL2DistanceCombineCount;

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2D( blockCount, threadCount, matrixHeight, normalizedWidth );

	MatrixRowsToVectorSquaredL2DistanceKernel<<<blockCount, threadCount>>>( GetRaw( matrixHandle ),
		matrixHeight, matrixWidth, GetRaw( vectorHandle ), GetRaw( resultHandle ), normalizedWidth );
}

void CCudaMathEngine::Reorg( const CBlobDesc& source, const CFloatHandle& sourceData, int stride, bool isForward,
	const CBlobDesc& result, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount = 0;
	int threadCount = 0;
	getCudaTaskGrid( blockCount, threadCount, source.BlobSize() );
	if( !isForward ) {
		ReorgKernel<<<blockCount, threadCount>>>( GetRaw( sourceData ), result.Width(), result.Height(),
			result.Channels(), result.ObjectCount(), stride, isForward, GetRaw( resultData ) );
	} else { 
		ReorgKernel<<<blockCount, threadCount>>>( GetRaw( sourceData ), source.Width(), source.Height(),
			source.Channels(), source.ObjectCount(), stride, isForward, GetRaw( resultData ) );
	}
}

void CCudaMathEngine::Reorg( const CBlobDesc& source, const CIntHandle& sourceData, int stride, bool isForward,
	const CBlobDesc& result, const CIntHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount = 0;
	int threadCount = 0;
	getCudaTaskGrid( blockCount, threadCount, source.BlobSize() );
	if( !isForward ) {
		ReorgKernel<<<blockCount, threadCount>>>( GetRaw( sourceData ), result.Width(), result.Height(),
			result.Channels(), result.ObjectCount(), stride, isForward, GetRaw( resultData ) );
	} else { 
		ReorgKernel<<<blockCount, threadCount>>>( GetRaw( sourceData ), source.Width(), source.Height(),
			source.Channels(), source.ObjectCount(), stride, isForward, GetRaw( resultData ) );
	}
}

void CCudaMathEngine::AddWidthIndex( const CBlobDesc& source, const CFloatHandle& sourceData, bool isForward, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount = 0;
	int threadCount = 0;
	getCudaTaskGrid( blockCount, threadCount, source.BlobSize() );

	AddWidthIndexKernel<<<blockCount, threadCount>>>(
		GetRaw( sourceData ), source.Width(), source.Height(),
		source.Channels(), source.ObjectCount(), isForward, GetRaw( resultData ) );
}

void CCudaMathEngine::AddWidthIndex( const CBlobDesc& source, const CIntHandle& sourceData, bool isForward, const CIntHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount = 0;
	int threadCount = 0;
	getCudaTaskGrid( blockCount, threadCount, source.BlobSize() );

	AddWidthIndexKernel <<<blockCount, threadCount >>>(
		GetRaw( sourceData ), source.Width(), source.Height(),
		source.Channels(), source.ObjectCount(), isForward, GetRaw( resultData ) );
}

void CCudaMathEngine::AddHeightIndex( const CBlobDesc& source, const CFloatHandle& sourceData, bool isForward, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount = 0;
	int threadCount = 0;
	getCudaTaskGrid( blockCount, threadCount, source.BlobSize() );

	AddHeightIndexKernel<<<blockCount, threadCount>>>(
		GetRaw(sourceData), source.Width(), source.Height(),
		source.Channels(), source.ObjectCount(), isForward, GetRaw(resultData) );
}

void CCudaMathEngine::AddHeightIndex( const CBlobDesc& source, const CIntHandle& sourceData, bool isForward,
	const CIntHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount = 0;
	int threadCount = 0;
	getCudaTaskGrid( blockCount, threadCount, source.BlobSize() );
	
	AddHeightIndexKernel <<<blockCount, threadCount >>>(
		GetRaw(sourceData), source.Width(), source.Height(),
		source.Channels(), source.ObjectCount(), isForward, GetRaw(resultData) );
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
