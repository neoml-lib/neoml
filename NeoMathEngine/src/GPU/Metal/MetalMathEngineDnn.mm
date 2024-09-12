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

#ifdef NEOML_USE_METAL

#include <MetalMathEngine.h>
#include <MathEngineCommon.h>
#include <MetalKernel.h>

@import Foundation;
@import MetalKit;

namespace NeoML {

static const int FloatDescArrayMaxBlobs = 32;

struct CFloatDescArray final {
	int Count;
	CBlobDesc Descs[FloatDescArrayMaxBlobs];
	CFloatHandle Data[FloatDescArrayMaxBlobs];
	int Widths[FloatDescArrayMaxBlobs];
};

void CMetalMathEngine::blobMergeByDim( int dimNum, const CBlobDesc* from, const CFloatHandle* fromData,
	int fromCount, const CBlobDesc& to, const CFloatHandle& toData )
{
	ASSERT_EXPR( toData.GetMathEngine() == this );
	ASSERT_EXPR( fromCount <= MaxBlobDescs );
	ASSERT_EXPR( 0 <= dimNum && dimNum < CBlobDesc::MaxDimensions );

	int s[CBlobDesc::MaxDimensions];
	CFloatDescArray fromArr;
	fromArr.Count = fromCount;
	for(int i = 0; i < fromCount; ++i) {
		fromArr.Descs[i] = from[i];
		ASSERT_EXPR( fromData[i].GetMathEngine() == this );
		fromArr.Data[i] = fromData[i];
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

	const int heightNorm = ( height + 15 ) / 16;
	int wStart = 0;
	for( int i = 0; i < fromArr.Count; i++ ) {
		C2DKernel kernel( *queue, "matrixKernelBlobMergeByDim", 1, 1, heightNorm, fromArr.Widths[i] );
		kernel.SetParam( height, 0 );
		kernel.SetParam( width, 1 );
		kernel.SetParam( fromArr.Descs[i], 2 );
		kernel.SetParam( fromArr.Data[i], 3 );
		kernel.SetParam( to, 4 );
		kernel.SetParam( toData, 5 );
		kernel.SetParam( heightNorm, 6 );
		kernel.SetParam( wStart, 7 );
		kernel.SetParam( fromArr.Widths[i], 8 );
		ASSERT_EXPR( kernel.Run() );

		wStart += fromArr.Widths[i];
	}
}

void CMetalMathEngine::blobSplitByDim( int dimNum, const CBlobDesc& from, const CConstFloatHandle& fromData,
	const CBlobDesc* to, const CFloatHandle* toData, int toCount )
{
	ASSERT_EXPR( fromData.GetMathEngine() == this );
	ASSERT_EXPR( toCount <= MaxBlobDescs );
	ASSERT_EXPR( 0 <= dimNum && dimNum < CBlobDesc::MaxDimensions );

	CFloatDescArray toArr;
	toArr.Count = toCount;
	int s[CBlobDesc::MaxDimensions];
	for(int i = 0; i < toCount; ++i) {
		toArr.Descs[i] = to[i];
		ASSERT_EXPR( toData[i].GetMathEngine() == this );
		toArr.Data[i] = toData[i];

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

	const int heightNorm = ( height + 15 ) / 16;
	int wStart = 0;
	for( int i = 0; i < toArr.Count; i++ ) {
		C2DKernel kernel( *queue, "matrixKernelBlobSplitByDim", 1, 1, heightNorm, toArr.Widths[i] );
		kernel.SetParam( height, 0 );
		kernel.SetParam( width, 1 );
		kernel.SetParam( from, 2 );
		kernel.SetParam( fromData, 3 );
		kernel.SetParam( toArr.Descs[i], 4 );
		kernel.SetParam( toArr.Data[i], 5 );
		kernel.SetParam( kernel.GetGridHeight(), 6 );
		kernel.SetParam( wStart, 7 );
		kernel.SetParam( toArr.Widths[i], 8 );
		ASSERT_EXPR( kernel.Run() );
		
		wStart += toArr.Widths[i];
	}
}

void CMetalMathEngine::BlobMergeByDim( TBlobDim dim, const CBlobDesc* from, const CFloatHandle* fromData,
	int fromCount, const CBlobDesc& to, const CFloatHandle& toData )
{
	ASSERT_EXPR(dim < BD_Count && fromCount <= MaxBlobDescs);
	blobMergeByDim(dim, from, fromData, fromCount, to, toData);
}

void CMetalMathEngine::BlobMergeByDim( TBlobDim /*dim*/, const CBlobDesc* /*from*/, const CIntHandle* /*fromData*/,
	int /*fromCount*/, const CBlobDesc& /*to*/, const CIntHandle& /*toData*/ )
{
	ASSERT_EXPR(false);
}

void CMetalMathEngine::BlobSplitByDim( TBlobDim dim, const CBlobDesc& from, const CConstFloatHandle& fromData,
	const CBlobDesc* to, const CFloatHandle* toData, int toCount )
{
	ASSERT_EXPR(dim < BD_Count && toCount <= MaxBlobDescs);
	blobSplitByDim(dim, from, fromData, to, toData, toCount);
}

void CMetalMathEngine::BlobSplitByDim( TBlobDim /*dim*/, const CBlobDesc& /*from*/, const CConstIntHandle& /*fromData*/,
	const CBlobDesc* /*to*/, const CIntHandle* /*toData*/, int /*toCount*/ )
{
	ASSERT_EXPR(false);
}

static const int BlobResizeImageCombine = 16;

void CMetalMathEngine::BlobResizeImage( const CBlobDesc& from, const CFloatHandle& fromData, int deltaLeft, int deltaRight,
	int deltaTop, int deltaBottom, TBlobResizePadding padding, float defaultValue,
	const CBlobDesc& to, const CFloatHandle& toData )
{
	ASSERT_EXPR( fromData.GetMathEngine() == this );
	ASSERT_EXPR( toData.GetMathEngine() == this );
	ASSERT_EXPR( padding == TBlobResizePadding::Constant );

	const int geom = to.Height() * to.Width();
	const int totalChannels = to.Channels() * to.Depth();

	C3DKernel kernel( *queue, "cubeKernelBlobResizeImage", 1, 1, BlobResizeImageCombine, to.ObjectCount(), totalChannels, geom );
	kernel.SetParam( from, 0 );
	kernel.SetParam( fromData, 1 );
	kernel.SetParam( deltaLeft, 2 );
	kernel.SetParam( deltaRight, 3 );
	kernel.SetParam( deltaTop, 4 );
	kernel.SetParam( deltaBottom, 5 );
	kernel.SetParam( defaultValue, 6 );
	kernel.SetParam( to, 7 );
	kernel.SetParam( toData, 8 );
	ASSERT_EXPR( kernel.Run() );
}

static const int BlobGetSubSequenceCombine = 16;

void CMetalMathEngine::BlobGetSubSequence( const CBlobDesc& from, const CFloatHandle& fromData, const CIntHandle& indexHandle,
	const CBlobDesc& to, const CFloatHandle& toData, int startPos, bool isRev )
{
	ASSERT_EXPR( fromData.GetMathEngine() == this );
	ASSERT_EXPR( indexHandle.IsNull() || indexHandle.GetMathEngine() == this );
	ASSERT_EXPR( toData.GetMathEngine() == this );	

	const char* kernelName = indexHandle.IsNull() ? "cubeKernelBlobGetSubSequenceNoIndex" : "cubeKernelBlobGetSubSequence";
	C3DKernel kernel( *queue, kernelName,
		1, 1, BlobGetSubSequenceCombine, to.BatchLength(), to.BatchWidth() * to.ListSize(), from.ObjectSize() );
	kernel.SetParam( from, 0 );
	kernel.SetParam( fromData, 1 );
	kernel.SetParam( to, 2 );
	kernel.SetParam( toData, 3 );
	kernel.SetParam( startPos, 4 );
	kernel.SetParam( isRev ? 1 : 0, 5 );
	if( !indexHandle.IsNull() ) {
		kernel.SetParam( indexHandle, 6 );
	}
	ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::Upsampling2DForward( const CBlobDesc& input, const CConstIntHandle& inputData, int heightCopyCount,
	int widthCopyCount, const CBlobDesc& result, const CIntHandle& resultData )
{
	ASSERT_EXPR( inputData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	ASSERT_EXPR( heightCopyCount > 0 );
	ASSERT_EXPR( widthCopyCount > 0 );
	ASSERT_EXPR( input.BatchLength() == result.BatchLength() );
	ASSERT_EXPR( input.BatchWidth() == result.BatchWidth() );
	ASSERT_EXPR( input.Channels() == result.Channels() );
	ASSERT_EXPR( input.Depth() == result.Depth() );
	ASSERT_EXPR( input.Height() * heightCopyCount == result.Height() );
	ASSERT_EXPR( input.Width() * widthCopyCount == result.Width() );

	const int inputHeight = input.Height();
	const int inputRowSize = input.Width() * input.Depth() * input.Channels();
	const int pixelSize = input.Depth() * input.Channels();
	const int resultHeight = result.Height();
	const int resultRowSize = result.Width() * result.Depth() * result.Channels();

	C2DKernel kernel( *queue, "matrixKernelUpsampling2DForwardInt", 1, 1, resultHeight, resultRowSize );
	kernel.SetParam( heightCopyCount, 0 );
	kernel.SetParam( widthCopyCount, 1 );
	kernel.SetParam( pixelSize, 2 );
	kernel.SetParam( input.ObjectCount(), 3 );
	kernel.SetParam( inputHeight, 4 );
	kernel.SetParam( inputRowSize, 5 );
	kernel.SetParam( inputData, 6 );
	kernel.SetParam( resultHeight, 7 );
	kernel.SetParam( resultRowSize, 8 );
	kernel.SetParam( resultData, 9 );
	ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::Upsampling2DForward( const CBlobDesc& input, const CConstFloatHandle& inputData, int heightCopyCount,
	int widthCopyCount, const CBlobDesc& result, const CFloatHandle& resultData )
{
	ASSERT_EXPR( inputData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	ASSERT_EXPR( heightCopyCount > 0 );
	ASSERT_EXPR( widthCopyCount > 0 );
	ASSERT_EXPR( input.BatchLength() == result.BatchLength() );
	ASSERT_EXPR( input.BatchWidth() == result.BatchWidth() );
	ASSERT_EXPR( input.Channels() == result.Channels() );
	ASSERT_EXPR( input.Depth() == result.Depth() );
	ASSERT_EXPR( input.Height() * heightCopyCount == result.Height() );
	ASSERT_EXPR( input.Width() * widthCopyCount == result.Width() );

	const int inputHeight = input.Height();
	const int inputRowSize = input.Width() * input.Depth() * input.Channels();
	const int pixelSize = input.Depth() * input.Channels();
	const int resultHeight = result.Height();
	const int resultRowSize = result.Width() * result.Depth() * result.Channels();

	C2DKernel kernel( *queue, "matrixKernelUpsampling2DForwardFloat", 1, 1, resultHeight, resultRowSize );
	kernel.SetParam( heightCopyCount, 0 );
	kernel.SetParam( widthCopyCount, 1 );
	kernel.SetParam( pixelSize, 2 );
	kernel.SetParam( input.ObjectCount(), 3 );
	kernel.SetParam( inputHeight, 4 );
	kernel.SetParam( inputRowSize, 5 );
	kernel.SetParam( inputData, 6 );
	kernel.SetParam( resultHeight, 7 );
	kernel.SetParam( resultRowSize, 8 );
	kernel.SetParam( resultData, 9 );
	ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::Upsampling2DBackward( const CBlobDesc&, const CConstFloatHandle&, int,
	int, const CBlobDesc&, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::BuildIntegerHist( const CConstIntHandle& numbersHandle, int numbersCount,
	const CIntHandle& resultHandle, int maxNumber )
{
	ASSERT_EXPR( numbersHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	VectorFill( resultHandle, 0, maxNumber );

	C1DKernel kernel( *queue, "vectorKernelBuildIntegerHist", 1, numbersCount );
	kernel.SetParam( numbersHandle, 0 );
	kernel.SetParam( numbersCount, 1 );
	kernel.SetParam( resultHandle, 2 );
	ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::Reorg( const CBlobDesc& source, const CFloatHandle& sourceData, int stride, bool isForward,
	const CBlobDesc& result, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CBlobDesc& input = isForward ? source : result;
	C2DKernel kernel( *queue, "blobReorgFloat",
		1, 1, source.ObjectCount() * input.Height(), input.Channels() * input.Width() );
	kernel.SetParam( sourceData, 0 );
	kernel.SetParam( input.Width(), 1 );
	kernel.SetParam( input.Height(), 2 );
	kernel.SetParam( input.Channels(), 3 );
	kernel.SetParam( source.ObjectCount(), 4 );
	kernel.SetParam( stride, 5 );
	kernel.SetParam( isForward, 6 );
	kernel.SetParam( resultData, 7 );
	ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::Reorg( const CBlobDesc& source, const CIntHandle& sourceData, int stride, bool isForward,
	const CBlobDesc& result, const CIntHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CBlobDesc& input = isForward ? source : result;
	C2DKernel kernel( *queue, "blobReorgInt",
		1, 1, source.ObjectCount() * input.Height(), input.Channels() * input.Width()  );
	kernel.SetParam( sourceData, 0 );
	kernel.SetParam( input.Width(), 1 );
	kernel.SetParam( input.Height(), 2 );
	kernel.SetParam( input.Channels(), 3 );
	kernel.SetParam( source.ObjectCount(), 4 );
	kernel.SetParam( stride, 5 );
	kernel.SetParam( isForward, 6 );
	kernel.SetParam( resultData, 7 );
	ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::SpaceToDepth( const CBlobDesc& source, const CConstFloatHandle& sourceData, int blockSize,
	const CBlobDesc& result, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	ASSERT_EXPR( source.ObjectCount() == result.ObjectCount() );
	ASSERT_EXPR( source.Height() == result.Height() * blockSize );
	ASSERT_EXPR( source.Width() == result.Width() * blockSize );
	ASSERT_EXPR( source.Depth() == 1 );
	ASSERT_EXPR( result.Depth() == 1 );
	ASSERT_EXPR( source.Channels() * blockSize * blockSize == result.Channels() );

	C2DKernel kernel( *queue, "spaceToDepthFloat", 1, 1, source.ObjectCount() * result.Height(),
		blockSize * source.Width() * source.Channels() );
	kernel.SetParam( sourceData, 0 );
	kernel.SetParam( source.ObjectCount() * result.Height(), 1 );
	kernel.SetParam( result.Width(), 2 );
	kernel.SetParam( source.Channels(), 3 );
	kernel.SetParam( blockSize, 4 );
	kernel.SetParam( true, 5 );
	kernel.SetParam( resultData, 6 );
	ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::SpaceToDepth( const CBlobDesc& source, const CConstIntHandle& sourceData, int blockSize,
	const CBlobDesc& result, const CIntHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	ASSERT_EXPR( source.ObjectCount() == result.ObjectCount() );
	ASSERT_EXPR( source.Height() == result.Height() * blockSize );
	ASSERT_EXPR( source.Width() == result.Width() * blockSize );
	ASSERT_EXPR( source.Depth() == 1 );
	ASSERT_EXPR( result.Depth() == 1 );
	ASSERT_EXPR( source.Channels() * blockSize * blockSize == result.Channels() );

	C2DKernel kernel( *queue, "spaceToDepthInt", 1, 1, source.ObjectCount() * result.Height(),
		blockSize * source.Width() * source.Channels() );
	kernel.SetParam( sourceData, 0 );
	kernel.SetParam( source.ObjectCount() * result.Height(), 1 );
	kernel.SetParam( result.Width(), 2 );
	kernel.SetParam( source.Channels(), 3 );
	kernel.SetParam( blockSize, 4 );
	kernel.SetParam( true, 5 );
	kernel.SetParam( resultData, 6 );
	ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::DepthToSpace( const CBlobDesc& source, const CConstFloatHandle& sourceData, int blockSize,
	const CBlobDesc& result, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	ASSERT_EXPR( source.ObjectCount() == result.ObjectCount() );
	ASSERT_EXPR( source.Height() * blockSize == result.Height() );
	ASSERT_EXPR( source.Width() * blockSize == result.Width() );
	ASSERT_EXPR( source.Depth() == 1 );
	ASSERT_EXPR( result.Depth() == 1 );
	ASSERT_EXPR( source.Channels() == result.Channels() * blockSize * blockSize );

	C2DKernel kernel( *queue, "spaceToDepthFloat", 1, 1, source.ObjectCount() * result.Height(),
		blockSize * result.Width() * result.Channels() );
	kernel.SetParam( sourceData, 0 );
	kernel.SetParam( source.ObjectCount() * source.Height(), 1 );
	kernel.SetParam( source.Width(), 2 );
	kernel.SetParam( result.Channels(), 3 );
	kernel.SetParam( blockSize, 4 );
	kernel.SetParam( false, 5 );
	kernel.SetParam( resultData, 6 );
	ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::DepthToSpace( const CBlobDesc& source, const CConstIntHandle& sourceData, int blockSize,
	const CBlobDesc& result, const CIntHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	ASSERT_EXPR( source.ObjectCount() == result.ObjectCount() );
	ASSERT_EXPR( source.Height() * blockSize == result.Height() );
	ASSERT_EXPR( source.Width() * blockSize == result.Width() );
	ASSERT_EXPR( source.Depth() == 1 );
	ASSERT_EXPR( result.Depth() == 1 );
	ASSERT_EXPR( source.Channels() == result.Channels() * blockSize * blockSize );

	C2DKernel kernel( *queue, "spaceToDepthInt", 1, 1, source.ObjectCount() * result.Height(),
		blockSize * result.Width() * result.Channels() );
	kernel.SetParam( sourceData, 0 );
	kernel.SetParam( source.ObjectCount() * source.Height(), 1 );
	kernel.SetParam( source.Width(), 2 );
	kernel.SetParam( result.Channels(), 3 );
	kernel.SetParam( blockSize, 4 );
	kernel.SetParam( false, 5 );
	kernel.SetParam( resultData, 6 );
	ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::AddWidthIndex( const CBlobDesc& source, const CConstFloatHandle& sourceData, bool isForward,
	const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	C1DKernel kernel( *queue, "vectorKernelAddWidthIndexFloat", 1, source.ObjectCount() * source.Channels() * source.Width() * source.Height() );
	kernel.SetParam( sourceData, 0 );
	kernel.SetParam( source.Width(), 1 );
	kernel.SetParam( source.Height(), 2 );
	kernel.SetParam( source.Channels(), 3 );
	kernel.SetParam( source.ObjectCount(), 4 );
	kernel.SetParam( isForward, 5 );
	kernel.SetParam( resultData, 6 );
	ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::AddWidthIndex( const CBlobDesc& source, const CConstIntHandle& sourceData, bool isForward,
	const CIntHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	C1DKernel kernel( *queue, "vectorKernelAddWidthIndexInt", 1, source.ObjectCount() * source.Channels() * source.Width() * source.Height() );
	kernel.SetParam( sourceData, 0 );
	kernel.SetParam( source.Width(), 1 );
	kernel.SetParam( source.Height(), 2 );
	kernel.SetParam( source.Channels(), 3 );
	kernel.SetParam( source.ObjectCount(), 4 );
	kernel.SetParam( isForward, 5 );
	kernel.SetParam( resultData, 6 );
	ASSERT_EXPR( kernel.Run() );	
}

void CMetalMathEngine::AddHeightIndex( const CBlobDesc& source, const CConstFloatHandle& sourceData, bool isForward,
	const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	C1DKernel kernel( *queue, "vectorKernelAddHeightIndexFloat", 1, source.ObjectCount() * source.Channels() * source.Width() * source.Height() );
	kernel.SetParam( sourceData, 0 );
	kernel.SetParam( source.Width(), 1 );
	kernel.SetParam( source.Height(), 2 );
	kernel.SetParam( source.Channels(), 3 );
	kernel.SetParam( source.ObjectCount(), 4 );
	kernel.SetParam( isForward, 5 );
	kernel.SetParam( resultData, 6 );
	ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::AddHeightIndex( const CBlobDesc& source, const CConstIntHandle& sourceData, bool isForward,
	const CIntHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	C1DKernel kernel( *queue, "vectorKernelAddHeightIndexInt", 1, source.ObjectCount() * source.Channels() * source.Width() * source.Height() );
	kernel.SetParam( sourceData, 0 );
	kernel.SetParam( source.Width(), 1 );
	kernel.SetParam( source.Height(), 2 );
	kernel.SetParam( source.Channels(), 3 );
	kernel.SetParam( source.ObjectCount(), 4 );
	kernel.SetParam( isForward, 5 );
	kernel.SetParam( resultData, 6 );
	ASSERT_EXPR( kernel.Run() ); 
}

void CMetalMathEngine::MatrixRowsToVectorSquaredL2Distance( const CConstFloatHandle& matrixHandle, const int matrixHeight,
	const int matrixWidth, const CConstFloatHandle& vectorHandle, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	VectorFill( resultHandle, 0, matrixHeight );

	C2DKernel kernel( *queue, "matrixKernelMatrixRowsToVectorSquaredL2Distance", 1, ( matrixWidth + 7 ) / 8, matrixHeight, matrixWidth );
	kernel.SetParam( matrixHandle, 0 );
	kernel.SetParam( matrixHeight, 1 );
	kernel.SetParam( matrixWidth, 2 );
	kernel.SetParam( vectorHandle, 3 );
	kernel.SetParam( resultHandle, 4 );
	kernel.SetSharedParam( kernel.GetThreadCount() * sizeof(float), 5 );
	ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::QrnnFPooling( bool reverse, int sequenceLength, int objectSize,
	const CConstFloatHandle& update, const CConstFloatHandle& forget, const CConstFloatHandle& initialState,
	const CFloatHandle& result )
{
	ASSERT_EXPR( sequenceLength >= 1 );
	ASSERT_EXPR( objectSize >= 1 );
	ASSERT_EXPR( update.GetMathEngine() == this );
	ASSERT_EXPR( forget.GetMathEngine() == this );
	ASSERT_EXPR( initialState.IsNull() || initialState.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );

	C1DKernel kernel( *queue, "vectorQrnnFPooling", 1, objectSize );
	kernel.SetParam( reverse, 0 );
	kernel.SetParam( sequenceLength, 1 );
	kernel.SetParam( objectSize, 2 );
	kernel.SetParam( update, 3 );
	kernel.SetParam( forget, 4 );
	kernel.SetParam( result, 6 );

	if( initialState.IsNull() ) {
		CFloatHandleStackVar zeros( *this, objectSize );
		VectorFill( zeros, 0.f, objectSize );
		kernel.SetParam( zeros, 5 );
		ASSERT_EXPR( kernel.Run() ); 
	} else {
		kernel.SetParam( initialState, 5 );
		ASSERT_EXPR( kernel.Run() ); 
	}
}

void CMetalMathEngine::QrnnFPoolingBackward( bool /*reverse*/, int /*sequenceLength*/, int /*objectSize*/,
	const CConstFloatHandle& /*update*/, const CConstFloatHandle& /*forget*/,
	const CConstFloatHandle& /*initialState*/, const CConstFloatHandle& /*result*/, const CFloatHandle& /*resultDiff*/,
	const CFloatHandle& /*updateDiff*/, const CFloatHandle& /*forgetDiff*/ )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::QrnnIfPooling( bool reverse, int sequenceLength, int objectSize,
	const CConstFloatHandle& update, const CConstFloatHandle& forget, const CConstFloatHandle& input,
	const CConstFloatHandle& initialState, const CFloatHandle& result )
{
	ASSERT_EXPR( sequenceLength >= 1 );
	ASSERT_EXPR( objectSize >= 1 );
	ASSERT_EXPR( update.GetMathEngine() == this );
	ASSERT_EXPR( forget.GetMathEngine() == this );
	ASSERT_EXPR( input.GetMathEngine() == this );
	ASSERT_EXPR( initialState.IsNull() || initialState.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );

	C1DKernel kernel( *queue, "vectorQrnnIfPooling", 1, objectSize );
	kernel.SetParam( reverse, 0 );
	kernel.SetParam( sequenceLength, 1 );
	kernel.SetParam( objectSize, 2 );
	kernel.SetParam( update, 3 );
	kernel.SetParam( forget, 4 );
	kernel.SetParam( input, 5 );
	kernel.SetParam( result, 7 );

	if( initialState.IsNull() ) {
		CFloatHandleStackVar zeros( *this, objectSize );
		VectorFill( zeros, 0.f, objectSize );
		kernel.SetParam( zeros, 6 );
		ASSERT_EXPR( kernel.Run() ); 
	} else {
		kernel.SetParam( initialState, 6 );
		ASSERT_EXPR( kernel.Run() ); 
	}
}

void CMetalMathEngine::QrnnIfPoolingBackward( bool /*reverse*/, int /*sequenceLength*/, int /*objectSize*/,
	const CConstFloatHandle& /*update*/, const CConstFloatHandle& /*forget*/, const CConstFloatHandle& /*input*/,
	const CConstFloatHandle& /*initialState*/, const CConstFloatHandle& /*result*/, const CFloatHandle& /*resultDiff*/,
	const CFloatHandle& /*updateDiff*/, const CFloatHandle& /*forgetDiff*/, const CFloatHandle& /*inputDiff*/ )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::IndRnnRecurrent( bool reverse, int sequenceLength, int batchSize, int objectSize,
	TActivationFunction activation, const CConstFloatHandle& wx, const CConstFloatHandle& mask, const CConstFloatHandle& u,
	const CFloatHandle& h )
{
	ASSERT_EXPR( sequenceLength >= 1 );
	ASSERT_EXPR( batchSize >= 1 );
	ASSERT_EXPR( objectSize >= 1 );
	ASSERT_EXPR( wx.GetMathEngine() == this );
	ASSERT_EXPR( mask.IsNull() ); // Inference-only kernel, that's why dropout can't be applied
	ASSERT_EXPR( u.GetMathEngine() == this );
	ASSERT_EXPR( h.GetMathEngine() == this );
	ASSERT_EXPR( activation == AF_Sigmoid || activation == AF_ReLU );

	C2DKernel kernel( *queue,
		activation == AF_Sigmoid ? "matrixIndRnnRecurrentSigmoid" : "matrixIndRnnRecurrentReLU",
		1, 1, batchSize, objectSize );
	kernel.SetParam( reverse, 0 );
	kernel.SetParam( sequenceLength, 1 );
	kernel.SetParam( batchSize, 2 );
	kernel.SetParam( objectSize, 3 );
	kernel.SetParam( wx, 4 );
	kernel.SetParam( u, 5 );
	kernel.SetParam( h, 6 );

	ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::IndRnnRecurrentBackward( bool /*reverse*/, int /*sequenceLength*/, int /*batchSize*/, int /*objectSize*/,
	TActivationFunction /*activation*/, const CConstFloatHandle& /*mask*/, const CConstFloatHandle& /*u*/,
	const CConstFloatHandle& /*h*/, const CConstFloatHandle& /*hDiff*/, const CFloatHandle& /*wxDiff*/ )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::IndRnnRecurrentLearn( bool /*reverse*/, int /*sequenceLength*/, int /*batchSize*/, int /*objectSize*/,
	TActivationFunction /*activation*/, const CConstFloatHandle& /*mask*/, const CConstFloatHandle& /*u*/,
	const CConstFloatHandle& /*h*/, const CConstFloatHandle& /*hDiff*/, const CFloatHandle& /*uDiff*/ )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::CtcLossForward( int /*resultLen*/, int /*batchSize*/, int /*classCount*/, int /*labelLen*/,
	int /*blankLabel*/, bool /*skipBlanks*/, const CConstFloatHandle& /*result*/, const CConstIntHandle& /*labels*/,
	const CConstIntHandle& /*labelLens*/, const CConstIntHandle& /*resultLens*/, const CConstFloatHandle& /*labelWeights*/,
	const CFloatHandle& /*loss*/, const CFloatHandle& /*lossGradient*/ )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::BertConv( const CConstFloatHandle& dataHandle, const CConstFloatHandle& kernelHandle, int seqLen,
	int batchSize, int numHeads, int headSize, int kernelSize, const CFloatHandle& outputHandle )
{
	ASSERT_EXPR( dataHandle.GetMathEngine() == this );
	ASSERT_EXPR( kernelHandle.GetMathEngine() == this );
	ASSERT_EXPR( outputHandle.GetMathEngine() == this );

	C1DKernel kernel( *queue, "vectorBertConv", 1, seqLen * batchSize * numHeads * headSize * kernelSize );
	kernel.SetParam( dataHandle, 0 );
	kernel.SetParam( kernelHandle, 1 );
	kernel.SetParam( seqLen, 2 );
	kernel.SetParam( batchSize, 3 );
	kernel.SetParam( numHeads, 4 );
	kernel.SetParam( headSize, 5 );
	kernel.SetParam( kernelSize, 6 );
	kernel.SetParam( outputHandle, 7 );

	ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::BertConvBackward( const CConstFloatHandle& /*dataHandle*/, const CConstFloatHandle& /*kernelHandle*/,
	const CConstFloatHandle& /*outDiffHandle*/, int /*seqLen*/, int /*batchSize*/, int /*numHeads*/, int /*headSize*/, int /*kernelSize*/,
	const CFloatHandle& /*dataDiffHandle*/, const CFloatHandle& /*kernelDiffHandle*/ )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::LinearInterpolation( const CConstFloatHandle&, const CFloatHandle&,
	TInterpolationCoords, TInterpolationRound, int, int, int, float )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::ScatterND( const CConstIntHandle&, const CConstFloatHandle&, const CFloatHandle&,
	const CBlobDesc&, int, int )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::ScatterND( const CConstIntHandle&, const CConstIntHandle&, const CIntHandle&,
	const CBlobDesc&, int, int )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::ChannelwiseWith1x1( const CBlobDesc&, const CBlobDesc&,
	const CRowwiseOperationDesc&, const CChannelwiseConvolutionDesc&,
	const CConstFloatHandle&, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::MobileNetV2Block( const CBlobDesc&, const CBlobDesc&,
	const CRowwiseOperationDesc&, const CChannelwiseConvolutionDesc&,
	const CConstFloatHandle&, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::MobileNetV3PreSEBlock( const CBlobDesc&, const CBlobDesc&, const CChannelwiseConvolutionDesc&,
	const CConstFloatHandle&, const CConstFloatHandle&, const CConstFloatHandle*, TActivationFunction, float,
	const CConstFloatHandle&, const CConstFloatHandle*, TActivationFunction, float, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::MobileNetV3PostSEBlock( const CBlobDesc&, int, const CConstFloatHandle&,
	const CConstFloatHandle&, const CConstFloatHandle*, TActivationFunction, float, const CConstFloatHandle&,
	const CConstFloatHandle*, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

} // namespace NeoML

#endif // NEOML_USE_METAL
