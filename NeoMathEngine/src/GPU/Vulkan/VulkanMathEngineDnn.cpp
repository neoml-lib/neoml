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

#include <common.h>
#pragma hdrstop

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_VULKAN

#include <VulkanMathEngine.h>
#include <VulkanShader.h>
#include <MathEngineCommon.h>
#include <MathEngineDnnDropout.h>

namespace NeoML {

// Include the shader code
#include <shaders/generated/Upsampling2DForward.h>
#include <shaders/generated/BlobResizeImage.h>
#include <shaders/generated/BlobSpatialDropout.h>
#include <shaders/generated/BuildIntegerHist.h>
#include <shaders/generated/BlobGetSubSequence.h>
#include <shaders/generated/BlobGetSubSequenceNoIndices.h>
#include <shaders/generated/BlobSplitByDim.h>
#include <shaders/generated/BlobMergeByDim.h>
#include <shaders/generated/BlobReorgFloat.h>
#include <shaders/generated/BlobReorgInt.h>


//------------------------------------------------------------------------------------------------------------

constexpr int FloatDescArrayMaxBlobs = 32;

struct CFloatDescArray {
	int Count;
	CBlobDesc Descs[FloatDescArrayMaxBlobs];
	CFloatHandle Data[FloatDescArrayMaxBlobs];
	int Widths[FloatDescArrayMaxBlobs];
};

void CVulkanMathEngine::blobMergeByDim(int dimNum, const CBlobDesc* from, const CFloatHandle* fromData, int fromCount,
	const CBlobDesc& to, const CFloatHandle& toData)
{
    ASSERT_EXPR( toData.GetMathEngine() == this );
	ASSERT_EXPR( fromCount <= MaxBlobDescs );
	ASSERT_EXPR( 0 < dimNum && dimNum < CBlobDesc::MaxDimensions );

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
    
    const int heightNorm = Ceil( height, 16 );
    int wStart = 0;
    for( int i = 0; i < fromArr.Count; i++ ) {
    	CMemoryHandle bufs[2] = { fromData[i], toData };
		size_t sizes[2] = { from[i].BlobSize() * sizeof(float), to.BlobSize() * sizeof(float) };

		PARAM_STRUCT(BlobMergeByDim) param = { 
			height,
			width,
			fromArr.Widths[i],
			wStart,
			height
		};

		runShader( shaderLoader->GET_SHADER_DATA(BlobMergeByDim, true, 0, 0, 2),
			&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2, fromArr.Widths[i], heightNorm, 1 );

        wStart += fromArr.Widths[i];
    }
}

void CVulkanMathEngine::blobSplitByDim(int dimNum, const CBlobDesc& from, const CFloatHandle& fromData,
	const CBlobDesc* to, const CFloatHandle* toData, int toCount)
{
    ASSERT_EXPR( fromData.GetMathEngine() == this );
    ASSERT_EXPR( toCount <= MaxBlobDescs );
	ASSERT_EXPR( 0 < dimNum && dimNum < CBlobDesc::MaxDimensions );

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
    
    const int heightNorm = Ceil( height, 16 );
    int wStart = 0;
    for( int i = 0; i < toArr.Count; i++ ) {
    	CMemoryHandle bufs[2] = { fromData, toData[i] };
		size_t sizes[2] = { from.BlobSize() * sizeof(float), to[i].BlobSize() * sizeof(float) };

		PARAM_STRUCT(BlobSplitByDim) param = { 
			height,
			width,
			toArr.Widths[i],
			wStart,
			height
		};

		runShader( shaderLoader->GET_SHADER_DATA(BlobSplitByDim, true, 0, 0, 2),
			&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2, toArr.Widths[i], heightNorm, 1 );

        wStart += toArr.Widths[i];
    }
}

void CVulkanMathEngine::BlobMergeByDim( TBlobDim dim, const CBlobDesc* from, const CFloatHandle* fromData,
	int fromCount, const CBlobDesc& to, const CFloatHandle& toData )
{
	ASSERT_EXPR(dim < BD_Count && fromCount <= MaxBlobDescs);
	blobMergeByDim(dim, from, fromData, fromCount, to, toData);
}

void CVulkanMathEngine::BlobMergeByDim(TBlobDim /*dim*/, const CBlobDesc* /*from*/, const CIntHandle* /*fromData*/, int /*fromCount*/, const CBlobDesc& /*to*/, const CIntHandle& /*toData*/)
{
	ASSERT_EXPR(false);
}

void CVulkanMathEngine::BlobSplitByDim( TBlobDim dim, const CBlobDesc& from, const CFloatHandle& fromData,
	const CBlobDesc* to, const CFloatHandle* toData, int toCount )
{
	ASSERT_EXPR(0 <= dim && dim < CBlobDesc::MaxDimensions);
	blobSplitByDim(dim, from, fromData, to, toData, toCount);
}

void CVulkanMathEngine::BlobSplitByDim(TBlobDim /*dim*/, const CBlobDesc& /*from*/, const CIntHandle& /*fromData*/, const CBlobDesc* /*to*/, const CIntHandle* /*toData*/, int /*toCount*/)
{
	ASSERT_EXPR(false);
}

static const int BlobResizeImageCombine = 16;

void CVulkanMathEngine::BlobResizeImage( const CBlobDesc& from, const CFloatHandle& fromData, int deltaLeft,
	int deltaRight, int deltaTop, int deltaBottom, float defaultValue, const CBlobDesc& to, const CFloatHandle& toData )
{
	const int geom = to.Height() * to.Width();
	const int totalChannels = to.Channels() * to.Depth();

	CMemoryHandle bufs[2] = { fromData, toData };
	size_t sizes[2] = { from.BlobSize() * sizeof(float), to.BlobSize() * sizeof(float) };

	PARAM_STRUCT(BlobResizeImage) param =
		{ from.ObjectCount(), totalChannels, from.Height(), from.Width(), to.Height(), to.Width(), deltaLeft, deltaRight, deltaTop, deltaBottom, defaultValue };

	runShader( shaderLoader->GET_SHADER_DATA(BlobResizeImage, true, 0, 0, 2),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2, Ceil(geom, BlobResizeImageCombine), totalChannels, to.ObjectCount() );
}

void CVulkanMathEngine::BlobGetSubSequence( const CBlobDesc& from, const CFloatHandle& fromData, const CIntHandle& indexHandle,
	const CBlobDesc& to, const CFloatHandle& toData, int startPos, bool isRev )
{
	if( indexHandle.IsNull() ) {
		CMemoryHandle bufs[2] = { fromData, toData };
		size_t sizes[2] = { from.BlobSize() * sizeof(float), to.BlobSize() * sizeof(float) };

		PARAM_STRUCT(BlobGetSubSequenceNoIndices) param =
			{ from.ListSize(), from.BatchWidth(), to.BatchLength(), to.ObjectSize(), startPos, isRev ? 1 : 0 };

		runShader( shaderLoader->GET_SHADER_DATA(BlobGetSubSequenceNoIndices, true, 0, 0, 2),
			&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2, Ceil(to.ObjectSize(), 16), to.BatchWidth() * to.ListSize(), to.BatchLength() );
	} else {
		CMemoryHandle bufs[3] = { fromData, indexHandle, toData };
		size_t sizes[3] = { from.BlobSize() * sizeof(float), to.ObjectCount() * sizeof(int), to.BlobSize() * sizeof(float) };

		PARAM_STRUCT(BlobGetSubSequence) param =
			{ from.ListSize(), from.BatchWidth(), to.BatchLength(), to.ObjectSize(), startPos, isRev ? 1 : 0 };

		runShader( shaderLoader->GET_SHADER_DATA(BlobGetSubSequence, true, 0, 0, 3),
			&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 3, Ceil(to.ObjectSize(), 16), to.BatchWidth() * to.ListSize(), to.BatchLength() );
	}
}

void CVulkanMathEngine::Upsampling2DForward( const CBlobDesc& input, const CFloatHandle& inputData, int heightCopyCount,
	int widthCopyCount, const CBlobDesc& result, const CFloatHandle& resultData )
{
	ASSERT_EXPR( inputData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	CMemoryHandle bufs[2] = { inputData, resultData };
	size_t sizes[2] = { input.BlobSize() * sizeof(float), result.BlobSize() * sizeof(float) };

	const int inputHeight = input.Height();
	const int inputRowSize = input.Width() * input.Depth() * input.Channels();
	const int pixelSize = input.Depth() * input.Channels();
	const int resultHeight = result.Height();
	const int resultRowSize = result.Width() * result.Depth() * result.Channels();

	PARAM_STRUCT(Upsampling2DForward) param = { 
		heightCopyCount, 
		widthCopyCount,
		pixelSize,
		input.ObjectCount(),
		inputHeight,
		inputRowSize,
		resultHeight,
		resultRowSize,
	};

	runShader( shaderLoader->GET_SHADER_DATA(Upsampling2DForward, true, 0, 0, 2), &param, sizeof(param),
		0, 0, 0, 0, bufs, sizes, 2, resultRowSize, resultHeight, 1 );
}

void CVulkanMathEngine::Upsampling2DBackward( const CBlobDesc&, const CFloatHandle&, int, int, const CBlobDesc&,
	const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::BuildIntegerHist( const CConstIntHandle& numbersHandle, int numbersCount,
	const CIntHandle& resultHandle, int maxNumber )
{
	CMemoryHandle bufs[2] = { numbersHandle, resultHandle };
	size_t sizes[2] = { numbersCount * sizeof(int), maxNumber * sizeof(float) };

	runVectorShader( shaderLoader->GET_SHADER_DATA(BuildIntegerHist, true, 0, 0, 2), 0, 0,
		0, 0, 0, 0, bufs, sizes, 2, numbersCount );
}

void CVulkanMathEngine::MatrixRowsToVectorSquaredL2Distance( const CConstFloatHandle&, const int, const int,
	const CConstFloatHandle&, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::Reorg( const CBlobDesc& source, const CFloatHandle& sourceData, int stride, bool isForward,
	const CBlobDesc& result, const CFloatHandle& resultData )
{
	CMemoryHandle bufs[2] = { sourceData, resultData };
	size_t sizes[2] = { source.BlobSize() * sizeof( float ), result.BlobSize() * sizeof( float ) };

	const CBlobDesc& input = ( isForward ) ? source : result;
	PARAM_STRUCT( BlobReorgFloat ) param = { source.ObjectCount(), input.Height(), input.Width(), input.Channels(),
		stride, input.Channels() / ( stride * stride ), isForward ? 1 : 0 };

	runShader( shaderLoader->GET_SHADER_DATA( BlobReorgFloat, true, 0, 0, 2 ),
		&param, sizeof( param ), 0, 0, 0, 0, bufs, sizes, 2, input.BatchWidth() * input.Height(), input.Channels() * input.Width(), 1 );
}

void CVulkanMathEngine::Reorg( const CBlobDesc& source, const CIntHandle& sourceData, int stride, bool isForward,
	const CBlobDesc& result, const CIntHandle& resultData )
{
	CMemoryHandle bufs[2] = { sourceData, resultData };
	size_t sizes[2] = { source.BlobSize() * sizeof( float ), result.BlobSize() * sizeof( float ) };

	const CBlobDesc& input = ( isForward ) ? source : result;
	PARAM_STRUCT( BlobReorgInt ) param = { source.ObjectCount(), input.Height(), input.Width(), input.Channels(),
		stride,  input.Channels() / ( stride * stride ), isForward ? 1 : 0 };

	runShader( shaderLoader->GET_SHADER_DATA( BlobReorgInt, true, 0, 0, 2 ),
		&param, sizeof( param ), 0, 0, 0, 0, bufs, sizes, 2, input.BatchWidth() * input.Height(), input.Channels() * input.Width(), 1 );
}

void CVulkanMathEngine::AddWidthIndex( const CBlobDesc&, const CFloatHandle&, bool, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::AddWidthIndex( const CBlobDesc&, const CIntHandle&, bool, const CIntHandle& )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::AddHeightIndex( const CBlobDesc&, const CFloatHandle&, bool, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::AddHeightIndex( const CBlobDesc&, const CIntHandle&, bool, const CIntHandle& )
{
	ASSERT_EXPR( false );
}

CDropoutDesc* CVulkanMathEngine::InitDropout( float rate, bool isSpatial, bool isBatchwise,
	const CBlobDesc& input, const CBlobDesc& output, int seed )
{
	return new CMathEngineDropoutDesc( mathEngine(), rate, isSpatial, isBatchwise, input, output, seed );
}

void CVulkanMathEngine::Dropout( const CDropoutDesc& dropoutDesc, const CFloatHandle& inputData, const CFloatHandle& outputData )
{
	ASSERT_EXPR( inputData.GetMathEngine() == this );
	ASSERT_EXPR( outputData.GetMathEngine() == this );

	const CMathEngineDropoutDesc& desc = static_cast<const CMathEngineDropoutDesc&>( dropoutDesc );
	const CBlobDesc& input = desc.Input;
    const CBlobDesc& output = desc.Output;

	if( desc.ForwardRate == 1.f ) {
		VectorCopy( outputData, inputData, input.BlobSize() );
		return;
	}

	const int objectSize = desc.IsSpatial ? input.Channels() : input.ObjectSize();
	const int batchLength = desc.IsBatchwise ? input.ObjectCount() : input.BatchLength();
	const int batchWidth = input.ObjectCount() / batchLength;
	const int maskSize = batchWidth * objectSize;

	ASSERT_EXPR( desc.Mask.Size() == maskSize );

	if( !desc.IsSpatial ) {
		MultiplyMatrixByDiagMatrix( inputData, batchLength, maskSize, desc.Mask, outputData, output.BlobSize() );
		return;
	}

	const int maskObjectSize = maskSize / batchWidth;

	CMemoryHandle bufs[3] = { inputData, desc.Mask.GetHandle(), outputData };
	size_t sizes[3] = { input.BlobSize() * sizeof(float), maskSize * sizeof(float), output.BlobSize() * sizeof(float) };

	PARAM_STRUCT(BlobSpatialDropout) param = {
		input.ObjectCount(),
		input.ObjectSize(),
		batchWidth,
		maskObjectSize, 
	};

	runShader( shaderLoader->GET_SHADER_DATA(BlobSpatialDropout, true, 0, 0, 3), &param, sizeof(param),
		0, 0, 0, 0, bufs, sizes, 3, maskObjectSize, input.ObjectSize() / maskObjectSize, input.ObjectCount() );
}

} // namespace NeoML

#endif // NEOML_USE_VULKAN
