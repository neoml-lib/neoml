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

#include <NeoMathEngine/CrtAllocatedObject.h>
#include <MetalMathEngine.h>
#include <MathEngineDnnConv.h>
#include <MathEngineCommon.h>
#include <MetalKernel.h>
#include <algorithm>

@import Foundation;
@import MetalKit;

namespace NeoML {

//-------------------------------------------------------------------------------------------------------------------
// Convolution

CConvolutionDesc* CMetalMathEngine::InitBlobConvolution( const CBlobDesc& source, int paddingHeight, int paddingWidth,
	int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, const CBlobDesc& filter, const CBlobDesc& result,
    const CActivationInfo& activation )
{
	ASSERT_EXPR( strideHeight > 0 );
	ASSERT_EXPR( strideWidth > 0 );
	ASSERT_EXPR( paddingHeight >= 0 );
	ASSERT_EXPR( paddingWidth >= 0 );
	ASSERT_EXPR( dilationHeight > 0 );
	ASSERT_EXPR( dilationWidth > 0 );
	ASSERT_EXPR( source.Channels() == filter.Channels() );
	ASSERT_EXPR( source.Depth() == filter.Depth() );
	ASSERT_EXPR( filter.Height() <= source.Height() + 2 * paddingHeight );
	ASSERT_EXPR( filter.Width() <= source.Width() + 2 * paddingWidth );
	ASSERT_EXPR( filter.BatchLength() == 1 );
	ASSERT_EXPR( result.BatchLength() == source.BatchLength() );
	ASSERT_EXPR( result.BatchWidth() == source.BatchWidth() );
	ASSERT_EXPR( result.Height() == 1 + ( source.Height() -
		( filter.Height() - 1 ) * dilationHeight + 2 * paddingHeight - 1 ) / strideHeight );
	ASSERT_EXPR( result.Width() == 1 + ( source.Width() -
		( filter.Width() - 1 ) * dilationWidth + 2 * paddingWidth - 1 ) / strideWidth );
	ASSERT_EXPR( result.Channels() == filter.BatchWidth() );
	ASSERT_EXPR( result.Depth() == 1 );
    ASSERT_EXPR( IsInPlaceActivation( activation.Type ) );

	CCommonConvolutionDesc* desc = new CCommonConvolutionDesc( source, result, filter, paddingHeight, paddingWidth,
		strideHeight, strideWidth, dilationHeight, dilationWidth,
        dynamic_cast<CCommonActivationDesc*>( InitActivation( activation, result.BlobSize() ) ) );
	return desc;
}

void CMetalMathEngine::BlobConvolution( const CConvolutionDesc& convDesc,
	const CFloatHandle& sourceData, const CFloatHandle& filterData, const CFloatHandle* freeTermData,
	const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( filterData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermData == 0 || freeTermData->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCommonConvolutionDesc& desc = static_cast<const CCommonConvolutionDesc&>( convDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Result;

    if( filter.Height() == 1 && filter.Width() == 1 && desc.PaddingHeight == 0 && desc.PaddingWidth == 0 && desc.StrideHeight == 1 && desc.StrideWidth == 1 ) {
        int channels = source.Depth() * source.Channels();
        int resultChannels = filter.ObjectCount();

        if( freeTermData == 0 ) {
            MultiplyMatrixByTransposedMatrix(sourceData, source.BlobSize() / channels, channels, channels,
                filterData, resultChannels, channels, resultData, resultChannels, result.BlobSize());
        } else {
            // TODO: Combine this kernels
            SetVectorToMatrixRows(resultData, result.BlobSize() / resultChannels, resultChannels, *freeTermData);
            multiplyMatrixByTransposedMatrixAndAdd(sourceData, source.BlobSize() / channels, channels, channels,
                filterData, resultChannels, channels, resultData, resultChannels, result.BlobSize());
        }

    } else {
        const int tempMatrixWidth = filter.ObjectSize();
        const int tempMatrixHeight = result.ObjectSize() / filter.ObjectCount();
        constexpr int memorySize = 2 * 1024 *1024;
        const int maxPossibleTempMatrixHeight = static_cast<int>( MAX( 1, ( memorySize / ( 8 * tempMatrixWidth ) ) ) );
        const int tempMatrixHeightBatchSize = MIN( tempMatrixHeight, maxPossibleTempMatrixHeight );

        CFloatHandleStackVar temp( mathEngine(), tempMatrixHeightBatchSize * tempMatrixWidth );

        for( int b = 0; b < source.ObjectCount(); b++ ) {
            int tempMatrixHeightIndex = 0;
            while( tempMatrixHeightIndex < tempMatrixHeight ) {
                int curTempMatrixHeight = MIN( tempMatrixHeight - tempMatrixHeightIndex, tempMatrixHeightBatchSize );
                
                C2DKernel kernel( *queue, "matrixKernelBlobConvolutionPrepare",
                    1, 1, curTempMatrixHeight, filter.Channels() * filter.Depth() );
                kernel.SetParam( source, 0 );
                kernel.SetParam( sourceData + b * source.ObjectSize(), 1 );
                kernel.SetParam( desc.PaddingHeight, 2 );
                kernel.SetParam( desc.PaddingWidth, 3 );
                kernel.SetParam( desc.StrideHeight, 4 );
                kernel.SetParam( desc.StrideWidth, 5 );
                kernel.SetParam( desc.DilationHeight, 6 );
                kernel.SetParam( desc.DilationWidth, 7 );
                kernel.SetParam( filter, 8 );
                kernel.SetParam( result, 9 );
                kernel.SetParam( tempMatrixHeightIndex, 10 );
                kernel.SetParam( curTempMatrixHeight, 11 );
                kernel.SetParam( temp, 12 );
                ASSERT_EXPR( kernel.Run() );

                MultiplyMatrixByTransposedMatrix( temp, curTempMatrixHeight, filter.ObjectSize(), filter.ObjectSize(),
                    filterData, filter.ObjectCount(), filter.ObjectSize(),
                    resultData + b * result.ObjectSize() + tempMatrixHeightIndex * filter.ObjectCount(),
                    filter.ObjectCount(), curTempMatrixHeight * filter.ObjectCount() );

                tempMatrixHeightIndex += curTempMatrixHeight;
            }

            if( freeTermData != 0 ) {
                C2DKernel kernel( *queue, "matrixKernelAddVectorToMatrixRows",
                    1, 1, result.ObjectSize() / filter.ObjectCount(), filter.ObjectCount() );
                kernel.SetParam( 1, 0 );
                kernel.SetParam( resultData + b * result.ObjectSize(), 1 );
                kernel.SetParam( resultData + b * result.ObjectSize(), 2 );
                kernel.SetParam( result.ObjectSize() / filter.ObjectCount(), 3 );
                kernel.SetParam( filter.ObjectCount(), 4 );
                kernel.SetParam( *freeTermData, 5 );
                ASSERT_EXPR( kernel.Run() );
            }
        }
    }

    Activation( *desc.Activation, resultData, resultData, desc.Result.BlobSize() );
}

void CMetalMathEngine::BlobConvolutionBackward( const CConvolutionDesc& convDesc, const CFloatHandle& outputData, const CFloatHandle& outputDiffData,
	const CFloatHandle& filterData, const CFloatHandle* freeTermData, const CFloatHandle& inputDiffData )
{
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( filterData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermData == 0 || freeTermData->GetMathEngine() == this );
	ASSERT_EXPR( inputDiffData.GetMathEngine() == this );

	const CCommonConvolutionDesc& desc = static_cast<const CCommonConvolutionDesc&>( convDesc );
	const CBlobDesc& inputDiff = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& outputDiff = desc.Result;

	if( !outputData.IsNull() ) {
		ActivationBackward( *desc.Activation, CFloatHandle(), outputData, outputDiffData, outputDiffData, desc.Result.BlobSize() );
	}
    
    const int tempBufferSize = outputDiff.Height() * outputDiff.Width() * filter.ObjectSize();
    CFloatHandleStackVar temp( mathEngine(), tempBufferSize );
    
    const int outputDiffGeometry = outputDiff.Height() * outputDiff.Width();
    for( int b = 0; b < outputDiff.ObjectCount(); b++ ) {
        MultiplyMatrixByMatrix( 1, outputDiffData + b * outputDiff.ObjectSize(), outputDiffGeometry,
            outputDiff.Channels() * outputDiff.Depth(),filterData, filter.ObjectSize(), temp, tempBufferSize );

        if( freeTermData != 0 ) {
            SetVectorToMatrixRows( inputDiffData + b * inputDiff.ObjectSize(),
                inputDiff.Height() * inputDiff.Width(), inputDiff.Depth() * inputDiff.Channels(), *freeTermData );
        } else {
            VectorFill( inputDiffData + b * inputDiff.ObjectSize(), 0.0f, inputDiff.ObjectSize() );
        }
        
        C1DKernel kernel( *queue, "vectorKernelBlobConvolutionBackward", 1, inputDiff.Height() );
        kernel.SetParam( outputDiff, 0 );
        kernel.SetParam( desc.PaddingHeight, 1 );
        kernel.SetParam( desc.PaddingWidth, 2 );
        kernel.SetParam( desc.StrideHeight, 3 );
        kernel.SetParam( desc.StrideWidth, 4 );
        kernel.SetParam( desc.DilationHeight, 5 );
        kernel.SetParam( desc.DilationWidth, 6 );
        kernel.SetParam( filter, 7 );
        kernel.SetParam( temp, 8 );
        kernel.SetParam( inputDiff, 9 );
        kernel.SetParam( inputDiffData + b * inputDiff.ObjectSize(), 10 );
        ASSERT_EXPR( kernel.Run() );
    }
}

void CMetalMathEngine::BlobConvolutionLearnAdd( const CConvolutionDesc&, const CFloatHandle&, const CFloatHandle&,
	const CFloatHandle&, const CFloatHandle&, const CFloatHandle*, bool )
{
	ASSERT_EXPR( false );
}

//----------------------------------------------------------------------------------------------------------------------------------------
// 3D convolution

C3dConvolutionDesc* CMetalMathEngine::InitBlob3dConvolution( const CBlobDesc& source,
	int paddingHeight, int paddingWidth, int paddingDepth,
	int strideHeight, int strideWidth, int strideDepth,
	const CBlobDesc& filter, const CBlobDesc& result, const CActivationInfo& activation )
{
    ASSERT_EXPR( paddingHeight >= 0 );
    ASSERT_EXPR( paddingWidth >= 0 );
    ASSERT_EXPR( paddingDepth >= 0 );
    ASSERT_EXPR( strideHeight > 0 );
    ASSERT_EXPR( strideWidth > 0 );
    ASSERT_EXPR( strideDepth > 0 );
    ASSERT_EXPR( source.Channels() == filter.Channels() );
    ASSERT_EXPR( filter.Height() <= source.Height() + 2 * paddingHeight );
    ASSERT_EXPR( filter.Width() <= source.Width() + 2 * paddingWidth );
    ASSERT_EXPR( filter.Depth() <= source.Depth() + 2 * paddingDepth );
    ASSERT_EXPR( filter.BatchLength() == 1 );
    ASSERT_EXPR( result.BatchLength() == source.BatchLength() );
    ASSERT_EXPR( result.BatchWidth() == source.BatchWidth() );
    ASSERT_EXPR( result.Height() == 1 + ( source.Height() + 2 * paddingHeight - filter.Height() ) / strideHeight );
    ASSERT_EXPR( result.Width() == 1 + ( source.Width() + 2 * paddingWidth - filter.Width() ) / strideWidth );
    ASSERT_EXPR( result.Depth() == 1 + ( source.Depth() + 2 * paddingDepth - filter.Depth() ) / strideDepth );
    ASSERT_EXPR( result.Channels() == filter.BatchWidth() );
    ASSERT_EXPR( IsInPlaceActivation( activation.Type ) );

	CCommon3dConvolutionDesc* desc = new CCommon3dConvolutionDesc( source, result, filter, paddingHeight, paddingWidth, paddingDepth,
		strideHeight, strideWidth, strideDepth,
		dynamic_cast<CCommonActivationDesc*>( InitActivation( activation, result.BlobSize() ) ) );
	return desc;
}

void CMetalMathEngine::Blob3dConvolution( const C3dConvolutionDesc& convDesc, const CFloatHandle& sourceData,
	const CFloatHandle& filterData, const CFloatHandle* freeTermData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( filterData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermData == 0 || freeTermData->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCommon3dConvolutionDesc& desc = static_cast<const CCommon3dConvolutionDesc&>( convDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Result;

    if( freeTermData == 0 ) {
        C3DKernel kernel( *queue, "cubeKernelBlob3DConvolution",
            1, 1, 1, result.Height(), result.Width(), result.Depth() * result.Channels() * result.ObjectCount() );
        kernel.SetParam( source, 0 );
        kernel.SetParam( sourceData, 1 );
        kernel.SetParam( desc.PaddingDepth, 2 );
        kernel.SetParam( desc.PaddingHeight, 3 );
        kernel.SetParam( desc.PaddingWidth, 4 );
        kernel.SetParam( desc.StrideDepth, 5 );
        kernel.SetParam( desc.StrideHeight, 6 );
        kernel.SetParam( desc.StrideWidth, 7 );
        kernel.SetParam( filter, 8 );
        kernel.SetParam( filterData, 9 );
        kernel.SetParam( result, 10 );
        kernel.SetParam( resultData, 11 );
        ASSERT_EXPR( kernel.Run() );
    } else {
        C3DKernel kernel( *queue, "cubeKernelBlob3DConvolutionWithFreeTerm",
            1, 1, 1, result.Height(), result.Width(), result.Depth() * result.Channels() * result.ObjectCount() );
        kernel.SetParam( source, 0 );
        kernel.SetParam( sourceData, 1 );
        kernel.SetParam( desc.PaddingDepth, 2 );
        kernel.SetParam( desc.PaddingHeight, 3 );
        kernel.SetParam( desc.PaddingWidth, 4 );
        kernel.SetParam( desc.StrideDepth, 5 );
        kernel.SetParam( desc.StrideHeight, 6 );
        kernel.SetParam( desc.StrideWidth, 7 );
        kernel.SetParam( filter, 8 );
        kernel.SetParam( filterData, 9 );
        kernel.SetParam( *freeTermData, 10 );
        kernel.SetParam( result, 11 );
        kernel.SetParam( resultData, 12 );
        ASSERT_EXPR( kernel.Run() );
    }

    Activation( *desc.Activation, resultData, resultData, desc.Result.BlobSize() );
}

void CMetalMathEngine::Blob3dConvolutionBackward( const C3dConvolutionDesc& convDesc, const CFloatHandle& outputData, const CFloatHandle& outputDiffData,
	const CFloatHandle& filterData, const CFloatHandle* freeTermData, const CFloatHandle& inputDiffData )
{
    ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
    ASSERT_EXPR( filterData.GetMathEngine() == this );
    ASSERT_EXPR( freeTermData == 0 || freeTermData->GetMathEngine() == this );
    ASSERT_EXPR( inputDiffData.GetMathEngine() == this );
    
    const CCommon3dConvolutionDesc& desc = static_cast<const CCommon3dConvolutionDesc&>( convDesc );
    const CBlobDesc& inputDiff = desc.Source;
    const CBlobDesc& filter = desc.Filter;
    const CBlobDesc& outputDiff = desc.Result;
    const int outputDiffGeometry = outputDiff.Depth() * outputDiff.Height() * outputDiff.Width();
    const int tempBufferSize = outputDiffGeometry * filter.ObjectSize();
    
	if( !outputData.IsNull() ) {
		ActivationBackward( *desc.Activation, CFloatHandle(), outputData, outputDiffData, outputDiffData, desc.Result.BlobSize() );
	}
    
    CFloatHandleStackVar temp( mathEngine(), tempBufferSize );
    
    for( int b = 0; b < outputDiff.ObjectCount(); b++ ) {
        MultiplyMatrixByMatrix( 1, outputDiffData + b * outputDiff.ObjectSize(), outputDiffGeometry,
            outputDiff.Channels(), filterData, filter.ObjectSize(), temp, tempBufferSize );
        
        if( freeTermData != 0 ) {
            SetVectorToMatrixRows( inputDiffData + b * inputDiff.ObjectSize(), inputDiff.Depth() * inputDiff.Height() * inputDiff.Width(),
                inputDiff.Channels(), *freeTermData );
        } else {
            VectorFill( inputDiffData + b * inputDiff.ObjectSize(), 0.0f, inputDiff.ObjectSize() );
        }
        
        C2DKernel kernel( *queue, "matrixKernelBlob3DConvolutionBackward",
            1, 1, inputDiff.Depth(), inputDiff.Height() );
        kernel.SetParam( outputDiff, 0 );
        kernel.SetParam( desc.PaddingDepth, 1 );
        kernel.SetParam( desc.PaddingHeight, 2 );
        kernel.SetParam( desc.PaddingWidth, 3 );
        kernel.SetParam( desc.StrideDepth, 4 );
        kernel.SetParam( desc.StrideHeight, 5 );
        kernel.SetParam( desc.StrideWidth, 6 );
        kernel.SetParam( filter, 7 );
        kernel.SetParam( temp, 8 );
        kernel.SetParam( inputDiff, 9 );
        kernel.SetParam( inputDiffData + b * inputDiff.ObjectSize(), 10 );
        ASSERT_EXPR( kernel.Run() );
    }
}

void CMetalMathEngine::Blob3dConvolutionLearnAdd( const C3dConvolutionDesc&, const CFloatHandle&, const CFloatHandle&,
	const CFloatHandle&, const CFloatHandle&, const CFloatHandle*, bool )
{
	ASSERT_EXPR( false );
}

//----------------------------------------------------------------------------------------------------------------------------------------
// Time convolution

CTimeConvolutionDesc* CMetalMathEngine::InitTimeConvolution( const CBlobDesc& source,
	int stride, int paddingFront, int paddingBack, int dilation, const CBlobDesc& filter, const CBlobDesc& result )
{
	ASSERT_EXPR( stride > 0 );
	ASSERT_EXPR( paddingFront >= 0 );
	ASSERT_EXPR( paddingBack >= 0 );
	ASSERT_EXPR( dilation > 0 );
	ASSERT_EXPR( filter.BatchLength() == 1 );
	ASSERT_EXPR( filter.Width() == 1 );
	ASSERT_EXPR( filter.Depth() == 1 );
	ASSERT_EXPR( filter.Channels() == source.ObjectSize() );
	ASSERT_EXPR( source.BatchLength() + paddingFront + paddingBack >= ( filter.Height() - 1 ) * dilation + 1 );
	ASSERT_EXPR( result.BatchLength() == ( source.BatchLength() - ( filter.Height() - 1 ) * dilation - 1 + paddingFront + paddingBack ) / stride + 1 );
	ASSERT_EXPR( result.BatchWidth() == source.BatchWidth() );
	ASSERT_EXPR( result.ListSize() == 1 && source.ListSize() == 1 );
	ASSERT_EXPR( result.Width() == 1 );
	ASSERT_EXPR( result.Height() == 1 );
	ASSERT_EXPR( result.Depth() == 1 );
	ASSERT_EXPR( result.Channels() == filter.BatchWidth() );
	ASSERT_EXPR( paddingFront < ( filter.Height() - 1 ) * dilation + 1 );
	ASSERT_EXPR( paddingBack < ( filter.Height() - 1 ) * dilation + 1 );

	CCommonTimeConvolutionDesc* desc = new CCommonTimeConvolutionDesc( source, filter, result, stride, paddingFront, paddingBack, dilation );
	return desc;
}

static const int BlobTimeConvolutionPrepareCombine = 16;

void CMetalMathEngine::BlobTimeConvolution( const CTimeConvolutionDesc& convDesc,
	const CFloatHandle& sourceData, const CFloatHandle& filterData, const CFloatHandle& freeTermData,
	const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( filterData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCommonTimeConvolutionDesc& desc = static_cast<const CCommonTimeConvolutionDesc&>( convDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Result;

	int workspaceSize = 0;

	bool useTempMatrix = desc.Stride > 1 || filter.Height() > 1;

	if( useTempMatrix ) {
		// Create a temporary matrix
		workspaceSize = result.BatchLength() * source.BatchWidth() * filter.Height() * source.ObjectSize();
	}

	CFloatHandleStackVar buffer( mathEngine(), workspaceSize );
	CConstFloatHandle sourceDataPtr;
	if( useTempMatrix ) {
        C3DKernel kernel( *queue, "cubeKernelBlobTimeConvolutionPrepare",
            1, 1, BlobTimeConvolutionPrepareCombine, filter.Height(), result.BatchLength(), source.BatchWidth() * source.ObjectSize() );
        kernel.SetParam( source, 0 );
        kernel.SetParam( sourceData, 1 );
        kernel.SetParam( desc.Stride, 2 );
        kernel.SetParam( desc.PaddingFront, 3 );
        kernel.SetParam( desc.Dilation, 4 );
        kernel.SetParam( filter, 5 );
        kernel.SetParam( filterData, 6 );
        kernel.SetParam( result, 7 );
        kernel.SetParam( resultData, 8 );
        kernel.SetParam( buffer.GetHandle(), 9 );
        ASSERT_EXPR( kernel.Run() );

		sourceDataPtr = buffer.GetHandle();
	} else {
		sourceDataPtr = sourceData;
	}

	// Convolution
	MultiplyMatrixByTransposedMatrix( sourceDataPtr,
		result.BatchLength() * source.BatchWidth(), filter.Height() * source.ObjectSize(), filter.Height() * source.ObjectSize(),
		filterData, filter.ObjectCount(), filter.Height() * source.ObjectSize(),
		resultData, filter.ObjectCount(), result.BlobSize());

	// Free term
	AddVectorToMatrixRows( 1, resultData, resultData, result.ObjectCount(), result.ObjectSize(), freeTermData );
}

void CMetalMathEngine::BlobTimeConvolutionBackward( const CTimeConvolutionDesc&, const CFloatHandle&,
	const CFloatHandle&, const CFloatHandle&, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::BlobTimeConvolutionLearnAdd( const CTimeConvolutionDesc&, const CFloatHandle&,
	const CFloatHandle&, const CFloatHandle&, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

//----------------------------------------------------------------------------------------------------------------------------------------
// Channelwise convolution

CChannelwiseConvolutionDesc* CMetalMathEngine::InitBlobChannelwiseConvolution( const CBlobDesc& source,
	int paddingHeight, int paddingWidth, int strideHeight, int strideWidth, 
	const CBlobDesc& filter, const CBlobDesc* freeTerm, const CBlobDesc& result, const CActivationInfo& activation )
{
	ASSERT_EXPR( source.Depth() == 1 );
	ASSERT_EXPR( filter.Height() > paddingHeight );
	ASSERT_EXPR( filter.Height() <= source.Height() + 2 * paddingHeight );
	ASSERT_EXPR( filter.Width() > paddingWidth );
	ASSERT_EXPR( filter.Width() <= source.Width() + 2 * paddingWidth );
	ASSERT_EXPR( filter.ObjectCount() == 1 );
	ASSERT_EXPR( filter.Channels() == source.Channels() );
	ASSERT_EXPR( freeTerm == 0 || freeTerm->BlobSize() == filter.Channels() );
	ASSERT_EXPR( result.BatchLength() == source.BatchLength() );
	ASSERT_EXPR( result.BatchWidth() == source.BatchWidth() );
	ASSERT_EXPR( result.ListSize() == source.ListSize() );
	ASSERT_EXPR( result.Depth() == 1 );
	ASSERT_EXPR( result.Channels() == source.Channels() );
	const int expectedOutputHeight = ( source.Height() - filter.Height() + 2 * paddingHeight ) / strideHeight + 1;
	const int expectedOutputWidth = ( source.Width() - filter.Width() + 2 * paddingWidth ) / strideWidth + 1;
	ASSERT_EXPR( result.Height() == expectedOutputHeight );
	ASSERT_EXPR( result.Width() == expectedOutputWidth );
    ASSERT_EXPR( IsInPlaceActivation( activation.Type ) );

	CCommonChannelwiseConvolutionDesc* desc = new CCommonChannelwiseConvolutionDesc( paddingHeight, paddingWidth, 
		strideHeight, strideWidth, source, filter, result,
        dynamic_cast<CCommonActivationDesc*>( InitActivation( activation, result.BlobSize() ) ) );
	return desc;
}

void CMetalMathEngine::BlobChannelwiseConvolution( const CChannelwiseConvolutionDesc& convDesc, const CConstFloatHandle& sourceData,
	const CConstFloatHandle& filterData, const CConstFloatHandle* freeTermData, const CFloatHandle& resultData )
{
    ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( filterData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermData == 0 || freeTermData->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCommonChannelwiseConvolutionDesc& desc = static_cast<const CCommonChannelwiseConvolutionDesc&>( convDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Result;

    if( filter.Width() == 3 && filter.Height() == 3 ) {
        if( desc.StrideHeight == 1 && desc.StrideWidth == 1 ) {
            const int combineH = 3;
            const int combineW = 4;
            const int channelBlocksCount = ( result.Height() + combineH - 1 ) / combineH * ( result.Width() + combineW - 1 ) / combineW;
            if( freeTermData == 0 ) {
                C3DKernel kernelConvolution( *queue, "cubeKernelBlobChannelwiseConvolution3x3",
                    1, 1, 1, result.ObjectCount(), channelBlocksCount, result.Channels() );
                kernelConvolution.SetParam( source, 0 );
                kernelConvolution.SetParam( sourceData, 1 );
                kernelConvolution.SetParam( filter, 2 );
                kernelConvolution.SetParam( filterData, 3 );
                kernelConvolution.SetParam( desc.PaddingHeight, 4 );
                kernelConvolution.SetParam( desc.PaddingWidth, 5 );
                kernelConvolution.SetParam( result, 6 );
                kernelConvolution.SetParam( resultData, 7 );
                ASSERT_EXPR( kernelConvolution.Run() );
            } else {
                C3DKernel kernelConvolution( *queue, "cubeKernelBlobChannelwiseConvolution3x3WithFreeTerm", 1, 1, 1,
					result.ObjectCount(), channelBlocksCount, result.Channels() );
                kernelConvolution.SetParam( source, 0 );
                kernelConvolution.SetParam( sourceData, 1 );
                kernelConvolution.SetParam( filter, 2 );
                kernelConvolution.SetParam( filterData, 3 );
                kernelConvolution.SetParam( desc.PaddingHeight, 4 );
                kernelConvolution.SetParam( desc.PaddingWidth, 5 );
                kernelConvolution.SetParam( *freeTermData, 6 );
                kernelConvolution.SetParam( result, 7 );
                kernelConvolution.SetParam( resultData, 8 );
                ASSERT_EXPR( kernelConvolution.Run() );
            }
            Activation( *desc.Activation, resultData, resultData, desc.Result.BlobSize() );
            return;
        } else if( desc.StrideHeight == 2 && desc.StrideWidth == 2 ) {
            const int combineH = 2;
            const int combineW = 2;
            const int channelBlocksCount = ( result.Height() + combineH - 1 ) / combineH * ( result.Width() + combineW - 1 ) / combineW;
            if( freeTermData == 0 ) {
                C3DKernel kernelConvolution( *queue, "cubeKernelBlobChannelwiseConvolution3x3Stride2x2",
					1, 1, 1, result.ObjectCount(), channelBlocksCount, result.Channels() );
                kernelConvolution.SetParam( source, 0 );
                kernelConvolution.SetParam( sourceData, 1 );
                kernelConvolution.SetParam( filter, 2 );
                kernelConvolution.SetParam( filterData, 3 );
                kernelConvolution.SetParam( desc.PaddingHeight, 4 );
                kernelConvolution.SetParam( desc.PaddingWidth, 5 );
                kernelConvolution.SetParam( result, 6 );
                kernelConvolution.SetParam( resultData, 7 );
                ASSERT_EXPR( kernelConvolution.Run() );
            } else {
                C3DKernel kernelConvolution( *queue, "cubeKernelBlobChannelwiseConvolution3x3Stride2x2WithFreeTerm",
					1, 1, 1, result.ObjectCount(), channelBlocksCount, result.Channels() );
                kernelConvolution.SetParam( source, 0 );
                kernelConvolution.SetParam( sourceData, 1 );
                kernelConvolution.SetParam( filter, 2 );
                kernelConvolution.SetParam( filterData, 3 );
                kernelConvolution.SetParam( desc.PaddingHeight, 4 );
                kernelConvolution.SetParam( desc.PaddingWidth, 5 );
                kernelConvolution.SetParam( *freeTermData, 6 );
                kernelConvolution.SetParam( result, 7 );
                kernelConvolution.SetParam( resultData, 8 );
                ASSERT_EXPR( kernelConvolution.Run() );
            }
            Activation( *desc.Activation, resultData, resultData, desc.Result.BlobSize() );
            return;
        }
    } 

    if( freeTermData == 0 ) {
        C3DKernel kernelConvolution( *queue, "cubeKernelBlobChannelwiseConvolutionBase", 1, 1, 1,
			result.ObjectCount(), result.Height() * result.Width(), result.Channels() );
        kernelConvolution.SetParam( source, 0 );
        kernelConvolution.SetParam( sourceData, 1 );
        kernelConvolution.SetParam( filter, 2 );
        kernelConvolution.SetParam( filterData, 3 );
        kernelConvolution.SetParam( desc.PaddingHeight, 4 );
        kernelConvolution.SetParam( desc.PaddingWidth, 5 );
        kernelConvolution.SetParam( desc.StrideHeight, 6 );
        kernelConvolution.SetParam( desc.StrideWidth, 7 );
        kernelConvolution.SetParam( result, 8 );
        kernelConvolution.SetParam( resultData, 9 );
        ASSERT_EXPR( kernelConvolution.Run() );
    } else {
        C3DKernel kernelConvolution( *queue, "cubeKernelBlobChannelwiseConvolutionBaseWithFreeTerm",
			1, 1, 1, result.ObjectCount(), result.Height() * result.Width(), result.Channels() );
        kernelConvolution.SetParam( source, 0 );
        kernelConvolution.SetParam( sourceData, 1 );
        kernelConvolution.SetParam( filter, 2 );
        kernelConvolution.SetParam( filterData, 3 );
        kernelConvolution.SetParam( desc.PaddingHeight, 4 );
        kernelConvolution.SetParam( desc.PaddingWidth, 5 );
        kernelConvolution.SetParam( desc.StrideHeight, 6 );
        kernelConvolution.SetParam( desc.StrideWidth, 7 );
        kernelConvolution.SetParam( *freeTermData, 8 );
        kernelConvolution.SetParam( result, 9 );
        kernelConvolution.SetParam( resultData, 10 );
        ASSERT_EXPR( kernelConvolution.Run() );
    }

    Activation( *desc.Activation, resultData, resultData, desc.Result.BlobSize() );
}

void CMetalMathEngine::BlobChannelwiseConvolutionBackward( const CChannelwiseConvolutionDesc&, const CFloatHandle&,
    const CFloatHandle&, const CFloatHandle&, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::BlobChannelwiseConvolutionLearnAdd( const CChannelwiseConvolutionDesc&, const CFloatHandle&,
	const CFloatHandle&, const CFloatHandle&, const CFloatHandle&, const CFloatHandle* )
{
	ASSERT_EXPR( false );
}

//------------------------------------------------------------------------------------------------------------
// RLE convolution

// RLE convolution descriptor
struct CMetalRleConvolutionDesc : public CRleConvolutionDesc {
	float StrokeValue;
	float NonStrokeValue;

	CConvolutionDesc* ConvDesc;

	CMetalRleConvolutionDesc() : ConvDesc( 0 ) {}
	virtual ~CMetalRleConvolutionDesc();
};

CMetalRleConvolutionDesc::~CMetalRleConvolutionDesc()
{
	if( ConvDesc != 0 ) {
		delete ConvDesc;
	}
}

CRleConvolutionDesc* CMetalMathEngine::InitBlobRleConvolution( const CBlobDesc& source, float strokeValue,
	float nonStrokeValue, int strideHeight, int strideWidth, const CBlobDesc& filter,
	const CBlobDesc& result )
{
	ASSERT_EXPR( strideHeight > 0 );
	ASSERT_EXPR( strideWidth > 0 );
	ASSERT_EXPR( source.Channels() == filter.Channels() );
	ASSERT_EXPR( source.Depth() == filter.Depth() );
	ASSERT_EXPR( filter.Height() <= source.Height() );
	ASSERT_EXPR( filter.Width() <= source.Width() );
	ASSERT_EXPR( filter.BatchLength() == 1 );
	ASSERT_EXPR( result.BatchLength() == source.BatchLength() );
	ASSERT_EXPR( result.BatchWidth() == source.BatchWidth() );
	ASSERT_EXPR( result.Height() == 1 + ( source.Height() - filter.Height() ) / strideHeight );
	ASSERT_EXPR( result.Width() == 1 + ( source.Width() - filter.Width() ) / strideWidth );
	ASSERT_EXPR( result.Channels() == filter.BatchWidth() );
	ASSERT_EXPR( result.Depth() == 1 );
	ASSERT_EXPR( filter.Width() <= MaxRleConvFilterWidth );
	ASSERT_EXPR( source.Width() <= MaxRleConvImageWidth );
	ASSERT_EXPR( source.Channels() == 1 );
	ASSERT_EXPR( ( filter.ObjectCount() % 4 ) == 0 );

	CMetalRleConvolutionDesc* desc = new CMetalRleConvolutionDesc();
	desc->StrokeValue = strokeValue;
	desc->NonStrokeValue = nonStrokeValue;
	desc->ConvDesc = InitBlobConvolution( source, 0, 0, strideHeight, strideWidth, 1, 1, filter, result, AF_None );
	return desc;
}

void CMetalMathEngine::BlobRleConvolution( const CRleConvolutionDesc& desc, const CFloatHandle& sourceData,
	const CFloatHandle& filterData, const CFloatHandle* freeTermData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( filterData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermData == 0 || freeTermData->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CMetalRleConvolutionDesc& rleDesc = static_cast<const CMetalRleConvolutionDesc&>( desc );
	const CCommonConvolutionDesc* convDesc = static_cast<const CCommonConvolutionDesc*>( rleDesc.ConvDesc );

	CFloatHandleVar inputConverted( mathEngine(), convDesc->Source.BlobSize() );
	blobConvertFromRle( rleDesc, sourceData, inputConverted );
	BlobConvolution( *(rleDesc.ConvDesc), inputConverted, filterData, freeTermData, resultData );
}

void CMetalMathEngine::blobConvertFromRle( const CMetalRleConvolutionDesc& desc, const CFloatHandle& sourceData, const CFloatHandle& resultData )
{
    const CCommonConvolutionDesc* convDesc = static_cast<const CCommonConvolutionDesc*>( desc.ConvDesc );
	const CBlobDesc& source = convDesc->Source;

	C2DKernel kernel( *queue, "matrixKernelBlobConvertFromRle", 1, 1, source.ObjectCount(), source.Height() );
	kernel.SetParam( source, 0 );
	kernel.SetParam( sourceData, 1 );
	const int sourceObjectSize = source.ObjectSize() * sizeof(float);
	kernel.SetParam( sourceObjectSize, 2 );
	kernel.SetParam( desc.StrokeValue, 3 );
	kernel.SetParam( desc.NonStrokeValue, 4 );
	kernel.SetParam( source, 5 );
	kernel.SetParam( resultData, 6 );
	kernel.Run();
}

void CMetalMathEngine::BlobRleConvolutionLearnAdd( const CRleConvolutionDesc&, const CFloatHandle&, const CFloatHandle&,
	const CFloatHandle&, const CFloatHandle* )
{
	ASSERT_EXPR( false );
}

} // namespace NeoML

#endif // NEOML_USE_METAL
