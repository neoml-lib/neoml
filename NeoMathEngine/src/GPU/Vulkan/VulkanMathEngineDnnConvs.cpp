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

#include <NeoMathEngine/CrtAllocatedObject.h>
#include <VulkanMathEngine.h>
#include <VulkanShader.h>
#include <MathEngineCommon.h>
#include <VulkanDll.h>
#include <MathEngineDnnConv.h>

namespace NeoML {

// Include the shader code
#include <shaders/generated/BlobConvolution3x3s1d1.h>
#include <shaders/generated/PrepareBlobWithPaddingAdreno.h>
#include <shaders/generated/BlobConvolution3x3s1d1Adreno.h>
#include <shaders/generated/PrepareBlobForConvolution.h>
#include <shaders/generated/PrepareBlobForConvolutionAdreno.h>
#include <shaders/generated/PrepareFilter3x3ForConvolutionAdreno.h>
#include <shaders/generated/PrepareBlobWithPaddingBuffers.h>
#include <shaders/generated/BlobConvolution.h>
#include <shaders/generated/BlobConvolutionAdreno.h>
#include <shaders/generated/BlobConvolution8.h>
#include <shaders/generated/BlobConvolution8Adreno.h>
#include <shaders/generated/PrepareFilterForConvolutionBackwardAdreno.h>
#include <shaders/generated/BlobConvolutionBackwardAdreno.h>
#include <shaders/generated/BlobConvolutionBackward.h>
#include <shaders/generated/BlobChannelwiseConvolutionAdreno.h>
#include <shaders/generated/BlobChannelwiseConvolution.h>
#include <shaders/generated/BlobChannelwiseConvolution3x3s1.h>
#include <shaders/generated/BlobChannelwiseConvolution3x3s2.h>
#include <shaders/generated/Blob3dConvolution.h>
#include <shaders/generated/Blob3dConvolutionBackward.h>
#include <shaders/generated/BlobConvertFromRLE.h>
#include <shaders/generated/BlobTimeConvolutionPrepare.h>

//------------------------------------------------------------------------------------------------------------
// RLE convolution

// RLE convolution descriptor
struct CVulkanRleConvolutionDesc : public CRleConvolutionDesc {
	float StrokeValue;
	float NonStrokeValue;
	CConvolutionDesc* ConvDesc{};

	CVulkanRleConvolutionDesc() = default;
	~CVulkanRleConvolutionDesc() override;
};

CVulkanRleConvolutionDesc::~CVulkanRleConvolutionDesc()
{
	if( ConvDesc != 0 ) {
		delete ConvDesc;
	}
}

CRleConvolutionDesc* CVulkanMathEngine::InitBlobRleConvolution( const CBlobDesc& source, float strokeValue,
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

	CVulkanRleConvolutionDesc* desc = new CVulkanRleConvolutionDesc();
	desc->StrokeValue = strokeValue;
	desc->NonStrokeValue = nonStrokeValue;
	desc->ConvDesc = InitBlobConvolution( source, 0, 0, strideHeight, strideWidth, 1, 1, filter, result );
	return desc;
}

void CVulkanMathEngine::blobConvertFromRleCommon( const CVulkanRleConvolutionDesc& desc, const CConstFloatHandle& sourceData,
	const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCommonConvolutionDesc* convDesc = static_cast<const CCommonConvolutionDesc*>( desc.ConvDesc );
	const CBlobDesc& source = convDesc->Source;
	
	CMemoryHandle bufs[2] = { sourceData, resultData };
	size_t sizes[2] = { source.BlobSize() * sizeof(float), source.BlobSize() * sizeof(float) };

	PARAM_STRUCT( BlobConvertFromRLE ) param = { 
		source.ObjectCount(),
		source.Height(),
		source.Width(),
		source.ObjectSize(),
		desc.StrokeValue,
		desc.NonStrokeValue
	};

	runShader( shaderLoader->GET_SHADER_DATA( BlobConvertFromRLE, true, 0, 0, 2),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2,
		source.Height(), source.ObjectCount(), 1 );
}

void CVulkanMathEngine::BlobRleConvolution( const CRleConvolutionDesc& desc, const CConstFloatHandle& sourceData,
	const CConstFloatHandle& filterData, const CConstFloatHandle* freeTermData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( filterData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermData == 0 || freeTermData->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CVulkanRleConvolutionDesc& rleDesc = static_cast<const CVulkanRleConvolutionDesc&>( desc );
	const CCommonConvolutionDesc* convDesc = static_cast<const CCommonConvolutionDesc*>( rleDesc.ConvDesc );

	CFloatHandleStackVar inputConverted( mathEngine(), convDesc->Source.BlobSize() );
	blobConvertFromRleCommon( rleDesc, sourceData, inputConverted );
	BlobConvolution( *(rleDesc.ConvDesc), inputConverted, filterData, freeTermData, resultData );
}

void CVulkanMathEngine::BlobRleConvolutionLearnAdd( const CRleConvolutionDesc&, const CConstFloatHandle&,
	const CConstFloatHandle&, const CFloatHandle&, const CFloatHandle* )
{
	ASSERT_EXPR( false );
}

//------------------------------------------------------------------------------------------------------------
// Time convolution

CTimeConvolutionDesc* CVulkanMathEngine::InitTimeConvolution( const CBlobDesc& source,
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

	CCommonTimeConvolutionDesc* desc = new CCommonTimeConvolutionDesc(
		source, result, filter, stride, paddingFront, paddingBack, dilation );
	ASSERT_EXPR( ( desc->Result.BatchLength() * desc->Source.BatchWidth() ) * desc->Filter.ObjectCount() <= desc->Result.BlobSize() );
	return desc;
}

void CVulkanMathEngine::BlobTimeConvolution( const CTimeConvolutionDesc& convDesc, const CConstFloatHandle& sourceData,
	const CConstFloatHandle& filterData, const CConstFloatHandle& freeTermData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( filterData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCommonTimeConvolutionDesc& desc = static_cast<const CCommonTimeConvolutionDesc&>( convDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Result;
	const bool useTempMatrix = desc.Stride > 1 || filter.Height() > 1;

	int workspaceSize = 0;
	if( useTempMatrix ) {
		// Create a temporary matrix
		workspaceSize = result.BatchLength() * source.BatchWidth() * filter.Height() * source.ObjectSize();
	}

	CFloatHandleStackVar buffer( mathEngine(), workspaceSize );
	CConstFloatHandle sourceDataPtr;
	if( useTempMatrix ) {
        CMemoryHandle bufs[2] = { sourceData, buffer.GetHandle() };
		size_t sizes[2] = { source.BlobSize() * sizeof(float), workspaceSize * sizeof(float) };

		PARAM_STRUCT(BlobTimeConvolutionPrepare) param = { source.BatchLength(), source.BatchWidth(), source.ObjectSize(), 
			result.BatchLength(), result.BatchWidth(), filter.Height(), desc.Stride, desc.PaddingFront, desc.Dilation };

		runShader( shaderLoader->GET_SHADER_DATA(BlobTimeConvolutionPrepare, true, 0, 0, 2),
			&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2,
			filter.Height(), result.BatchLength(), Ceil( source.BatchWidth() * source.ObjectSize(), 16 ) );

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

void CVulkanMathEngine::BlobTimeConvolutionBackward( const CTimeConvolutionDesc&, const CConstFloatHandle&,
	const CConstFloatHandle&, const CConstFloatHandle&, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::BlobTimeConvolutionLearnAdd( const CTimeConvolutionDesc&, const CConstFloatHandle&,
	const CConstFloatHandle&, const CFloatHandle&, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

//------------------------------------------------------------------------------------------------------------
// 3d convolution

C3dConvolutionDesc* CVulkanMathEngine::InitBlob3dConvolution( const CBlobDesc& source,
	int paddingHeight, int paddingWidth, int paddingDepth,
	int strideHeight, int strideWidth, int strideDepth,
	const CBlobDesc& filter, const CBlobDesc& result )
{
	CCommon3dConvolutionDesc* desc = new CCommon3dConvolutionDesc(
		source, result, filter, paddingHeight, paddingWidth, paddingDepth,
		strideHeight, strideWidth, strideDepth );
	return desc;
}

void CVulkanMathEngine::Blob3dConvolution( const C3dConvolutionDesc& convDesc, const CConstFloatHandle& sourceData,
	const CConstFloatHandle& filterData, const CConstFloatHandle* freeTerm, const CFloatHandle& resultData )
{
	const CCommon3dConvolutionDesc& desc = static_cast<const CCommon3dConvolutionDesc&>( convDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Result;

	CMemoryHandle bufs[4] = { sourceData, filterData, freeTerm != nullptr ? *freeTerm : filterData, resultData };
	size_t sizes[4] = { source.BlobSize() * sizeof(float), filter.BlobSize() * sizeof(float), 
		freeTerm != nullptr ? filter.ObjectCount() * sizeof(float): sizeof(float), result.BlobSize() * sizeof(float) };

	PARAM_STRUCT(Blob3dConvolution) param = { 
		desc.PaddingWidth, desc.PaddingHeight, desc.PaddingDepth,
		desc.StrideWidth, desc.StrideHeight, desc.StrideDepth,
		(freeTerm != nullptr) ? 1 : 0,
		source.Channels(), desc.Source.Height(), source.Width(), source.Depth(),
		filter.Height(), filter.Width(), filter.Depth(), filter.ObjectCount(),
		result.Height(), result.Width(), result.Depth(), result.ObjectCount() };

	runShader( shaderLoader->GET_SHADER_DATA(Blob3dConvolution, true, 0, 0, 4),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 4,
		result.Width(), result.Height(), result.Channels() * result.ObjectCount() * result.Depth() );
}

void CVulkanMathEngine::Blob3dConvolutionBackward( const C3dConvolutionDesc& convDesc, const CConstFloatHandle& sourceData,
	const CConstFloatHandle& filterData, const CConstFloatHandle* freeTerm, const CFloatHandle& resultData )
{   
	const CCommon3dConvolutionDesc& desc = static_cast<const CCommon3dConvolutionDesc&>( convDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Result;

    const int sourceGeometry = source.Depth() * source.Height() * source.Width();
    const int tempBufferSize = sourceGeometry * filter.ObjectSize();
    CFloatHandleStackVar temp( mathEngine(), tempBufferSize );
    
    for( int b = 0; b < source.ObjectCount(); b++ ) {
        MultiplyMatrixByMatrix( 1, sourceData + b * source.ObjectSize(), sourceGeometry, source.Channels() * source.Depth(),
        	filterData, filter.ObjectSize(), temp, tempBufferSize );

        if( freeTerm != 0 ) {
        	SetVectorToMatrixRows( resultData + b * result.ObjectSize(), 
        		result.Depth() * result.Height() * result.Width(), result.Channels(), *freeTerm );
        } else {
            VectorFill( resultData + b * result.ObjectSize(), 0.0f, result.ObjectSize() );
        }

        CMemoryHandle bufs[3] = { sourceData, temp.GetHandle(), resultData + b * result.ObjectSize() };
		size_t sizes[3] = { source.BlobSize() * sizeof(float), tempBufferSize * sizeof(float), result.ObjectSize() * sizeof(float) };

		PARAM_STRUCT(Blob3dConvolutionBackward) param = { 
			desc.PaddingWidth, desc.PaddingHeight, desc.PaddingDepth,
			desc.StrideWidth, desc.StrideHeight, desc.StrideDepth,
			result.Channels(), result.Height(), result.Width(), result.Depth(),
			filter.Height(), filter.Width(), filter.Depth(),
			source.Height(), source.Width(), source.Depth(),
		};

		runShader( shaderLoader->GET_SHADER_DATA(Blob3dConvolutionBackward, true, 0, 0, 3),
			&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 3,
			result.Depth(), result.Height(), 1 );
    }
}

void CVulkanMathEngine::Blob3dConvolutionLearnAdd( const C3dConvolutionDesc&, const CConstFloatHandle&,
	const CConstFloatHandle&, const CFloatHandle&, const CFloatHandle*, bool )
{
	ASSERT_EXPR( false );
}

//------------------------------------------------------------------------------------------------------------
// Convolution

CConvolutionDesc* CVulkanMathEngine::InitBlobConvolution( const CBlobDesc& source, int paddingHeight, int paddingWidth,
	int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, const CBlobDesc& filter, const CBlobDesc& result )
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

	CCommonConvolutionDesc* desc = new CCommonConvolutionDesc( source, result, filter, paddingHeight, paddingWidth,
		strideHeight, strideWidth, dilationHeight, dilationWidth );
	return desc;
}

void CVulkanMathEngine::BlobConvolution( const CConvolutionDesc& convDesc,
	const CConstFloatHandle& sourceData, const CConstFloatHandle& filterData, const CConstFloatHandle* freeTermData,
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

	// Special cases
	if( filter.Width() == 1 && filter.Height() == 1 && desc.StrideHeight == 1 && desc.StrideWidth == 1 ) {
		blobConvolution1x1s1Common( desc, sourceData, filterData, freeTermData, resultData );
		return;
	} else if( filter.Width() == 3 && filter.Height() == 3
		&& desc.StrideHeight == 1 && desc.StrideWidth == 1
		&& desc.DilationHeight == 1 && desc.DilationWidth == 1 )
	{
		if( device->Type == VDT_Adreno ) {
			blobConvolution3x3s1d1Adreno( desc, sourceData, filterData, freeTermData, resultData );
			return;
		} else {
			blobConvolution3x3s1d1( desc, sourceData, filterData, freeTermData, resultData );
			return;
		}
	}

	int totalChannels = result.Depth() * result.Channels();
	int channels8 = totalChannels / 8;

	if( device->Type == VDT_Adreno ) {
		int sourceChannelGroupSize = 0;
		prepareBlobForConvolutionAdreno( source, sourceData, TVI_ConvSource, sourceChannelGroupSize );
		int filterChannelGroupSize = 0;
		prepareBlobForConvolutionAdreno( filter, filterData, TVI_ConvFilter, filterChannelGroupSize );

		if( freeTermData != 0 ) {
			batchVectorToImage( 1, *freeTermData, filter.ObjectCount(), TVI_FreeTerm );
		}

		if( channels8 > 0 ) {
			blobConvolutionImpl8Adreno( desc, sourceData, filterData, freeTermData != nullptr, resultData, channels8,
				sourceChannelGroupSize, filterChannelGroupSize );
		}

		if( ( totalChannels - channels8 * 8 ) != 0 ) {
			blobConvolutionImpl1Adreno( desc, sourceData, filterData, freeTermData != nullptr, resultData, channels8 * 8,
				sourceChannelGroupSize, filterChannelGroupSize );
		}
	} else {
		int tempFilterSize = filter.Width() * filter.ObjectCount() * filter.Height() * Ceil( filter.Channels() * filter.Depth(), 4 ) * 4;
		CFloatHandleStackVar tempFilter( mathEngine(), tempFilterSize );
		prepareBlobForConvolution( filter, filterData, tempFilter );

		int tempSourceSize = source.Width() * source.ObjectCount() * source.Height() * Ceil( source.Channels() * source.Depth(), 4 ) * 4;
		CFloatHandleStackVar tempSource( mathEngine(), tempSourceSize );
		prepareBlobForConvolution( source, sourceData, tempSource );

		if( channels8 > 0 ) {
			blobConvolutionImpl8( desc, tempSource, tempFilter, freeTermData, resultData, totalChannels );
		}

		if( ( totalChannels - channels8 * 8 ) != 0 ) {
			blobConvolutionImpl1( desc, tempSource, tempFilter, freeTermData, resultData, channels8 * 8, totalChannels );
		}
	}
}

void CVulkanMathEngine::BlobConvolutionBackward( const CConvolutionDesc& convDesc, const CConstFloatHandle& outputDiffData,
	const CConstFloatHandle& filterData, const CConstFloatHandle* freeTermData, const CFloatHandle& inputDiffData )
{
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( filterData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermData == 0 || freeTermData->GetMathEngine() == this );
	ASSERT_EXPR( inputDiffData.GetMathEngine() == this );

	const CCommonConvolutionDesc& desc = static_cast<const CCommonConvolutionDesc&>( convDesc );
	const CBlobDesc& inputDiff = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& outputDiff = desc.Result;

	int inputChannels = inputDiff.Depth() * inputDiff.Channels();
	int outputChannels4 = Ceil(filter.ObjectCount(), 4);

	if( device->Type == VDT_Adreno ) {
		int outputDiffChannelGroupSize = 0;
		const CVulkanImage& imageOutputDiff =
			prepareBlobForConvolutionAdreno( outputDiff, outputDiffData, TVI_ConvSource, outputDiffChannelGroupSize );
		const CVulkanImage& imageFilter = blobConvolutionBackwardPrepareFilterAdreno( filter, filterData, TVI_ConvFilter );
		const CVulkanImage* imageFreeTerm = &imageFilter;
		if( freeTermData != 0 ) {
			imageFreeTerm = &batchVectorToImage( 1, *freeTermData, inputChannels, TVI_FreeTerm );
		}

		const CVulkanImage* images[] = { &imageOutputDiff, &imageFilter, imageFreeTerm };

		CMemoryHandle bufs[1] = { inputDiffData };
		size_t sizes[1] = { inputDiff.BlobSize() * sizeof( float ) };

		PARAM_STRUCT( BlobConvolutionBackwardAdreno ) param = { { desc.PaddingWidth, desc.PaddingHeight },
			{ desc.StrideWidth, desc.StrideHeight }, { desc.DilationWidth, desc.DilationHeight }, ( freeTermData != 0 ) ? 1 : 0,
			outputDiff.Width(), outputDiff.Height(), outputDiff.ObjectCount(), inputDiff.Width(), inputDiff.Height(),
			inputChannels, filter.Width(), filter.Height(), outputChannels4, 0, outputDiffChannelGroupSize };

		runShader( shaderLoader->GET_SHADER_DATA( BlobConvolutionBackwardAdreno, true, 0, 3, 1 ),
			&param, sizeof( param ), 0, 0, images, 3, bufs, sizes, 1,
			inputDiff.Width(), inputDiff.ObjectCount() * inputDiff.Height(), inputChannels );
	} else {
		CMemoryHandle bufs[4] = { outputDiffData, filterData, (freeTermData == 0) ? filterData : *freeTermData, inputDiffData };
		size_t sizes[4] = { outputDiff.BlobSize() * sizeof(float), filter.BlobSize() * sizeof(float),
			inputChannels * sizeof( float ), inputDiff.BlobSize() * sizeof( float ) };

		PARAM_STRUCT( BlobConvolutionBackward ) param = { { desc.PaddingWidth, desc.PaddingHeight },
			{ desc.StrideWidth, desc.StrideHeight }, { desc.DilationWidth, desc.DilationHeight }, ( freeTermData != 0 ) ? 1 : 0,
			outputDiff.Width(), outputDiff.Height(), outputDiff.ObjectCount(), inputDiff.Width(), inputDiff.Height(),
			inputChannels, filter.Width(), filter.Height(), filter.ObjectCount(), 0 };

		runShader( shaderLoader->GET_SHADER_DATA( BlobConvolutionBackward, true, 0, 0, 4 ),
			&param, sizeof( param ), 0, 0, 0, 0, bufs, sizes, 4,
			inputDiff.Width(), inputDiff.ObjectCount() * inputDiff.Height(), inputChannels );
	}
}

void CVulkanMathEngine::BlobConvolutionLearnAdd( const CConvolutionDesc&, const CConstFloatHandle&, const CConstFloatHandle&,
	const CFloatHandle&, const CFloatHandle*, bool )
{
	ASSERT_EXPR( false );
}

// Implements convolution 1x1 with stride 1
void CVulkanMathEngine::blobConvolution1x1s1Common( const CCommonConvolutionDesc& desc,
	const CConstFloatHandle& sourceData, const CConstFloatHandle& filterData, const CConstFloatHandle* freeTermData,
	const CFloatHandle& resultData )
{
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Result;

	int channels = source.Depth() * source.Channels();
	int resultChannels = filter.ObjectCount();

	if( freeTermData == 0 ) {
		MultiplyMatrixByTransposedMatrix( sourceData, source.BlobSize() / channels, channels, channels,
			filterData, resultChannels, channels, resultData, resultChannels, result.BlobSize() );
		return;
	}

	if( device->Type != VDT_Adreno ) {
		blobConvolution1x1s1( 1, *freeTermData, sourceData, source.BlobSize() / channels, channels, channels,
			filterData, resultChannels, channels, resultData, resultChannels, result.BlobSize() );
	} else {
		batchMultiplyMatrixByMatrixAdreno( false, 1, sourceData,
			source.BlobSize() / channels, channels, channels, false, filterData, resultChannels, channels,
			channels, true, resultData, resultChannels, result.BlobSize() );
		addVectorToMatrixRowsAdreno( 1, resultData, resultData, result.BlobSize() / resultChannels,
			resultChannels, *freeTermData );
	}
}

void CVulkanMathEngine::prepareBlobForConvolution( const CBlobDesc& blob, const CConstFloatHandle& blobData, CFloatHandleStackVar& result )
{
	ASSERT_EXPR( !device->IsImageBased );

	int totalChannels = blob.Depth() * blob.Channels();
	int channels4 = Ceil( totalChannels, 4 );

	CMemoryHandle bufs[2] = { blobData, result.GetHandle() };
	size_t sizes[2] = { blob.BlobSize() * sizeof( float ), result.Size() * sizeof( float ) };

	PARAM_STRUCT( PrepareBlobForConvolution ) param =
		{ { blob.Width(), blob.Height() }, blob.ObjectCount(), totalChannels, channels4 };

	runShader( shaderLoader->GET_SHADER_DATA( PrepareBlobForConvolution, true, 0, 0, 2 ),
		&param, sizeof( param ), 0, 0, 0, 0, bufs, sizes, 2,
		blob.Width() * blob.ObjectCount(), blob.Height() * channels4, 1 );
}

const CVulkanImage& CVulkanMathEngine::prepareBlobForConvolutionAdreno( const CBlobDesc& blob,
	const CConstFloatHandle& blobData, int imageId, int& channelGroupSize )
{
	ASSERT_EXPR( device->Type == VDT_Adreno );
	ASSERT_EXPR( device->IsImageBased );

	int totalChannels = blob.Channels() * blob.Depth();
	int channels4 = Ceil(totalChannels, 4);

	channelGroupSize = getChannelGroupSize( blob.Height(), channels4 );
	int groupCount = Ceil(channels4, channelGroupSize);

	const CVulkanImage* image = getTmpImage( (TTmpVulkanImage)imageId,
		blob.Width() * blob.ObjectCount() * groupCount, blob.Height() * channelGroupSize );

	const CVulkanImage* images[] = { image };

	CMemoryHandle bufs[1] = { blobData };
	size_t sizes[1] = { blob.BlobSize() * sizeof(float) };

	PARAM_STRUCT( PrepareBlobForConvolutionAdreno ) param =
		{ { blob.Width(), blob.Height() }, blob.ObjectCount(), totalChannels, channels4, channelGroupSize };

	runShader( shaderLoader->GET_SHADER_DATA( PrepareBlobForConvolutionAdreno, true, 1, 0, 1),
		&param, sizeof(param), images, 1, 0, 0, bufs, sizes, 1,
		blob.Width() * blob.ObjectCount(), blob.Height() * channels4, 1 );

	return *image;
}

void CVulkanMathEngine::blobConvolution3x3s1d1Adreno( const CCommonConvolutionDesc& desc,
	const CConstFloatHandle& sourceData, const CConstFloatHandle& filterData, const CConstFloatHandle* freeTermData,
	const CFloatHandle& resultData )
{
	ASSERT_EXPR( device->Type == VDT_Adreno );
	ASSERT_EXPR( device->IsImageBased );

	const CBlobDesc& source = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Result;

	int channels = source.Depth() * source.Channels();

	int paddingTop = desc.PaddingHeight;
	int height3 = Ceil(result.Height(), 3);
	int totalHeight = height3 * 3 + 2;
	int paddingBottom = totalHeight - source.Height() - paddingTop;

	int paddingLeft = desc.PaddingWidth;
	int width4 = Ceil(result.Width(), 4);
	int totalWidth = width4 * 4 + 4;
	int paddingRight = totalWidth - source.Width() - paddingLeft;

	int totalWidth4 = totalWidth / 4;

	int inputChannelGroupSize = 0;
	const CVulkanImage& imageSource = blobConvolution3x3s1d1PrepareSourceAdreno( source, sourceData,
		paddingTop, paddingBottom, paddingLeft, paddingRight, TVI_ConvSource, inputChannelGroupSize);
	const CVulkanImage& imageFilter = blobConvolution3x3s1d1PrepareFilterAdreno( filter, filterData, TVI_ConvFilter );
	const CVulkanImage* imageFreeTerm = &imageFilter;
	if( freeTermData != 0 ) {
		imageFreeTerm = &batchVectorToImage(1, *freeTermData, filter.ObjectCount(), TVI_FreeTerm);
	}

	/////////////////////////////////////////////////////////////////////////////////
	// Convolution code
	const CVulkanImage* samplers[3] = { &imageSource, &imageFilter, imageFreeTerm };

	CMemoryHandle bufs[1] = { resultData };
	size_t sizes[1] = { result.BlobSize() * sizeof(float) };

	PARAM_STRUCT(BlobConvolution3x3s1d1Adreno) param = { totalWidth4, totalHeight, channels,
		source.ObjectCount(), result.Width(), result.Height(), filter.ObjectCount(), (freeTermData == 0) ? 0 : 1,
		inputChannelGroupSize };

	runShader( shaderLoader->GET_SHADER_DATA(BlobConvolution3x3s1d1Adreno, true, 0, 3, 1),
		&param, sizeof(param), 0, 0, samplers, 3, bufs, sizes, 1,
		width4 * result.ObjectCount(), height3 * filter.ObjectCount(), 1 );
}

const CVulkanImage& CVulkanMathEngine::blobConvolution3x3s1d1PrepareFilterAdreno( const CBlobDesc& filter,
	const CConstFloatHandle& filterData, int imageId )
{
	ASSERT_EXPR( device->Type == VDT_Adreno );
	ASSERT_EXPR( device->IsImageBased );

	int channels = filter.Depth() * filter.Channels();
	int fullHeight = channels * 3;

	const CVulkanImage* images[1] =
		{ getTmpImage( (TTmpVulkanImage)imageId, filter.ObjectCount(), fullHeight ) };

	CMemoryHandle bufs[1] = { filterData };
	size_t sizes[1] = { filter.BlobSize() * sizeof(float) };

	PARAM_STRUCT( PrepareFilter3x3ForConvolutionAdreno ) param = { filter.ObjectCount(), channels };

	runShader( shaderLoader->GET_SHADER_DATA( PrepareFilter3x3ForConvolutionAdreno, true, 1, 0, 1),
		&param, sizeof(param), images, 1, 0, 0, bufs, sizes, 1, filter.ObjectCount(), fullHeight, 1 );

	return *images[0];
}

const CVulkanImage& CVulkanMathEngine::blobConvolution3x3s1d1PrepareSourceAdreno( const CBlobDesc& blob,
	const CConstFloatHandle& blobData, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight,
	int imageId, int& channelGroupSize )
{
	ASSERT_EXPR( device->Type == VDT_Adreno );
	ASSERT_EXPR( device->IsImageBased );

	int totalWidth = blob.Width() + paddingLeft + paddingRight;

	ASSERT_EXPR((totalWidth % 4) == 0);

	int totalWidth4 = totalWidth / 4 * blob.ObjectCount();
	int channels = blob.Depth() * blob.Channels();
	int totalHeight = (blob.Height() + paddingTop + paddingBottom);

	channelGroupSize = getChannelGroupSize( totalHeight, channels );
	int channelGroupCount = Ceil(channels, channelGroupSize);

	const CVulkanImage* images[1] = { getTmpImage( (TTmpVulkanImage)imageId,
		totalWidth4 * channelGroupCount, totalHeight * channelGroupSize ) };

	totalHeight *= channels;

	CMemoryHandle bufs[1] = { blobData };
	size_t sizes[1] = { blob.BlobSize() * sizeof(float) };

	PARAM_STRUCT(PrepareBlobWithPaddingAdreno) param = { channels,
		blob.Width(), blob.Height(), blob.ObjectCount(), paddingTop, paddingBottom, paddingLeft, paddingRight,
		channelGroupSize };

	runVectorShader(shaderLoader->GET_SHADER_DATA(PrepareBlobWithPaddingAdreno, true, 1, 0, 1),
		&param, sizeof(param), images, 1, 0, 0, bufs, sizes, 1, totalWidth4 * totalHeight);

	return *images[0];
}

void CVulkanMathEngine::blobConvolution3x3s1d1( const CCommonConvolutionDesc& desc,
	const CConstFloatHandle& sourceData, const CConstFloatHandle& filterData, const CConstFloatHandle* freeTermData,
	const CFloatHandle& resultData )
{
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Result;

	// Convert the input and the filter into NCHW format
	int channels = source.Depth() * source.Channels();

	int paddingTop = desc.PaddingHeight;
	int height3 = Ceil(result.Height(), 3);
	int totalHeight = height3 * 3 + 2;
	int paddingBottom = totalHeight - source.Height() - paddingTop;

	int paddingLeft = desc.PaddingWidth;
	int width4 = Ceil(result.Width(), 4);
	int totalWidth = width4 * 4 + 2;
	int paddingRight = totalWidth - source.Width() - paddingLeft;

	CFloatHandleStackVar prepSource( mathEngine(), source.ObjectCount() * channels * totalHeight * totalWidth );
	blobConvolution3x3s1d1PrepareSource( source, sourceData, paddingTop, paddingBottom, paddingLeft, paddingRight,
		prepSource );

	CFloatHandleStackVar prepFilter( mathEngine(), filter.BlobSize() );
	TransposeMatrix( filter.ObjectCount(), filterData,
		filter.Height() * filter.Width(), 1, filter.Depth() * filter.Channels(), 1, prepFilter, prepFilter.Size() );

	/////////////////////////////////////////////////////////////////////////////////
	// Convolution code
	CMemoryHandle bufs[4] = { prepSource.GetHandle(), prepFilter.GetHandle(),
		(freeTermData == 0) ? prepFilter.GetHandle() : *freeTermData, resultData };
	size_t sizes[4] = { prepSource.Size() * sizeof(float), prepFilter.Size() * sizeof(float),
		filter.ObjectCount() * sizeof(float), result.BlobSize() * sizeof(float) };

	PARAM_STRUCT( BlobConvolution3x3s1d1 ) param = { totalWidth, totalHeight, channels, source.ObjectCount(),
		result.Width(), result.Height(), filter.ObjectCount(), (freeTermData == 0) ? 0 : 1 };

	runShader( shaderLoader->GET_SHADER_DATA( BlobConvolution3x3s1d1, true, 0, 0, 4), &param,
		sizeof(param), 0, 0, 0, 0, bufs, sizes, 4, width4, height3 * result.ObjectCount(), filter.ObjectCount() );
}

void CVulkanMathEngine::blobConvolution3x3s1d1PrepareSource( const CBlobDesc& blob, const CConstFloatHandle& blobData,
	int paddingTop, int paddingBottom, int paddingLeft, int paddingRight, const CFloatHandle& buffer )
{
	int channels = blob.Depth() * blob.Channels();
	int totalHeight = blob.Height() + paddingTop + paddingBottom;
	int totalWidth = blob.Width() + paddingLeft + paddingRight;

	int vectorSize = blob.ObjectCount() * totalHeight * totalWidth * channels;

	CMemoryHandle bufs[2] = { blobData, buffer };
	size_t sizes[2] = { blob.BlobSize() * sizeof(float), vectorSize * sizeof(float) };

	PARAM_STRUCT(PrepareBlobWithPaddingBuffers) param = { channels, blob.Width(), blob.Height(), blob.ObjectCount(),
		paddingTop, paddingBottom, paddingLeft, paddingRight };

	runVectorShader( shaderLoader->GET_SHADER_DATA(PrepareBlobWithPaddingBuffers, true, 0, 0, 2),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2, vectorSize );
}

void CVulkanMathEngine::blobConvolutionImpl1Adreno( const CCommonConvolutionDesc& desc,
	const CConstFloatHandle& /*sourceData*/, const CConstFloatHandle& /*filterData*/, bool isFreeTerm,
	const CFloatHandle& resultData, int startChannel, int inputChannelGroupSize, int filterChannelGroupSize )
{
	ASSERT_EXPR( device->Type == VDT_Adreno );
	ASSERT_EXPR( device->IsImageBased );

	const CBlobDesc& source = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Result;

	int totalInputChannels = source.Channels() * source.Depth();
	int inputChannels4 = Ceil(totalInputChannels, 4);

	int channelsToProc = result.Depth() * result.Channels() - startChannel;

	const CVulkanImage* samplers[3] = {
		getTmpImage(TVI_ConvSource),
		getTmpImage(TVI_ConvFilter),
		( isFreeTerm ? getTmpImage(TVI_FreeTerm) : getTmpImage(TVI_ConvFilter) )
	};

	CMemoryHandle bufs[1] = { resultData };
	size_t sizes[1] = { result.BlobSize() * sizeof(float) };

	PARAM_STRUCT( BlobConvolutionAdreno ) param = { { desc.PaddingWidth, desc.PaddingHeight },
		{ desc.StrideWidth, desc.StrideHeight }, { desc.DilationWidth, desc.DilationHeight }, isFreeTerm ? 1 : 0,
		result.Width(), result.Height(), result.ObjectCount(), source.Width(), source.Height(), inputChannels4,
		filter.Width(), filter.Height(), filter.ObjectCount(), startChannel,
		inputChannelGroupSize, filterChannelGroupSize };

	runShader( shaderLoader->GET_SHADER_DATA( BlobConvolutionAdreno, true, 0, 3, 1),
		&param, sizeof(param), 0, 0, samplers, 3, bufs, sizes, 1,
		result.Width() * result.ObjectCount(), channelsToProc * result.Height(), 1 );
}

void CVulkanMathEngine::blobConvolutionImpl8Adreno( const CCommonConvolutionDesc& desc,
	const CConstFloatHandle& /*sourceData*/, const CConstFloatHandle& /*filterData*/, bool isFreeTerm,
	const CFloatHandle& resultData, int channels8, int inputChannelGroupSize, int filterChannelGroupSize )
{
	ASSERT_EXPR( device->Type == VDT_Adreno );
	ASSERT_EXPR( device->IsImageBased );

	const CBlobDesc& source = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Result;

	int totalInputChannels = source.Channels() * source.Depth();
	int inputChannels4 = Ceil(totalInputChannels, 4);

	const CVulkanImage* samplers[3] = {
		getTmpImage(TVI_ConvSource),
		getTmpImage(TVI_ConvFilter),
		( isFreeTerm ? getTmpImage(TVI_FreeTerm) : getTmpImage(TVI_ConvFilter) )
	};

	CMemoryHandle bufs[1] = { resultData };
	size_t sizes[1] = { result.BlobSize() * sizeof(float) };

	PARAM_STRUCT( BlobConvolution8Adreno ) param = { { desc.PaddingWidth, desc.PaddingHeight },
		{ desc.StrideWidth, desc.StrideHeight }, { desc.DilationWidth, desc.DilationHeight }, isFreeTerm ? 1 : 0,
		result.Width(), result.Height(), result.ObjectCount(), source.Width(), source.Height(), inputChannels4,
		filter.Width(), filter.Height(), filter.ObjectCount(), channels8,
		inputChannelGroupSize, filterChannelGroupSize };

	runShader( shaderLoader->GET_SHADER_DATA( BlobConvolution8Adreno, true, 0, 3, 1), &param, sizeof(param),
		0, 0, samplers, 3, bufs, sizes, 1, result.Width() * result.ObjectCount(), channels8 * result.Height(), 1 );
}

void CVulkanMathEngine::blobConvolutionImpl1( const CCommonConvolutionDesc& desc,
	const CFloatHandleStackVar& sourceData, const CFloatHandleStackVar& filterData,
	const CConstFloatHandle* freeTermData, const CFloatHandle& resultData, int startChannel, int totalChannels )
{
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Result;

	int totalInputChannels = source.Channels() * source.Depth();
	int channelsToProc = totalChannels - startChannel;

	CMemoryHandle bufs[4] = { sourceData.GetHandle(), filterData.GetHandle(),
		( freeTermData != 0 ) ? *freeTermData : filterData, resultData };
	size_t sizes[4] = { sourceData.Size() * sizeof( float ), filterData.Size() * sizeof( float ),
		totalChannels * sizeof( float ), result.BlobSize() * sizeof( float ) };

	PARAM_STRUCT( BlobConvolution ) param = { { desc.PaddingWidth, desc.PaddingHeight },
		{ desc.StrideWidth, desc.StrideHeight }, { desc.DilationWidth, desc.DilationHeight }, freeTermData != 0,
		result.Width(), result.Height(), result.ObjectCount(), source.Width(), source.Height(), totalInputChannels,
		filter.Width(), filter.Height(), filter.ObjectCount(), startChannel };

	runShader( shaderLoader->GET_SHADER_DATA( BlobConvolution, true, 0, 0, 4 ),
		&param, sizeof( param ), 0, 0, 0, 0, bufs, sizes, 4,
		result.Width() * result.ObjectCount(), channelsToProc * result.Height(), 1 );
}

void CVulkanMathEngine::blobConvolutionImpl8( const CCommonConvolutionDesc& desc, const CFloatHandleStackVar& sourceData, const CFloatHandleStackVar& filterData,
	const CConstFloatHandle* freeTermData, const CFloatHandle& resultData, int totalChannels )
{
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Result;

	int channels8 = totalChannels / 8;
	int totalInputChannels = source.Channels() * source.Depth();

	CMemoryHandle bufs[4] = { sourceData.GetHandle(), filterData.GetHandle(),
		( freeTermData != 0 ) ? *freeTermData : filterData.GetHandle(), resultData };
	size_t sizes[4] = { sourceData.Size() * sizeof( float ), filterData.Size() * sizeof( float ),
		totalChannels * sizeof( float ), result.BlobSize() * sizeof( float ) };

	PARAM_STRUCT( BlobConvolution8 ) param = { { desc.PaddingWidth, desc.PaddingHeight },
		{ desc.StrideWidth, desc.StrideHeight } , { desc.DilationWidth, desc.DilationHeight }, freeTermData != 0,
		result.Width(), result.Height(), result.ObjectCount(), source.Width(), source.Height(), totalInputChannels,
		filter.Width(), filter.Height(), filter.ObjectCount(), channels8 };

	runShader( shaderLoader->GET_SHADER_DATA( BlobConvolution8, true, 0, 0, 4 ), &param, sizeof( param ),
		0, 0, 0, 0, bufs, sizes, 4, result.Width() * result.ObjectCount(), channels8 * result.Height(), 1 );
}

// Filter transform from NHWC -> Image(N/4 * H, C * W)
const CVulkanImage& CVulkanMathEngine::blobConvolutionBackwardPrepareFilterAdreno( const CBlobDesc& blob,
	const CConstFloatHandle& blobData, int imageId )
{
	ASSERT_EXPR( device->Type == VDT_Adreno );
	ASSERT_EXPR( device->IsImageBased );

	int totalChannels = blob.Channels() * blob.Depth();
	int batchSize4 = Ceil(blob.ObjectCount(), 4);
	int xSize = blob.Width() * totalChannels;
	int ySize = blob.Height() * batchSize4;

	const CVulkanImage* images[1] = { getTmpImage((TTmpVulkanImage)imageId, xSize, ySize) };

	CMemoryHandle bufs[1] = { blobData };
	size_t sizes[1] = { blob.BlobSize() * sizeof(float) };

	PARAM_STRUCT( PrepareFilterForConvolutionBackwardAdreno ) param =
		{ { blob.Width(), blob.Height() }, totalChannels, blob.ObjectCount(), batchSize4 };

	runShader( shaderLoader->GET_SHADER_DATA( PrepareFilterForConvolutionBackwardAdreno, true, 1, 0, 1),
		&param, sizeof(param), images, 1, 0, 0, bufs, sizes, 1, xSize, ySize, 1 );

	return *images[0];
}

//------------------------------------------------------------------------------------------------------------
// Channelwise convolution

CChannelwiseConvolutionDesc* CVulkanMathEngine::InitBlobChannelwiseConvolution( const CBlobDesc& source,
	int paddingHeight, int paddingWidth, int strideHeight, int strideWidth, 
	const CBlobDesc& filter, const CBlobDesc* freeTerm, const CBlobDesc& result )
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

	CCommonChannelwiseConvolutionDesc* desc = new CCommonChannelwiseConvolutionDesc( paddingHeight, paddingWidth, 
		strideHeight, strideWidth, source, filter, result );
	return desc;
}

void CVulkanMathEngine::blobChannelwiseConvolution3x3s1( const CChannelwiseConvolutionDesc& convDesc, const CConstFloatHandle& sourceData,
	const CConstFloatHandle& filterData, const CConstFloatHandle* freeTermData, const CFloatHandle& resultData )
{
	const CCommonChannelwiseConvolutionDesc& desc = static_cast<const CCommonChannelwiseConvolutionDesc&>( convDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Result;

	int totalChannels = result.Depth() * result.Channels();

	CMemoryHandle bufs[4] = { sourceData, filterData,
	(freeTermData != 0) ? *freeTermData : filterData, resultData };
	size_t sizes[4] = { source.BlobSize() * sizeof(float), filter.BlobSize() * sizeof(float),
		totalChannels * sizeof(float), result.BlobSize() * sizeof(float) };

	PARAM_STRUCT( BlobChannelwiseConvolution3x3s1 ) param = {
			{ desc.PaddingWidth, desc.PaddingHeight },
			(freeTermData != 0) ? 1 : 0,
			totalChannels,
			result.Width(),
			result.Height(),
			result.ObjectCount(),
			source.Width(), source.Height(), 
			filter.Width(), filter.Height() 
		};

	const int combineH = 2;
    const int combineW = 2;
    const int channelBlocksCount = Ceil( result.Height(), combineH ) * Ceil( result.Width() , combineW );
	runShader( shaderLoader->GET_SHADER_DATA( BlobChannelwiseConvolution3x3s1, true, 0, 0, 4),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 4,
		Ceil( totalChannels, 2 ), channelBlocksCount, result.ObjectCount() );
}

void CVulkanMathEngine::blobChannelwiseConvolution3x3s2( const CChannelwiseConvolutionDesc& convDesc, const CConstFloatHandle& sourceData,
	const CConstFloatHandle& filterData, const CConstFloatHandle* freeTermData, const CFloatHandle& resultData )
{
	const CCommonChannelwiseConvolutionDesc& desc = static_cast<const CCommonChannelwiseConvolutionDesc&>( convDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Result;

	int totalChannels = result.Depth() * result.Channels();

	CMemoryHandle bufs[4] = { sourceData, filterData,
		( freeTermData != 0 ) ? *freeTermData : filterData, resultData };
	size_t sizes[4] = { source.BlobSize() * sizeof( float ), filter.BlobSize() * sizeof( float ),
		totalChannels * sizeof( float ), result.BlobSize() * sizeof( float ) };

	PARAM_STRUCT( BlobChannelwiseConvolution3x3s2 ) param = {
		{ desc.PaddingWidth, desc.PaddingHeight },
		( freeTermData != 0 ) ? 1 : 0,
		totalChannels,
		result.Width(),
		result.Height(),
		result.ObjectCount(),
		source.Width(), source.Height(),
		filter.Width(), filter.Height()
	};

	const int combineH = 2;
	const int combineW = 2;
	const int channelBlocksCount = Ceil( result.Height(), combineH ) * Ceil( result.Width(), combineW );
	runShader( shaderLoader->GET_SHADER_DATA( BlobChannelwiseConvolution3x3s2, true, 0, 0, 4 ),
		&param, sizeof( param ), 0, 0, 0, 0, bufs, sizes, 4,
		totalChannels, channelBlocksCount, result.ObjectCount() );
}

void CVulkanMathEngine::BlobChannelwiseConvolution( const CChannelwiseConvolutionDesc& convDesc, const CConstFloatHandle& sourceData,
	const CConstFloatHandle& filterData, const CConstFloatHandle* freeTermData, const CFloatHandle& resultData )
{
	const CCommonChannelwiseConvolutionDesc& desc = static_cast<const CCommonChannelwiseConvolutionDesc&>( convDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Result;

	if( device->Type == VDT_Adreno ) {
		int inputChannelGroupSize = 0;
		const CVulkanImage& imageSource = prepareBlobForConvolutionAdreno(source, sourceData, TVI_ConvSource, inputChannelGroupSize);
		int filterChannelGroupSize = 0;
		const CVulkanImage& imageFilter = prepareBlobForConvolutionAdreno(filter, filterData, TVI_ConvFilter, filterChannelGroupSize);
		const CVulkanImage* imageFreeTerm = &imageFilter;
		if( freeTermData != 0 ) {
			imageFreeTerm = &batchVectorToImage( 1, *freeTermData, filter.Channels(), TVI_FreeTerm );
		}

		int totalChannels = result.Depth() * result.Channels();
		int channels4 = Ceil(totalChannels, 4);

		const CVulkanImage* samplers[3] = { &imageSource, &imageFilter, imageFreeTerm };

		CMemoryHandle bufs[1] = { resultData };
		size_t sizes[1] = { result.BlobSize() * sizeof(float) };

		PARAM_STRUCT(BlobChannelwiseConvolutionAdreno) param = { { desc.PaddingWidth, desc.PaddingHeight },
			{ desc.StrideWidth, desc.StrideHeight }, { 1, 1 }, (freeTermData != 0) ? 1 : 0,
			totalChannels, result.Width(), result.Height(), result.ObjectCount(), source.Width(), source.Height(),
			filter.Width(), filter.Height(), inputChannelGroupSize, filterChannelGroupSize };

		runShader( shaderLoader->GET_SHADER_DATA(BlobChannelwiseConvolutionAdreno, true, 0, 3, 1),
			&param, sizeof(param), 0, 0, samplers, 3, bufs, sizes, 1,
			result.Width() * result.ObjectCount(), channels4 * result.Height(), 1 );
	} else {
		if(filter.Width() == 3 && filter.Height() == 3 && desc.StrideHeight == 1 && desc.StrideWidth == 1) {
			blobChannelwiseConvolution3x3s1(desc, sourceData, filterData, freeTermData, resultData);
			return;
		} else if( filter.Width() == 3 && filter.Height() == 3 && desc.StrideHeight == 2 && desc.StrideWidth == 2 ) {
			blobChannelwiseConvolution3x3s2( desc, sourceData, filterData, freeTermData, resultData );
			return;
		}

		int totalChannels = result.Depth() * result.Channels();

		CMemoryHandle bufs[4] = { sourceData, filterData,
			(freeTermData != 0) ? *freeTermData : filterData, resultData };
		size_t sizes[4] = { source.BlobSize() * sizeof(float), filter.BlobSize() * sizeof(float),
			totalChannels * sizeof(float), result.BlobSize() * sizeof(float) };

		PARAM_STRUCT( BlobChannelwiseConvolution ) param = {
			{ desc.PaddingWidth, desc.PaddingHeight },
			{ desc.StrideWidth, desc.StrideHeight },
			(freeTermData != 0) ? 1 : 0,
			totalChannels,
			result.Width(),
			result.Height(),
			result.ObjectCount(),
			source.Width(), source.Height(), 
			filter.Width(), filter.Height() 
		};

		runShader( shaderLoader->GET_SHADER_DATA( BlobChannelwiseConvolution, true, 0, 0, 4),
			&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 4,
			totalChannels, result.Height() * result.Width(), result.ObjectCount() );
	}
}

void CVulkanMathEngine::BlobChannelwiseConvolutionBackward( const CChannelwiseConvolutionDesc&, const CConstFloatHandle&,
	const CConstFloatHandle&, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::BlobChannelwiseConvolutionLearnAdd( const CChannelwiseConvolutionDesc&, const CConstFloatHandle&,
	const CConstFloatHandle&, const CFloatHandle&, const CFloatHandle* )
{
	ASSERT_EXPR( false );
}

} // namespace NeoML

#endif // NEOML_USE_VULKAN
