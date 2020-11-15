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
#include <CudaMathEngineDnnConvs.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>

#include <Kernels/CudaDnnRleConvKernels.h>

namespace NeoML {

void CCudaMathEngine::blobConvertFromRle( const CCudaRleConvolutionDesc& desc, const CFloatHandle& sourceData,
	const CFloatHandle& resultData )
{
	const CCudaConvolutionDescInternal& convDesc = static_cast<const CCudaConvolutionDesc*>( desc.ConvDesc )->Internal;
	const CCudaBlobDesc& source = convDesc.Source;

	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	ASSERT_EXPR( source.Depth() == 1 );
	ASSERT_EXPR( source.Channels() == 1 );
	SetCudaDevice( device->DeviceNumber );

	dim3 blockCount;
	dim3 threadCount; 

	getCudaTaskGrid2D(blockCount, threadCount, source.ObjectCount(), source.Height());

	BlobConvertFromRleKernel<<<blockCount, threadCount>>>( convDesc, desc.StrokeValue, desc.NonStrokeValue,
		GetRaw( sourceData ), source.ObjectSize() * sizeof(float), GetRaw( resultData ) );
}

CRleConvolutionDesc* CCudaMathEngine::InitBlobRleConvolution( const CBlobDesc& source, float strokeValue,
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

	CCudaRleConvolutionDesc* desc = new CCudaRleConvolutionDesc();
	desc->StrokeValue = strokeValue;
	desc->NonStrokeValue = nonStrokeValue;

	desc->ConvDesc = static_cast<CCudaConvolutionDesc*>( InitBlobConvolution( source, 0, 0, strideHeight, strideWidth, 1, 1, filter, result, AF_None ) );

	return desc;
}

void CCudaMathEngine::BlobRleConvolution( const CRleConvolutionDesc& desc, const CFloatHandle& sourceData,
	const CFloatHandle& filterData, const CFloatHandle* freeTermData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( filterData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermData == 0 || freeTermData->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCudaConvolutionDescInternal& convDesc = static_cast<const CCudaRleConvolutionDesc&>( desc ).ConvDesc->Internal;
	CFloatHandleVar inputConverted( mathEngine(), convDesc.Source.BlobSize() );
	blobConvertFromRle( static_cast<const CCudaRleConvolutionDesc&>(desc), sourceData, inputConverted );
	BlobConvolution( *static_cast<const CCudaRleConvolutionDesc&>(desc).ConvDesc, inputConverted, filterData, freeTermData, resultData );
}

void CCudaMathEngine::BlobRleConvolutionLearnAdd( const CRleConvolutionDesc& desc,
	const CFloatHandle& sourceData, const CFloatHandle& outputDiffData, const CFloatHandle& filterDiffData,
	const CFloatHandle* freeTermDiffData )
{
	const CCudaConvolutionDescInternal& convDesc = static_cast<const CCudaRleConvolutionDesc&>( desc ).ConvDesc->Internal;
	CFloatHandleVar inputConverted( mathEngine(), convDesc.Source.BlobSize() );
	blobConvertFromRle( static_cast<const CCudaRleConvolutionDesc&>(desc), sourceData, inputConverted );
	BlobConvolutionLearnAdd( *static_cast<const CCudaRleConvolutionDesc&>(desc).ConvDesc, inputConverted, CFloatHandle(),
		outputDiffData, filterDiffData, freeTermDiffData, false );
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
