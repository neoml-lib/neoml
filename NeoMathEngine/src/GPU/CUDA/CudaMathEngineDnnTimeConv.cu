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
#include <CudaMathEngineDnnConvs.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>
#include <CudaDevice.h>
#include <CudaCommon.h>

#include <Kernels/CudaDnnTimeConvKernels.h>

namespace NeoML {

CTimeConvolutionDesc* CCudaMathEngine::InitTimeConvolution( const CBlobDesc& source,
	int stride, int paddingFront, int paddingBack, int dilation,
	const CBlobDesc& filter, const CBlobDesc& result )
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

	CCudaTimeConvolutionDesc* desc = new CCudaTimeConvolutionDesc();
	desc->Internal.Source = source;
	desc->Internal.Filter = filter;
	desc->Internal.Result = result;
	desc->Internal.Stride = stride;
	desc->Internal.PaddingFront = paddingFront;
	desc->Internal.PaddingBack = paddingBack;
	desc->Internal.Dilation = dilation;
	return desc;
}

void CCudaMathEngine::BlobTimeConvolution( const CTimeConvolutionDesc& convDesc,
	const CFloatHandle& sourceData, const CFloatHandle& filterData, const CFloatHandle& freeTermData,
	const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( filterData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCudaTimeConvolutionDescInternal& desc = static_cast<const CCudaTimeConvolutionDesc&>( convDesc ).Internal;
	const CCudaBlobDesc& source = desc.Source;
	const CCudaBlobDesc& filter = desc.Filter;
	const CCudaBlobDesc& result = desc.Result;

	if( filter.Height() == 1 && desc.Stride == 1 ) {
		// This assert has already been checked in InitTimeConvolution
		ASSERT_EXPR( desc.PaddingFront == 0 && desc.PaddingBack == 0 );
		// Trivial case
		MultiplyMatrixByTransposedMatrix(sourceData,
			source.BatchLength() * source.BatchWidth(), source.ObjectSize(), source.ObjectSize(),
			filterData, filter.ObjectCount(), source.ObjectSize(),
			resultData + desc.PaddingFront * filter.ObjectCount(), filter.ObjectCount(), result.BlobSize());
	} else {
		// Convolution through temp matrix
		const int tempMatrixWidth = filter.ObjectSize();
		const int tempMatrixHeight = result.BlobSize() / filter.ObjectCount();
		// Max amount of memory allowed is a half of math engine's free memory
		const int maxInMemoryHeight = max( 1,
			min( static_cast<int>( GetFreeMemorySize() / 2 / ( sizeof( float ) * tempMatrixWidth ) ), tempMatrixHeight ) );

		int matrixRowIndex = 0;
		CFloatHandle currResult = resultData;
		CFloatHandleStackVar tempMatrixPart( mathEngine(), maxInMemoryHeight * tempMatrixWidth );

		// Build temp matrix part by part and add filterDiff of that part
		while( matrixRowIndex < tempMatrixHeight ) {
			const int currPartHeight = min( tempMatrixHeight - matrixRowIndex, maxInMemoryHeight );

			dim3 blockCount;
			dim3 threadCount;
			getCudaTaskGrid2D( blockCount, threadCount, currPartHeight, tempMatrixWidth );

			BuildTempMatrixKernel<<<blockCount, threadCount>>>( desc, GetRaw( sourceData ), currPartHeight,
				tempMatrixWidth, GetRaw( tempMatrixPart.GetHandle() ), matrixRowIndex );
			MultiplyMatrixByTransposedMatrix(tempMatrixPart, currPartHeight, tempMatrixWidth, tempMatrixWidth,
				filterData, filter.ObjectCount(), tempMatrixWidth,
				currResult, filter.ObjectCount(), result.BlobSize());

			matrixRowIndex += currPartHeight;
			currResult += currPartHeight * filter.ObjectCount();
		}
	}

	// Free term
	AddVectorToMatrixRows( 1, resultData, resultData, result.ObjectCount(), result.ObjectSize(), freeTermData );
}

void CCudaMathEngine::BlobTimeConvolutionBackward( const CTimeConvolutionDesc& convDesc,
	const CFloatHandle& outputDiffData, const CFloatHandle& filterData, const CFloatHandle& /*freeTerm*/,
	const CFloatHandle& inputDiffData )
{
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( filterData.GetMathEngine() == this );
	ASSERT_EXPR( inputDiffData.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const CCudaTimeConvolutionDescInternal& desc = static_cast<const CCudaTimeConvolutionDesc&>( convDesc ).Internal;
	const CCudaBlobDesc& inputDiff = desc.Source;
	const CCudaBlobDesc& filter = desc.Filter;
	const CCudaBlobDesc& outputDiff = desc.Result;

	int workspaceSize = 0;

	bool useTempMatrix = desc.Stride > 1 || filter.Height() > 1;

	int targetDataSize = outputDiff.BatchLength() * inputDiff.BatchWidth() * filter.Height() * inputDiff.ObjectSize();

	if( useTempMatrix ) {
		// Create a temporary matrix
		workspaceSize = targetDataSize;
	}

	CFloatHandleStackVar buffer( mathEngine(), workspaceSize );
	CFloatHandle targetData = (useTempMatrix) ? buffer.GetHandle() : inputDiffData;

	// Reverse convolution
	MultiplyMatrixByMatrix( 1, outputDiffData, outputDiff.ObjectCount(), outputDiff.ObjectSize(),
		filterData, filter.ObjectSize(), targetData, targetDataSize );

	if( useTempMatrix ) {
		// Add up the data from the temporary matrix
		int combineCount = BlobTimeConvolutionBackwardUnpackCombine / filter.Height();
		if(combineCount == 0) {
			combineCount = 1;
		}
		int xSizeNorm = (inputDiff.ObjectSize() + combineCount - 1) / combineCount;

		dim3 blockCount;
		dim3 threadCount;
		getCudaTaskGrid2DMinYX(1, 512, blockCount, threadCount, inputDiff.ObjectCount(), xSizeNorm);
		BlobTimeConvolutionBackwardUnpackKernel<<<blockCount, threadCount>>>( desc, GetRaw( outputDiffData ),
			GetRaw( filterData ), GetRaw( inputDiffData ), xSizeNorm, combineCount, GetRaw( targetData ) );
	}
}

void CCudaMathEngine::BlobTimeConvolutionLearnAdd( const CTimeConvolutionDesc& convDesc, const CFloatHandle& inputData,
	const CFloatHandle& outputDiffData, const CFloatHandle& filterDiffData, const CFloatHandle& freeTermDiffData )
{
	ASSERT_EXPR( inputData.GetMathEngine() == this );
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( filterDiffData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermDiffData.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const CCudaTimeConvolutionDescInternal& desc = static_cast<const CCudaTimeConvolutionDesc&>( convDesc ).Internal;
	const CCudaBlobDesc& filterDiff = desc.Filter;
	const CCudaBlobDesc& outputDiff = desc.Result;

	// Train the filter
	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, desc.Filter.BlobSize() );
	blobTimeConvolutionLearnFilterKernel<<<blockCount, threadCount>>>( desc, GetRaw( inputData ),
		GetRaw( outputDiffData ), GetRaw( filterDiffData ) );

	// Train the free term
	SumMatrixRowsAdd( 1, freeTermDiffData, outputDiffData, outputDiff.ObjectCount(), filterDiff.ObjectCount() );
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
