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

	if( desc.Stride == 1 && filter.Height() == 1 ) {
		// This assert has already been checked in InitTimeConvolution
		ASSERT_EXPR( desc.PaddingFront == 0 && desc.PaddingBack == 0 );
		// Trivial case
		MultiplyMatrixByMatrix( 1, outputDiffData, outputDiff.ObjectCount(), outputDiff.ObjectSize(),
			filterData, filter.ObjectSize(), inputDiffData, inputDiff.BlobSize() );
	} else {
		// Let's try to build temp matrix
		const int tempMatrixWidth = filter.ObjectSize();
		const int tempMatrixHeight = outputDiff.BlobSize() / filter.ObjectCount();
		// Max amount of memory allowed is a half of math engine's free memory
		const int maxInMemoryHeight = max( 1,
			min( static_cast<int>( GetFreeMemorySize() / 2 / ( sizeof( float ) * tempMatrixWidth ) ), tempMatrixHeight ) );

		int matrixRowIndex = 0;
		CFloatHandle currOutputDiff = outputDiffData;
		CFloatHandleStackVar tempMatrixPart( mathEngine(), maxInMemoryHeight * tempMatrixWidth );

		VectorFill( inputDiffData, 0.f, inputDiff.BlobSize() );
		
		const int combineCount = max( 1, BlobTimeConvolutionBackwardUnpackCombine / filter.Height() );
		const int xSizeNorm = (inputDiff.ObjectSize() + combineCount - 1) / combineCount;
		while( matrixRowIndex < tempMatrixHeight ) {
			const int currPartHeight = min( tempMatrixHeight - matrixRowIndex, maxInMemoryHeight );

			MultiplyMatrixByMatrix( 1, currOutputDiff, currPartHeight, outputDiff.ObjectSize(),
				filterData, filter.ObjectSize(), tempMatrixPart, maxInMemoryHeight * tempMatrixWidth );

			dim3 blockCount;
			dim3 threadCount;
			getCudaTaskGrid2DMinYX(1, 512, blockCount, threadCount, inputDiff.ObjectCount(), xSizeNorm);
			BlobTimeConvolutionBackwardUnpackKernel<<<blockCount, threadCount>>>( desc, GetRaw( filterData ),
				GetRaw( inputDiffData ), xSizeNorm, combineCount, GetRaw( tempMatrixPart.GetHandle() ), matrixRowIndex, currPartHeight );

			currOutputDiff += currPartHeight * outputDiff.ObjectSize();
			matrixRowIndex += currPartHeight;
		}
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
	if( filterDiff.Height() == 1 && desc.Stride == 1 ) {
		// This assert has already been checked in InitTimeConvolution
		ASSERT_EXPR( desc.PaddingFront == 0 && desc.PaddingBack == 0 );
		// Trivial case
		MultiplyTransposedMatrixByMatrixAndAdd( outputDiffData, desc.Source.ObjectCount(),
			outputDiff.ObjectSize(), outputDiff.ObjectSize(), inputData, desc.Source.ObjectSize(),
			desc.Source.ObjectSize(), filterDiffData, filterDiff.ObjectSize(), filterDiff.BlobSize() );
	} else {
		// Let's try to build temp matrix
		const int tempMatrixWidth = filterDiff.ObjectSize();
		const int tempMatrixHeight = outputDiff.BlobSize() / filterDiff.ObjectCount();
		// Max amount of memory allowed is a half of math engine's free memory
		const int maxInMemoryHeight = min( static_cast<int>( GetFreeMemorySize() / 2 / ( sizeof( float ) * tempMatrixWidth ) ),
			tempMatrixHeight );

		if( maxInMemoryHeight == 0 ) {
			// naive implementatino which doesn't use additional memory
			int blockCount;
			int threadCount;
			getCudaTaskGrid( blockCount, threadCount, desc.Filter.BlobSize() );
			BlobTimeConvolutionLearnFilterKernel<<<blockCount, threadCount>>>( desc, GetRaw( inputData ),
				GetRaw( outputDiffData ), GetRaw( filterDiffData ) );
		} else {
			int matrixRowIndex = 0;
			CFloatHandle currOutputDiff = outputDiffData;
			CFloatHandleStackVar tempMatrixPart( mathEngine(), maxInMemoryHeight * tempMatrixWidth );
			const int filterCount = desc.Result.ObjectSize();

			// Build temp matrix part by part and add filterDiff of that part
			while( matrixRowIndex < tempMatrixHeight ) {
				const int currPartHeight = min( tempMatrixHeight - matrixRowIndex, maxInMemoryHeight );

				dim3 blockCount;
				dim3 threadCount;
				getCudaTaskGrid2D( blockCount, threadCount, currPartHeight, tempMatrixWidth );

				BuildTempMatrixKernel<<<blockCount, threadCount>>>( desc, GetRaw( inputData ), currPartHeight,
					tempMatrixWidth, GetRaw( tempMatrixPart.GetHandle() ), matrixRowIndex );
				MultiplyTransposedMatrixByMatrixAndAdd( currOutputDiff, currPartHeight, filterCount, filterCount,
					tempMatrixPart.GetHandle(), tempMatrixWidth, tempMatrixWidth, filterDiffData, tempMatrixWidth, filterDiff.BlobSize() );

				matrixRowIndex += currPartHeight;
				currOutputDiff += currPartHeight * filterCount;
			}
		}
	}

	// Train the free term
	SumMatrixRowsAdd( 1, freeTermDiffData, outputDiffData, outputDiff.ObjectCount(), filterDiff.ObjectCount() );
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
