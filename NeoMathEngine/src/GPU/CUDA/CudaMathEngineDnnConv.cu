/* Copyright © 2017-2023 ABBYY

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
#include <CudaCommon.h>
#include <CudaDevice.h>
#include <MathEngineCommon.h>
#include <MemoryHandleInternal.h>

#include <Kernels/CudaDnnConvKernels.h>

namespace NeoML {

// Temporary matrix height
static inline int tempMatrixHeight( const CCudaConvolutionDescInternal& desc )
{
	return desc.Source.ObjectCount() * desc.Result.Height() * desc.Result.Width();
}

// Temporary matrix width
static inline int tempMatrixWidth( const CCudaConvolutionDescInternal& desc )
{
	return desc.Filter.ObjectSize();
}

CConvolutionDesc* CCudaMathEngine::InitBlobConvolution( const CBlobDesc& input, int paddingHeight,
	int paddingWidth, int strideHeight, int strideWidth, int dilationHeight, int dilationWidth,
	const CBlobDesc& filter, const CBlobDesc& output )
{
	const int totalInputChannels = input.Channels() * input.Depth();
	const int totalOutputChannels = output.Channels() * output.Depth();

	CCudaConvolutionDesc* desc = new CCudaConvolutionDesc();
	desc->Internal.Source = input;
	desc->Internal.Filter = filter;
	desc->Internal.Result = output;
	desc->Internal.StrideHeight = strideHeight;
	desc->Internal.StrideWidth = strideWidth;
	desc->Internal.PaddingHeight = paddingHeight;
	desc->Internal.PaddingWidth = paddingWidth;
	desc->Internal.DilationHeight = dilationHeight;
	desc->Internal.DilationWidth = dilationWidth;
	return desc;
}

void CCudaMathEngine::BlobConvolution( const CConvolutionDesc& convDesc,
	const CConstFloatHandle& sourceData, const CConstFloatHandle& filterData, const CConstFloatHandle* freeTermData,
	const CFloatHandle& resultData )
{
	SetCudaDevice( device->DeviceNumber );
	const CCudaConvolutionDescInternal& desc = static_cast<const CCudaConvolutionDesc&>( convDesc ).Internal;
	const CCudaBlobDesc& source = desc.Source;
	const CCudaBlobDesc& filter = desc.Filter;
	const CCudaBlobDesc& result = desc.Result;

	if( filter.Height() == 3 && filter.Width() == 3
		&& desc.StrideHeight == 1 && desc.StrideWidth == 1
		&& desc.DilationHeight == 1 && desc.DilationWidth == 1
		&& source.Channels() * source.Depth() < 16 )
	{
		const int widthNorm = ( desc.Result.Width() + 7 ) / 8;
		// Use a convolution kernel of size 3*3 with stride 1
		dim3 blockCount;
		dim3 threadCount;
		getCudaTaskGrid3DMinZYX( 1, 1, 1024, blockCount, threadCount, result.ObjectCount() * result.Height(), widthNorm,
			filter.ObjectCount(), 512 );

		Conv3x3s1d1Kernel1x8<<<blockCount, threadCount>>>( desc, GetRaw( sourceData ), GetRaw( filterData ),
			freeTermData == 0 ? 0 : GetRaw( *freeTermData ), GetRaw( resultData ), widthNorm );
		return;
	}

	if( filter.Height() == 1 && filter.Width() == 1
		&& desc.StrideHeight == 1 && desc.StrideWidth == 1
		&& desc.PaddingHeight == 0 && desc.PaddingWidth == 0 )
	{
		// The convolution is a matrix product anyway, without a temporary matrix
		if( freeTermData != 0 ) {
			// Fill the output matrix with the free term values
			SetVectorToMatrixRows( resultData, result.ObjectCount() * result.Height() * result.Width(),
				filter.ObjectCount(), *freeTermData );

			multiplyMatrixByTransposedMatrixAndAdd( sourceData,
				source.ObjectCount() * result.Height() * result.Width(),
				filter.ObjectSize(), filter.ObjectSize(), filterData,
				filter.ObjectCount(), filter.ObjectSize(), resultData,
				filter.ObjectCount() );
		} else {
			MultiplyMatrixByTransposedMatrix( sourceData,
				source.ObjectCount() * result.Height() * result.Width(),
				filter.ObjectSize(), filter.ObjectSize(), filterData,
				filter.ObjectCount(), filter.ObjectSize(), resultData,
				filter.ObjectCount(), result.BlobSize() );
		}
		return;
	}

	const int tempMatrixWidth = filter.ObjectSize();
	const int tempMatrixHeight = result.ObjectCount() * result.ObjectSize() / filter.ObjectCount();
	const int tempMatrixHeightBatchSize = getCudaTempMatrixMaxHeight( tempMatrixHeight, tempMatrixWidth );

	CFloatHandleStackVar tempMatrix( mathEngine(), tempMatrixHeightBatchSize * tempMatrixWidth );

	int tempMatrixHeightIndex = 0;
	while( tempMatrixHeightIndex < tempMatrixHeight ) {
		const int curTempMatrixHeight = min( tempMatrixHeight - tempMatrixHeightIndex, tempMatrixHeightBatchSize );

		dim3 blockCount;
		dim3 threadCount;
		getCudaTaskGrid2D( blockCount, threadCount, curTempMatrixHeight, source.Depth() * source.Channels() );

		BuildTempMatrixKernel<<<blockCount, threadCount>>>( desc, GetRaw( sourceData ),
			tempMatrixHeightIndex, curTempMatrixHeight, GetRaw( tempMatrix.GetHandle() ) );

		MultiplyMatrixByTransposedMatrix( tempMatrix, curTempMatrixHeight, filter.ObjectSize(), filter.ObjectSize(),
			filterData, filter.ObjectCount(), filter.ObjectSize(),
			resultData + tempMatrixHeightIndex * filter.ObjectCount(),
			filter.ObjectCount(), curTempMatrixHeight * filter.ObjectCount() );

		tempMatrixHeightIndex += curTempMatrixHeight;
	}

	if( freeTermData != 0 ) {
		// Fill the output with the free term values
		AddVectorToMatrixRows( 1, resultData, resultData,
			result.BlobSize() / filter.ObjectCount(), filter.ObjectCount(), *freeTermData );
	}

}

void CCudaMathEngine::BlobConvolutionBackward( const CConvolutionDesc& convDesc, const CConstFloatHandle& outputDiff,
	const CConstFloatHandle& filter, const CConstFloatHandle* freeTerm, const CFloatHandle& inputDiff )
{
	SetCudaDevice( device->DeviceNumber );
	const CCudaConvolutionDescInternal& desc = static_cast<const CCudaConvolutionDesc&>( convDesc ).Internal;
	const int filterCount = desc.Filter.ObjectCount();
	const int filterObjectSize = desc.Filter.ObjectSize();

	if( desc.Filter.Height() == 1 && desc.Filter.Width() == 1
		&& desc.StrideHeight == 1 && desc.StrideWidth == 1
		&& desc.PaddingHeight == 0 && desc.PaddingWidth == 0 )
	{
		// The convolution backward pass is a matrix product without creating a temporary matrix
		MultiplyMatrixByMatrix( 1, outputDiff, desc.Result.BlobSize() / filterCount, filterCount,
			filter, filterObjectSize, inputDiff, desc.Source.BlobSize() );
		if( freeTerm != 0 ) {
			AddVectorToMatrixRows( 1, inputDiff, inputDiff, desc.Source.ObjectCount() * desc.Source.Height() * desc.Source.Width(),
				desc.Source.Channels() * desc.Source.Depth(), *freeTerm );
		}
		return;
	}

	if( freeTerm != 0 ) {
		// Fill the input gradients with the free terms
		SetVectorToMatrixRows( inputDiff, desc.Source.ObjectCount() * desc.Source.Height() * desc.Source.Width(),
			desc.Source.Channels() * desc.Source.Depth(), *freeTerm );
	} else {
		VectorFill( inputDiff, 0.f, desc.Source.BlobSize() );
	}

	TBackwardOperationType operation = BOT_AtomicAdd;
	if( ( desc.Filter.Width() - 1 ) * desc.DilationWidth + 1 <= desc.StrideWidth
		&& ( desc.Filter.Height() - 1 ) * desc.DilationHeight + 1 <= desc.StrideHeight )
	{
		// The filter areas do not intersect, so atomic operations are not needed
		operation = freeTerm == 0 ? BOT_Set : BOT_Add;
	}
	
	// Get the temporary matrix
	const int matrixHeight = tempMatrixHeight( desc );
	const int matrixWidth = tempMatrixWidth( desc );
	const int tempMatrixHeightBatchSize = getCudaTempMatrixMaxHeight( matrixHeight, matrixWidth );
	CFloatHandleStackVar tempMatrix( *this, tempMatrixHeightBatchSize * matrixWidth );

	int tempMatrixHeightIndex = 0;
	while( tempMatrixHeightIndex < matrixHeight ) {
		const int curTempMatrixHeight = min( matrixHeight - tempMatrixHeightIndex, tempMatrixHeightBatchSize );
		MultiplyMatrixByMatrix( 1, outputDiff + tempMatrixHeightIndex * filterCount, curTempMatrixHeight, filterCount,
			filter, filterObjectSize, tempMatrix, tempMatrix.Size() );

		const int widthNorm = ( matrixWidth + BuildInputFromTempMatrixCombine - 1 ) / BuildInputFromTempMatrixCombine;
		// Get the input gradients from the temporary matrix data
		dim3 blockCount;
		dim3 threadCount;
		getCudaTaskGrid2D( blockCount, threadCount, curTempMatrixHeight, widthNorm );

		BuildInputFromTempMatrixKernel<<<blockCount, threadCount>>>( desc, GetRaw( tempMatrix.GetHandle() ),
			curTempMatrixHeight, matrixWidth, GetRaw( inputDiff ), operation, widthNorm, tempMatrixHeightIndex );
		tempMatrixHeightIndex += curTempMatrixHeight;
	}
}

void CCudaMathEngine::BlobConvolutionLearnAdd( const CConvolutionDesc& convDesc,
	const CConstFloatHandle& input, const CConstFloatHandle& outputDiff, const CFloatHandle& filterDiff,
	const CFloatHandle* freeTermDiff, bool isFreeTermDiffFromInput )
{
	SetCudaDevice( device->DeviceNumber );
	const CCudaConvolutionDescInternal& desc = static_cast<const CCudaConvolutionDesc&>( convDesc ).Internal;

	if( freeTermDiff != 0 ) {
		// Get the free term gradient
		if( !isFreeTermDiffFromInput ) {
			SumMatrixRowsAdd( 1, *freeTermDiff, outputDiff, desc.Result.BlobSize() / desc.Filter.ObjectCount(),
				desc.Filter.ObjectCount() );
		} else {
			SumMatrixRowsAdd( 1, *freeTermDiff, input, desc.Source.BlobSize() / desc.Source.Channels(),
				desc.Source.Channels() );
		}
	}

	// Build the temporary matrix
	const int matrixHeight = tempMatrixHeight( desc );
	const int matrixWidth = tempMatrixWidth( desc );
	const int filterCount = desc.Filter.ObjectCount();
	const int tempMatrixHeightBatchSize = getCudaTempMatrixMaxHeight( matrixHeight, matrixWidth );
	CFloatHandleStackVar tempMatrix( *this, tempMatrixHeightBatchSize * matrixWidth );

	int tempMatrixHeightIndex = 0;
	while( tempMatrixHeightIndex < matrixHeight ) {
		const int curTempMatrixHeight = min( matrixHeight - tempMatrixHeightIndex, tempMatrixHeightBatchSize );

		dim3 blockCount;
		dim3 threadCount;
		getCudaTaskGrid2D( blockCount, threadCount, curTempMatrixHeight, desc.Source.Depth() * desc.Source.Channels() );

		BuildTempMatrixKernel<<<blockCount, threadCount>>>( desc, GetRaw( input ), tempMatrixHeightIndex, curTempMatrixHeight,
			GetRaw( tempMatrix.GetHandle() ) );

		// Get the filter gradients by multiplying the temporary matrix and the output gradients
		MultiplyTransposedMatrixByMatrixAndAdd( outputDiff + tempMatrixHeightIndex * filterCount, curTempMatrixHeight,
			filterCount, filterCount, tempMatrix, matrixWidth, matrixWidth, filterDiff, matrixWidth, desc.Filter.BlobSize() );
		tempMatrixHeightIndex += curTempMatrixHeight;
	}
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
