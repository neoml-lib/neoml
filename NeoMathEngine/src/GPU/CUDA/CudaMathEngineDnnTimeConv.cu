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

#include <Kernels/CudaDnnTimeConvKernels.h>

namespace NeoML {

void CCudaMathEngine::blobTimeConvolutionPrepare( const CCudaTimeConvolutionDescInternal& desc, float* data, const CFloatHandle& sourceData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );

	const CCudaBlobDesc& filter = desc.Filter;
	const CCudaBlobDesc& result = desc.Result;
	const CCudaBlobDesc& source = desc.Source;

	int xSizeNorm = source.BatchWidth() * source.ObjectSize();
	xSizeNorm = (xSizeNorm + BlobTimeConvolutionPrepareCombine - 1) / BlobTimeConvolutionPrepareCombine;

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid3DMinZYX(1, 1, 512, blockCount, threadCount, filter.Height(), result.BatchLength(), xSizeNorm);
	BlobTimeConvolutionPrepareKernel<<<blockCount, threadCount, 0, cudaStream>>>( desc, GetRaw( sourceData ), xSizeNorm, data );
}

CTimeConvolutionDesc* CCudaMathEngine::InitTimeConvolution( const CBlobDesc& source,
	int stride, int padding, int dilation,
	const CBlobDesc& filter, const CBlobDesc& result )
{
	ASSERT_EXPR( stride > 0 );
	ASSERT_EXPR( padding >= 0 );
	ASSERT_EXPR( dilation > 0 );
	ASSERT_EXPR( filter.BatchLength() == 1 );
	ASSERT_EXPR( filter.Width() == 1 );
	ASSERT_EXPR( filter.Depth() == 1 );
	ASSERT_EXPR( filter.Channels() == source.ObjectSize() );
	ASSERT_EXPR( source.BatchLength() + 2 * padding >= ( filter.Height() - 1 ) * dilation + 1 );
	ASSERT_EXPR( result.BatchLength() == ( source.BatchLength() - ( filter.Height() - 1 ) * dilation - 1 + 2 * padding ) / stride + 1 );
	ASSERT_EXPR( result.BatchWidth() == source.BatchWidth() );
	ASSERT_EXPR( result.ListSize() == 1 && source.ListSize() == 1 );
	ASSERT_EXPR( result.Width() == 1 );
	ASSERT_EXPR( result.Height() == 1 );
	ASSERT_EXPR( result.Depth() == 1 );
	ASSERT_EXPR( result.Channels() == filter.BatchWidth() );
	ASSERT_EXPR( padding < ( filter.Height() - 1 ) * dilation + 1 );

	CCudaTimeConvolutionDesc* desc = new CCudaTimeConvolutionDesc();
	desc->Internal.Source = source;
	desc->Internal.Filter = filter;
	desc->Internal.Result = result;
	desc->Internal.Stride = stride;
	desc->Internal.Padding = padding;
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

	int workspaceSize = 0;

	bool useTempMatrix = desc.Stride > 1 || filter.Height() > 1;

	if( useTempMatrix ) {
		// Create a temporary matrix
		workspaceSize = result.BatchLength() * source.BatchWidth() * filter.Height() * source.ObjectSize();
	}

	CFloatHandleStackVar buffer( mathEngine(), workspaceSize );
	CConstFloatHandle curSourceData;
	if( useTempMatrix ) {
		blobTimeConvolutionPrepare( desc, GetRaw(buffer.GetHandle()), sourceData );
		curSourceData = buffer.GetHandle();
	} else {
		curSourceData = sourceData;
	}

	// Convolution
	MultiplyMatrixByTransposedMatrix(curSourceData,
		result.BatchLength() * source.BatchWidth(), filter.Height() * source.ObjectSize(), filter.Height() * source.ObjectSize(),
		filterData, filter.ObjectCount(), filter.Height() * source.ObjectSize(),
		resultData, filter.ObjectCount(), result.BlobSize());

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
		BlobTimeConvolutionBackwardUnpackKernel<<<blockCount, threadCount, 0, cudaStream>>>( desc, GetRaw( outputDiffData ),
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

	const CCudaTimeConvolutionDescInternal& desc = static_cast<const CCudaTimeConvolutionDesc&>( convDesc ).Internal;
	const CCudaBlobDesc& filterDiff = desc.Filter;
	const CCudaBlobDesc& outputDiff = desc.Result;

	// Train the filter
	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, desc.Filter.BlobSize() );
	blobTimeConvolutionLearnFilterKernel<<<blockCount, threadCount, 0, cudaStream>>>( desc, GetRaw( inputData ),
		GetRaw( outputDiffData ), GetRaw( filterDiffData ) );

	// Train the free term
	SumMatrixRowsAdd( 1, freeTermDiffData, outputDiffData, outputDiff.ObjectCount(), filterDiff.ObjectCount() );
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
