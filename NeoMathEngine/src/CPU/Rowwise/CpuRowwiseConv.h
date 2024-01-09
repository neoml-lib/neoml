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

#pragma once

#include "CpuRowwiseInterface.h"
#include "CpuRowwiseCommon.h"
#include <CpuMathEngineDnnConv.h>
#include <CpuMathEngine.h>

namespace NeoML {

class CCpuMathEngine::CCpuRowwiseConv : public ICpuRowwiseImpl, public CRowwiseOperationDesc {
public:
	CCpuRowwiseConv( CCpuMathEngine& mathEngine,
			int inputChannels, int padH, int padW, int strideH, int strideW,
			int dilH, int dilW, int fC, int fH, int fW, const float* filter, const float* freeTerm ) :
		mathEngine( mathEngine ),
		desc( CBlobDesc{}, CBlobDesc{}, CBlobDesc( { 1, fC, 1, fH, fW, 1, inputChannels } ), padH, padW,
			strideH, strideW, dilH, dilW ),
		filter( filter ),
		freeTerm( freeTerm ),
		inputRowRequirement( 0 ),
		outputRowRequirement( 0 )
	{}

	CBlobDesc Reshape( const CBlobDesc& inputSize ) override;
	int InputRowRequirement() const override { return inputRowRequirement; }
	int OutputRowRequirement() const override { return outputRowRequirement; }
	int InOperationBufferSize() const override;
	int OutputRowCount() const override { return desc.Result.ObjectCount() * desc.Result.Height(); }
	int OutputRowSize() const override { return desc.Result.Width() * desc.Result.Channels(); }
	bool IsTrivial() const override { return false; }
	CProcessingReport Process( const float* input, int inputRowIndex, int inputRowsAvailable,
		float* output, int outputRowIndex, int outputRowsAvailable, float* buffer ) const override;

private:
	CCpuMathEngine& mathEngine;
	CCpuConvolutionDesc desc;
	const float* const filter;
	const float* const freeTerm;

	int inputRowRequirement{};
	int outputRowRequirement{};


	bool is1x1Conv() const { return desc.Filter.GeometricalSize() == 1 && desc.PaddingHeight == 0
		&& desc.PaddingWidth == 0 && desc.StrideHeight == 1 && desc.StrideWidth == 1; }
	int getCacheItemCount() const;
};

//-------------------------------------------------------------------------------------------------------------

inline CBlobDesc CCpuMathEngine::CCpuRowwiseConv::Reshape( const CBlobDesc& inputSize )
{
	auto convOutputSize = [] ( int input, int filter, int padding, int stride, int dilation ) -> int
	{
		return 1 + ( input - ( filter - 1 ) * dilation + 2 * padding - 1 ) / stride;
	};

	desc.Source = inputSize;
	desc.Result = desc.Source;
	desc.Result.SetDimSize( BD_Height, convOutputSize( inputSize.Height(), desc.Filter.Height(), desc.PaddingHeight,
		desc.StrideHeight, desc.DilationHeight ) );
	desc.Result.SetDimSize( BD_Width, convOutputSize( inputSize.Width(), desc.Filter.Width(), desc.PaddingWidth,
		desc.StrideWidth, desc.DilationWidth ) );
	desc.Result.SetDimSize( BD_Channels, desc.Filter.ObjectCount() );

	if( mathEngine.simdMathEngine != nullptr ) {
		desc.SimdConvolutionDesc.reset( mathEngine.simdMathEngine->InitBlobConvolution(
			desc.Source, desc.PaddingHeight, desc.PaddingWidth, desc.StrideHeight, desc.StrideWidth,
			desc.DilationHeight, desc.DilationWidth, desc.Filter, desc.Result ) );
	}

	const int effectiveFilterSize = 1 + ( desc.Filter.Height() - 1 ) * desc.DilationHeight;

	inputRowRequirement = effectiveFilterSize;
	outputRowRequirement = 0;
	if( desc.SimdConvolutionDesc == nullptr && desc.Result.Width() < RowwiseMatMulRequiredHeight ) {
		// Tricky case: if conv will be calculating 1 row at a time matmul will be ineffective
		// (only SIMD algo doesn't have matmul inside)
		outputRowRequirement = ( RowwiseMatMulRequiredHeight + desc.Result.Width() - 1 ) / desc.Result.Width();
		inputRowRequirement = effectiveFilterSize + desc.StrideHeight * ( outputRowRequirement - 1 );
	}
	return desc.Result;
}

inline int CCpuMathEngine::CCpuRowwiseConv::InOperationBufferSize() const
{
	if( is1x1Conv() || desc.SimdConvolutionDesc != nullptr ) {
		return 0;
	}

	return getCacheItemCount() * desc.Filter.ObjectSize();
}

inline ICpuRowwiseImpl::CProcessingReport CCpuMathEngine::CCpuRowwiseConv::Process( const float* input, int inputRowIndex,
	int inputRowsAvailable, float* output, int outputRowIndex, int outputRowsAvailable, float* buffer ) const
{
	CProcessingReport report = RowwiseConvProcessingReport( inputRowIndex, inputRowsAvailable, outputRowIndex,
		outputRowsAvailable, desc.Source.Height(), desc.Result.Height(), desc.Filter.Height(), desc.PaddingHeight,
		desc.StrideHeight, desc.DilationHeight );
	if( report.OutputRowsCalculated == 0 ) {
		PRESUME_EXPR( report.InputRowsMayBeRemoved == 0 );
		return report;
	}

	if( desc.SimdConvolutionDesc != nullptr ) {
		mathEngine.simdMathEngine->BlobConvolutionRowwise( *desc.SimdConvolutionDesc,
			input, inputRowIndex, filter, freeTerm,
			output, outputRowIndex, report.OutputRowsCalculated );
		return report;
	}

	const int inputRowSize = desc.Source.Width() * desc.Source.Channels();
	const int firstInputRowUsed = RowwiseConvFirstInputRow( outputRowIndex, desc.Source.Height(),
		desc.Result.Height(), desc.StrideHeight, desc.PaddingHeight );
	if( inputRowIndex < firstInputRowUsed ) {
		const int diff = firstInputRowUsed - inputRowIndex;
		input += diff * inputRowSize;
		inputRowIndex += diff;
		inputRowsAvailable -= diff;
	}

	if( is1x1Conv() ) {
		const int firstHeight = report.OutputRowsCalculated * desc.Result.Width();
		const int firstWidth = desc.Source.Channels();
		const int secondHeight = desc.Result.Channels();
		const int secondWidth = firstWidth;
		const int resultWidth = secondHeight;

		mathEngine.multiplyMatrixByTransposedMatrix( input, firstHeight, firstWidth, firstWidth,
			filter, secondHeight, secondWidth, output, resultWidth );
		if( freeTerm != nullptr ) {
			mathEngine.addVectorToMatrixRows( output, output, firstHeight,
				resultWidth, resultWidth, resultWidth, freeTerm );
		}
		return report;
	}

	const int filterObjectCount = desc.Filter.ObjectCount();
	const int filterObjectSize = desc.Filter.ObjectSize();
	const int cacheItemCount = getCacheItemCount();

	const int imageStartOffset = inputRowIndex * desc.Source.Width() * desc.Source.Channels();
	const int count = report.OutputRowsCalculated * desc.Result.Width();
	const int start = outputRowIndex * desc.Result.Width();

	const int firstWidth = filterObjectSize;
	const int secondHeight = filterObjectCount;
	const int secondWidth = firstWidth;
	const int resultWidth = secondHeight;

	int index = 0;
	while( index < count ) {
		const int size = std::min( count - index, cacheItemCount );

		mathEngine.fillTempData( input - imageStartOffset, buffer, desc, start + index, size );

		float* resultDataPtr = output + index * filterObjectCount;

		const int firstHeight = size;

		mathEngine.multiplyMatrixByTransposedMatrix( buffer, firstHeight, firstWidth, firstWidth,
			filter, secondHeight, secondWidth, resultDataPtr, resultWidth );

		if( freeTerm!= nullptr ) {
			mathEngine.addVectorToMatrixRows( resultDataPtr, resultDataPtr, size,
				resultWidth, resultWidth, resultWidth, freeTerm );
		}
		index += size;
	}

	return report;
}

inline int CCpuMathEngine::CCpuRowwiseConv::getCacheItemCount() const
{
	const int resultItemCount = desc.Result.Width() * desc.Result.Height();
	return std::max( 1, std::min( ceilTo( BlobConvolutionCacheSize / desc.Filter.ObjectSize(), 16 ),
		resultItemCount ) );
}

} // namespace NeoML
