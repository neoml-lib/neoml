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

#include <memory>

#include "CpuRowwiseCommon.h"
#include "CpuRowwiseInterface.h"
#include <CpuMathEngineDnnChannelwiseConv.h>
#include <CpuMathEngine.h>

namespace NeoML {

class CCpuMathEngine::CCpuRowwiseChConvWith1x1 : public ICpuRowwiseImpl, public CRowwiseOperationDesc {
public:
	CCpuRowwiseChConvWith1x1( CCpuMathEngine& mathEngine, int stride, const float* chFilter, const float* chFreeTerm,
			TActivationFunction activation, float reluParam, const float* convFilter,
			const float* convFreeTerm, int outputChannels, bool residual ) :
		mathEngine( mathEngine ),
		chFilter( chFilter ),
		chFreeTerm( chFreeTerm ),
		activation( activation ),
		reluParam( reluParam ),
		convFilter( convFilter ),
		convFreeTerm( convFreeTerm ),
		outputChannels( outputChannels ),
		residual( residual ),
		desc( 1, 1, stride, stride, CBlobDesc{}, CBlobDesc{}, CBlobDesc{} ),
		inputRowRequirement( 0 ),
		outputRowRequirement( 0 ),
		smallMatricesMulDescs( mathEngine )
	{}

	CBlobDesc Reshape( const CBlobDesc& inputSize ) override;
	int InputRowRequirement() const override { return inputRowRequirement; }
	int OutputRowRequirement() const override { return outputRowRequirement; }
	int InOperationBufferSize() const override
		{ return desc.Result.Channels() * desc.Result.Width() * getMaxOutputRowsPerStep(); }
	int OutputRowCount() const override { return desc.Result.ObjectCount() * desc.Result.Height(); }
	int OutputRowSize() const override { return desc.Result.Width() * outputChannels; }
	bool IsTrivial() const override { return false; }
	CProcessingReport Process( const float* input, int inputRowIndex, int inputRowsAvailable,
		float* output, int outputRowIndex, int outputRowsAvailable, float* buffer ) const override;

private:
	CCpuMathEngine& mathEngine;
	const float* const chFilter;
	const float* const chFreeTerm;
	const TActivationFunction activation;
	const float reluParam;
	const float* const convFilter;
	const float* const convFreeTerm;
	const int outputChannels;
	const bool residual;

	CCommonChannelwiseConvolutionDesc desc;
	int inputRowRequirement{};
	int outputRowRequirement{};
	// The array of matrices multiplication optimization descriptors
	CCpuSmallMatricesMultiplyDescsArray</*Height*/> smallMatricesMulDescs;
	// If ( outputChannels or inputChannels ) is being changed,
	// for the smallMatricesMulDescs method DestroyAll() should be called to recreate JIT.

	int getMaxOutputRowsPerStep() const;
};

//--------------------------------------------------------------------------------------------------------------

// NOTICE: This method Reshape() is called from the RunOnce() function, avoid heavy reinitializations here
inline CBlobDesc CCpuMathEngine::CCpuRowwiseChConvWith1x1::Reshape( const CBlobDesc& inputSize )
{
	CBlobDesc outputSize = inputSize;
	outputSize.SetDimSize( BD_Height, 1 + ( outputSize.Height() - 1 ) / desc.StrideHeight );
	outputSize.SetDimSize( BD_Width, 1 + ( outputSize.Width() - 1 ) / desc.StrideWidth );
	CBlobDesc filterSize( CT_Float );
	filterSize.SetDimSize( BD_Height, 3 );
	filterSize.SetDimSize( BD_Width, 3 );
	filterSize.SetDimSize( BD_Channels, inputSize.Channels() );
	desc = CCommonChannelwiseConvolutionDesc( desc.PaddingHeight, desc.PaddingWidth, desc.StrideHeight,
		desc.StrideWidth, inputSize, filterSize, outputSize );
	outputSize.SetDimSize( BD_Channels, outputChannels );

	if( desc.Result.Width() < RowwiseMatMulRequiredHeight ) {
		outputRowRequirement = ( RowwiseMatMulRequiredHeight + desc.Result.Width() - 1 ) / desc.Result.Width();
		inputRowRequirement = 3 + ( outputRowRequirement - 1 ) * desc.StrideHeight;
	} else {
		inputRowRequirement = 3;
		outputRowRequirement = 0;
	}

	return outputSize;
}

// Number of rows which can be processed at one time in down conv
inline int CCpuMathEngine::CCpuRowwiseChConvWith1x1::getMaxOutputRowsPerStep() const
{
	// Determine which row size is bigger: before or after the 1x1 conv
	const int maxRowSize = std::max( desc.Result.Channels(), outputChannels ) * desc.Result.Width();
	// Determine the number required for effective calculation
	// Taking into consideration both facts:
	//     - 1x1 conv is a matmul, that's why outputRowRequirement must be met
	//     - if rows are too small then take into consideration RowwiseCacheSize
	//     - both of the above can be 0 (in case of wide rows with many channels)
	const int recommendedRowCount = std::max( { 1, outputRowRequirement, RowwiseCacheSize / maxRowSize } );
	// But there is no need to allocate more data than the whole output
	return std::min( desc.Result.ObjectCount() * desc.Result.Height(), recommendedRowCount );
}

inline ICpuRowwiseImpl::CProcessingReport CCpuMathEngine::CCpuRowwiseChConvWith1x1::Process( const float* input,
	int inputRowIndex, int inputRowsAvailable, float* output, int outputRowIndex, int outputRowsAvailable,
	float* buffer ) const
{
	PRESUME_EXPR( !residual || inputRowIndex <= outputRowIndex );
	CProcessingReport report = RowwiseConvProcessingReport( inputRowIndex, inputRowsAvailable, outputRowIndex,
		outputRowsAvailable, desc.Source.Height(), desc.Result.Height(), desc.Filter.Height(), desc.PaddingHeight,
		desc.StrideHeight, 1 );
	if( report.OutputRowsCalculated == 0 ) {
		PRESUME_EXPR( report.InputRowsMayBeRemoved == 0 );
		return report;
	}

	const int outputRowsAfterThisCall = outputRowIndex + report.OutputRowsCalculated;
	const int maxRowsPerStep = getMaxOutputRowsPerStep();
	const int chInputRowSize = desc.Source.Channels() * desc.Source.Width();
	const int chOutputRowSize = desc.Result.Channels() * desc.Result.Width();
	const int outputWidth = desc.Result.Width();
	const int inputChannels = desc.Source.Channels();

	const int firstWidth = inputChannels;
	const int secondHeight = outputChannels;
	const int resultWidth = secondHeight;

	const float* residualInput = input + ( outputRowIndex - inputRowIndex ) * desc.Source.Width() * desc.Source.Channels();
	const int maxChRowsPerStep = std::max( 1, RowwiseCacheSize / std::max( chOutputRowSize, chInputRowSize ) );

	while( outputRowIndex < outputRowsAfterThisCall ) {
		// Determine how many output rows of whole block we're going to calculate this step
		const int outputRowsThisStep = std::min( maxRowsPerStep, outputRowsAfterThisCall - outputRowIndex );
		const int outputRowsAfterThisStep = outputRowIndex + outputRowsThisStep;

		// Process channelwise conv and write its result into buffer
		int chOutputRowIndex = outputRowIndex;
		float* chOutput = buffer;
		while( chOutputRowIndex < outputRowsAfterThisStep ) {
			// Because of matrix multiplication in 1x1 conv chOutputRowSize * outputRowsThisStep may be quite big
			// Which is why channelwise convolution is performed in smaller steps
			// (unlike matrix multiplication it doesn't slow down when output rows don't have enough columns)
			const int outputImageRowIndex = chOutputRowIndex % desc.Result.Height();
			const int chRowsThisStep = std::min( { outputRowsAfterThisStep - chOutputRowIndex, maxChRowsPerStep,
				desc.Result.Height() - outputImageRowIndex } );

			ProcessChannelwise3x3( desc, chRowsThisStep, input, inputRowIndex % desc.Source.Height(),
				chFilter, chFreeTerm, chOutput, outputImageRowIndex );
			MOBILENET_ACTIVATION( activation, reluParam, chOutput, chRowsThisStep * chOutputRowSize );
			chOutput += chRowsThisStep * chOutputRowSize;
			chOutputRowIndex += chRowsThisStep;

			if( chOutputRowIndex % desc.Result.Height() == 0 && chOutputRowIndex < outputRowsAfterThisCall ) {
				// Switch to the next image in batch
				const int nextImageIndex = chOutputRowIndex / desc.Result.Height();
				const int diff = nextImageIndex * desc.Source.Height() - inputRowIndex;
				PRESUME_EXPR( diff >= 0 );
				input += diff * desc.Source.Width() * inputChannels;
				inputRowIndex += diff;
				inputRowsAvailable -= diff;
			}
		}

		const int firstHeight = outputRowsThisStep * outputWidth;
		auto mulDesc = smallMatricesMulDescs.Get( firstHeight,
			firstHeight, firstWidth, /*secondWidth*/firstWidth, resultWidth );

		mathEngine.multiplyMatrixByTransposedWithFreeTerm( /*first*/buffer, firstHeight, firstWidth,
			/*second*/convFilter, secondHeight, /*freeTerm*/convFreeTerm, /*result*/output, mulDesc );
		if( residual ) {
			vectorAdd( output, residualInput, output, firstHeight * outputChannels );
			residualInput += outputRowsThisStep * desc.Source.Width() * inputChannels;
		}

		output += outputRowsThisStep * outputChannels * outputWidth;
		outputRowIndex += outputRowsThisStep;
	}

	return report;
}

} // namespace NeoML
