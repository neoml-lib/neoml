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

#include <algorithm>

#include "CpuRowwiseCommon.h"
#include "CpuRowwiseInterface.h"
#include <CpuMathEngineDnnChannelwiseConv.h>
#include <CpuMathEngine.h>

namespace NeoML {

class CCpuMathEngine::CCpuRowwiseMobileNetV2 : public ICpuRowwiseImpl, public CRowwiseOperationDesc {
public:
	CCpuRowwiseMobileNetV2( CCpuMathEngine& mathEngine, int inputChannels,
			const float* expandFilter, const float* expandFreeTerm, int expandedChannels,
			TActivationFunction expandActivation, float expandReluParam,
			const float* channelwiseFilter, const float* channelwiseFreeTerm, int stride,
			TActivationFunction channelwiseActivation, float channelwiseReluParam,
			const float* downFilter, const float* downFreeTerm, int outputChannels, bool residual ) :
		mathEngine( mathEngine ),
		inputChannels( inputChannels ),
		expandFilter( expandFilter ),
		expandFreeTerm( expandFreeTerm ),
		expandedChannels( expandedChannels ),
		expandActivation( expandActivation ),
		expandReluParam( expandReluParam ),
		desc( 1, 1, stride, stride, CBlobDesc(), CBlobDesc( { 3, 3, 1, expandedChannels } ), CBlobDesc() ),
		channelwiseFilter( channelwiseFilter ),
		channelwiseFreeTerm( channelwiseFreeTerm ),
		channelwiseActivation( channelwiseActivation ),
		channelwiseReluParam( channelwiseReluParam ),
		downFilter( downFilter ),
		downFreeTerm( downFreeTerm ),
		outputChannels( outputChannels ),
		residual( residual ),
		inputRowRequirement( 0 ),
		outputRowRequirement( 0 )
	{
	}

	CBlobDesc Reshape( const CBlobDesc& inputSize ) override;
	int InputRowRequirement() const override { return inputRowRequirement; }
	int OutputRowRequirement() const override { return outputRowRequirement; }
	int InOperationBufferSize() const override { return getMaxOutputRowsPerStep() * expandedChannels * desc.Result.Width(); }
	int OutputRowCount() const override { return desc.Result.ObjectCount() * desc.Result.Height(); }
	int OutputRowSize() const override { return desc.Result.Width() * outputChannels; }
	bool IsTrivial() const override { return false; }
	CProcessingReport Process( const float* input, int inputRowIndex, int inputRowsAvailable,
		float* output, int outputRowIndex, int outputRowsAvailable, float* buffer ) const override;

private:
	CCpuMathEngine& mathEngine;
	int inputChannels;
	const float* expandFilter;
	const float* expandFreeTerm;
	int expandedChannels;
	TActivationFunction expandActivation;
	float expandReluParam;
	CCommonChannelwiseConvolutionDesc desc;
	const float* channelwiseFilter;
	const float* channelwiseFreeTerm;
	TActivationFunction channelwiseActivation;
	float channelwiseReluParam;
	const float* downFilter;
	const float* downFreeTerm;
	int outputChannels;
	bool residual;
	// Inner rowwise buffer is needed because we need to transfer some rows after expand conv
	// between different Process calls (usual inOperationBuffer doesn't save data between calls)
	// It's caused by the fact that single input row is used by multiply output rows in 3x3 channelwise
	mutable std::unique_ptr<CCpuRowwiseBuffer> chInput;
	int inputRowRequirement;
	int outputRowRequirement;

	int getMaxInputRowsPerStep() const;
	int getMaxOutputRowsPerStep() const;
};

inline CBlobDesc CCpuMathEngine::CCpuRowwiseMobileNetV2::Reshape( const CBlobDesc& inputSize )
{
	CBlobDesc chInputSize = inputSize;
	chInputSize.SetDimSize( BD_Channels, expandedChannels );
	CBlobDesc outputSize = chInputSize;
	if( desc.StrideHeight == 2 ) {
		outputSize.SetDimSize( BD_Height, ( outputSize.Height() + 1 ) / 2 );
		outputSize.SetDimSize( BD_Width, ( outputSize.Width() + 1 ) / 2 );
	}
	desc = CCommonChannelwiseConvolutionDesc( desc.PaddingHeight, desc.PaddingWidth, desc.StrideHeight, desc.StrideWidth,
		chInputSize, desc.Filter, outputSize );
	outputSize.SetDimSize( BD_Channels, outputChannels );

	// Because of math desc.Result.Width() <= desc.Source.Width()
	if( desc.Result.Width() < RowwiseMatMulRequiredHeight ) {
		outputRowRequirement = ( RowwiseMatMulRequiredHeight + desc.Result.Width() - 1 ) / desc.Result.Width();
		inputRowRequirement = 3 + ( outputRowRequirement - 1 ) * desc.StrideHeight;
	} else {
		outputRowRequirement = 0;
		inputRowRequirement = 3;
	}

	return outputSize;
}

// Number of rows which can be processed at one time in expand conv
inline int CCpuMathEngine::CCpuRowwiseMobileNetV2::getMaxInputRowsPerStep() const
{
	// Determine which row size is bigger: before or after the expand conv
	const int maxRowSize = std::max( inputChannels, expandedChannels ) * desc.Source.Width();
	// Determine the number required for effective calculation
	// Taking into consideration both facts:
	//     - inputRowRequirement is always present and must be met (otherwise the math will become broken)
	//     - if rows are too small then take into consideration RowwiseCacheSize
	const int recommendedRowCount = std::max( inputRowRequirement, RowwiseCacheSize / maxRowSize );
	// But there is no need to allocate more data than the whole input
	return std::min( desc.Source.ObjectCount() * desc.Source.Height(), recommendedRowCount );
}

// Number of rows which can be processed at one time in down conv
inline int CCpuMathEngine::CCpuRowwiseMobileNetV2::getMaxOutputRowsPerStep() const
{
	// Determine which row size is bigger: before or after the down conv
	const int maxRowSize = std::max( expandedChannels, outputChannels ) * desc.Result.Width();
	// Determine the number required for effective calculation
	// Taking into consideration both facts:
	//     - downConv is a matmul, that's why outputRowRequirement must be met
	//     - if rows are too small then take into consideration RowwiseCacheSize
	//     - both of the above can be 0 (in case of wide rows with many channels)
	const int recommendedRowCount = std::max( { 1, outputRowRequirement, RowwiseCacheSize / maxRowSize } );
	// But there is no need to allocate more data than the whole output
	return std::min( desc.Result.ObjectCount() * desc.Result.Height(), recommendedRowCount );
}

inline ICpuRowwiseImpl::CProcessingReport CCpuMathEngine::CCpuRowwiseMobileNetV2::Process( const float* input, int inputRowIndex,
	int inputRowsAvailable, float* output, int outputRowIndex, int outputRowsAvailable, float* buffer ) const
{
	CProcessingReport report = RowwiseConvProcessingReport( inputRowIndex, inputRowsAvailable, outputRowIndex,
		outputRowsAvailable, desc.Source.Height(), desc.Result.Height(), desc.Filter.Height(), desc.PaddingHeight,
		desc.StrideHeight, 1 );
	if( report.OutputRowsCalculated == 0 ) {
		PRESUME_EXPR( report.InputRowsMayBeRemoved == 0 );
		return report;
	}

	const int maxOutputRowsPerStep = getMaxOutputRowsPerStep();
	const int maxChRowsPerStep = std::max( 1,
		RowwiseCacheSize / ( desc.Source.Width() * expandedChannels ) );

	if( chInput == nullptr ) {
		chInput.reset( new CCpuRowwiseBuffer( mathEngine, getMaxInputRowsPerStep(),
			desc.Source.Width() * expandedChannels, desc.Source.Height() * desc.Source.ObjectCount() ) );
	}

	PRESUME_EXPR( chInput->DataRowProcessed() >= inputRowIndex );
	PRESUME_EXPR( chInput->DataRowProcessed() <= inputRowIndex + inputRowsAvailable );
	PRESUME_EXPR( chInput->DataRowCount() <= 3 - desc.StrideHeight );
	PRESUME_EXPR( chInput->DataRowIndex() == RowwiseConvFirstInputRow( outputRowIndex, desc.Source.Height(),
		desc.Result.Height(), desc.StrideHeight, desc.PaddingHeight ) );

	const int inputWidth = desc.Source.Width();
	const int outputWidth = desc.Result.Width();
	const int inputRowSize = inputWidth * inputChannels;
	const int chOutputRowSize = outputWidth * expandedChannels;

	const float* residualInput = input + ( outputRowIndex - inputRowIndex ) * inputRowSize;
	// CCpuMathEngine::ExecuteRowwise will never execute this block in-place
	// But in order to avoid copy-paste of this chunk of code it's also used in CCpuMathEngine::MobileNetV2Block
	// and CCpuMathEngine::MobileNetV2Block must support in-place execution
	const bool isInPlace = ( residualInput == output );

	const int outputRowsThisCall = outputRowIndex + report.OutputRowsCalculated;
	const int lastImageIndex = ( outputRowsThisCall - 1 ) / desc.Result.Height();
	// Total number of input rows used during this call
	const int inputRowsUsedThisCall = lastImageIndex * desc.Source.Height()
		+ std::min( desc.Source.Height(), ( outputRowsThisCall - 1 - lastImageIndex * desc.Result.Height() ) * desc.StrideHeight + 2 );

	while( outputRowIndex < outputRowsThisCall ) {
		// Process a bunch of rows of input image (till channelwise convolution: expandConv + expandReLU)
		const int inputRowsThisStep = std::min( { inputRowsUsedThisCall - chInput->DataRowProcessed(),
			chInput->EmptyRowCount() } );

		if( inputRowsThisStep > 0 ) {
			const float* expandConvInput = input + ( chInput->DataRowProcessed() - inputRowIndex ) * inputRowSize;
			// Apply expand convolution with activation
			mathEngine.multiplyMatrixByTransposedWithFreeTerm( expandConvInput, inputRowsThisStep * inputWidth, inputChannels,
				expandFilter, expandedChannels, expandFreeTerm, chInput->EmptyRows() );
			MOBILENET_ACTIVATION( expandActivation, expandReluParam, chInput->EmptyRows(), inputRowsThisStep * chInput->RowSize() );
			chInput->AddRows( inputRowsThisStep );
		}

		// Calculate how many output rows we can calculate with the processed input rows
		const int imageIndex = ( chInput->DataRowProcessed() - 1 ) / desc.Source.Height();
		const int inputImageRowsInBuffer = chInput->DataRowProcessed() - imageIndex * desc.Source.Height();
		const int outputImageRowsCanBeProcessed = std::min( outputRowsThisCall - imageIndex * desc.Result.Height(),
			inputImageRowsInBuffer >= desc.Source.Height() ? desc.Result.Height()
				: ( inputImageRowsInBuffer < 2 ? 0 : 1 + ( inputImageRowsInBuffer - 2 ) / desc.StrideHeight ) );
		const int outputRowsCanBeProcessed = imageIndex * desc.Result.Height() + outputImageRowsCanBeProcessed;
		const float* chInputBuff = chInput->DataRows();
		int chInputRowIndex = chInput->DataRowIndex();

		while( outputRowIndex < outputRowsCanBeProcessed ) {
			// Process channelwise output rows (while there are any)
			const int outputRowsThisStep = std::min<int>( maxOutputRowsPerStep, outputRowsCanBeProcessed - outputRowIndex );

			float* chOutput = buffer;
			int chOutputRowIndex = outputRowIndex;
			PRESUME_EXPR( chOutputRowIndex / desc.Result.Height() == chInputRowIndex / desc.Source.Height() );

			while( chOutputRowIndex < outputRowIndex + outputRowsThisStep ) {
				const int chRowsThisStep = std::min( { maxChRowsPerStep,
					outputRowIndex + outputRowsThisStep - chOutputRowIndex,
					desc.Result.Height() - chOutputRowIndex % desc.Result.Height() } );
				ProcessChannelwise3x3( desc, chRowsThisStep, chInputBuff, chInputRowIndex % desc.Source.Height(),
					channelwiseFilter, channelwiseFreeTerm, chOutput, chOutputRowIndex % desc.Result.Height() );
				MOBILENET_ACTIVATION( channelwiseActivation, channelwiseReluParam, chOutput, chRowsThisStep * chOutputRowSize );
				chOutput += chRowsThisStep * chOutputRowSize;
				chOutputRowIndex += chRowsThisStep;

				if( chOutputRowIndex % desc.Result.Height() == 0 && chOutputRowIndex < outputRowsThisCall ) {
					// Switch to the next image in batch
					const int nextImageIndex = chOutputRowIndex / desc.Result.Height();
					const int diff = nextImageIndex * desc.Source.Height() - chInputRowIndex;
					PRESUME_EXPR( diff >= 0 );
					chInputBuff += diff * chInput->RowSize();
					chInputRowIndex += diff;
				}
			}

			if( residual && isInPlace ) {
				// Block input and output are located in the same memory
				// It's possible to simultaneously calculate down conv output and add the residual connection
				mathEngine.multiplyMatrixByTransposedMatrixAndAdd( buffer, outputRowsThisStep * outputWidth,
					expandedChannels, expandedChannels, downFilter, outputChannels, expandedChannels, output,
					outputChannels );
			} else {
				mathEngine.multiplyMatrixByTransposedMatrix( buffer, outputRowsThisStep * outputWidth,
					expandedChannels, expandedChannels, downFilter, outputChannels, expandedChannels,
					output, outputChannels );
			}

			if( downFreeTerm != nullptr ) {
				mathEngine.addVectorToMatrixRows( output, output, outputRowsThisStep * outputWidth, outputChannels,
					outputChannels, outputChannels, downFreeTerm );
			}

			if( residual && !isInPlace ) {
				// Input and output are located in different memory regions
				// Add residual connection
				vectorAdd( output, residualInput, output, outputRowsThisStep * outputWidth * outputChannels );
				residualInput += outputRowsThisStep * inputWidth * inputChannels;
			}

			output += outputRowsThisStep * outputChannels * outputWidth;
			outputRowIndex += outputRowsThisStep;
		}

		if( outputRowIndex < desc.Result.ObjectCount() * desc.Result.Height() ) {
			const int firstInputRowNeeded = RowwiseConvFirstInputRow( outputRowIndex,
				desc.Source.Height(), desc.Result.Height(), desc.StrideHeight, desc.PaddingHeight );
			if( firstInputRowNeeded > chInput->DataRowIndex() ) {
				chInput->RemoveRows( firstInputRowNeeded - chInput->DataRowIndex() );
			}
		}
	}

	if( outputRowIndex == desc.Result.ObjectCount() * desc.Result.Height() ) {
		chInput.reset( nullptr );
	}

	return report;
}

} // namespace NeoML
