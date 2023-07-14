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
#include <CpuMathEngineDnnChannelwiseConv.h>
#include <CpuMathEngine.h>

namespace NeoML {

class CCpuMathEngine::CRowwiseMobileNetV2 : public IRowwiseCpuImpl, public CRowwiseOperationDesc {
public:
	CRowwiseMobileNetV2( CCpuMathEngine& mathEngine, int inputChannels,
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
		residual( residual )
	{
	}

	int MinInputRowCount() const override { return 3; }

	CBlobDesc Reshape( const CBlobDesc& inputSize ) override;
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
	mutable std::unique_ptr<CRowwiseBuffer> chInput;

	int getMaxInputRowsPerStep() const { return std::max<int>( 1,
		( RowwiseCacheSize / ( std::max<int>( inputChannels, expandedChannels ) * desc.Source.Width() ) ) ); }
	int getMaxOutputRowsPerStep() const { return std::min( desc.Result.Height(), std::max<int>( 1,
		( RowwiseCacheSize / ( std::max<int>( outputChannels, expandedChannels ) * desc.Result.Width() ) ) ) ); }
};

inline CBlobDesc CCpuMathEngine::CRowwiseMobileNetV2::Reshape( const CBlobDesc& inputSize )
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
	return outputSize;
}

inline IRowwiseCpuImpl::CProcessingReport CCpuMathEngine::CRowwiseMobileNetV2::Process( const float* input, int inputRowIndex,
	int inputRowsAvailable, float* output, int outputRowIndex, int outputRowsAvailable, float* buffer ) const
{
	CProcessingReport report = RowwiseConvProcessingReport( inputRowIndex, inputRowsAvailable, outputRowIndex,
		outputRowsAvailable, desc.Source.Height(), desc.Result.Height(), desc.Filter.Height(), desc.PaddingHeight,
		desc.StrideHeight, 1 );
	if( report.OutputRowsCalculated == 0 ) {
		PRESUME_EXPR( report.InputRowsMayBeRemoved == 0 );
		return report;
	}

	const int maxInputRowsPerStep = getMaxInputRowsPerStep();
	const int maxOutputRowsPerStep = getMaxOutputRowsPerStep();

	if( chInput == nullptr ) {
		chInput.reset( new CRowwiseBuffer( mathEngine,
			std::min( desc.Source.Height(), maxInputRowsPerStep + 2 ),
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
	const bool isInPlace = ( residualInput == output );

	const int outputRowsThisCall = outputRowIndex + report.OutputRowsCalculated;
	// Total number of input rows used during this call
	const int inputRowsUsedThisCall = std::min( desc.Source.ObjectCount() * desc.Source.Height(),
		( outputRowsThisCall - 1 ) * desc.StrideHeight + 2 );

	int outputRowsProcessed = outputRowIndex;
	while( outputRowsProcessed < outputRowsThisCall ) {
		// Process a bunch of rows of input image (till channelwise convolution: expandConv + expandReLU)
		const int imageIndex = outputRowsProcessed / desc.Result.Height();
		const int inputRowsInBuffer = std::min( desc.Source.Height(),
			chInput->DataRowProcessed() - imageIndex * desc.Source.Height() );
		const int inputRowsThisStep = std::min( { inputRowsUsedThisCall - chInput->DataRowProcessed(),
			chInput->EmptyRowCount(), desc.Source.Height() - inputRowsInBuffer } );

		if( inputRowsThisStep > 0 ) {
			const float* expandConvInput = input + ( chInput->DataRowProcessed() - inputRowIndex ) * inputRowSize;
			// Apply expand convolution with activation
			mathEngine.multiplyMatrixByTransposedWithFreeTerm( expandConvInput, inputRowsThisStep * inputWidth, inputChannels,
				expandFilter, expandedChannels, expandFreeTerm, chInput->EmptyRows() );
			if( expandActivation == AF_HSwish ) {
				vectorHSwish( chInput->EmptyRows(), chInput->EmptyRows(), inputRowsThisStep * chInput->RowSize() );
			} else if( expandActivation == AF_ReLU ) {
				if( expandReluParam > 0 ) {
					vectorReLU( chInput->EmptyRows(), chInput->EmptyRows(),
						inputRowsThisStep * chInput->RowSize(), expandReluParam );
				} else {
					vectorReLU( chInput->EmptyRows(), chInput->EmptyRows(), inputRowsThisStep * chInput->RowSize() );
				}
			}
			chInput->AddRows( inputRowsThisStep );
		}

		// Calculate how many output rows we can calculate with the processed input rows
		const int inputImageRowsInBuffer = chInput->DataRowProcessed() - imageIndex * desc.Source.Height();
		const int outputImageRowsCanBeProcessed = std::min( outputRowsThisCall - imageIndex * desc.Result.Height(),
			inputImageRowsInBuffer >= desc.Source.Height() ? desc.Result.Height()
				: ( inputImageRowsInBuffer < 2 ? 0 : 1 + ( inputImageRowsInBuffer - 2 ) / desc.StrideHeight ) );
		const int outputRowsCanBeProcessed = imageIndex * desc.Result.Height() + outputImageRowsCanBeProcessed;

		while( outputRowsProcessed < outputRowsCanBeProcessed ) {
			// Process channelwise output rows (while there are any)
			const int outputRowsThisStep = std::min<int>( maxOutputRowsPerStep, outputRowsCanBeProcessed - outputRowsProcessed );

			ProcessChannelwise3x3( desc, outputRowsThisStep, chInput->DataRows(), chInput->DataRowIndex() % desc.Source.Height(),
				channelwiseFilter, channelwiseFreeTerm, buffer, outputRowsProcessed % desc.Result.Height() );
			if( channelwiseActivation == AF_HSwish ) {
				vectorHSwish( buffer, buffer, outputRowsThisStep * chOutputRowSize );
			} else if( channelwiseActivation == AF_ReLU ) {
				if( channelwiseReluParam > 0 ) {
					vectorReLU( buffer, buffer, outputRowsThisStep * chOutputRowSize, channelwiseReluParam );
				} else {
					vectorReLU( buffer, buffer, outputRowsThisStep * chOutputRowSize );
				}
			}
			outputRowsProcessed += outputRowsThisStep;

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
		}

		if( outputRowsProcessed < desc.Result.ObjectCount() * desc.Result.Height() ) {
			const int firstInputRowNeeded = RowwiseConvFirstInputRow( outputRowsProcessed,
				desc.Source.Height(), desc.Result.Height(), desc.StrideHeight, desc.PaddingHeight );
			if( firstInputRowNeeded > chInput->DataRowIndex() ) {
				chInput->RemoveRows( firstInputRowNeeded - chInput->DataRowIndex() );
			}
		}
	}

	if( outputRowsProcessed == desc.Result.ObjectCount() * desc.Result.Height() ) {
		chInput.reset( nullptr );
	}

	return report;
}

} // namespace NeoML
