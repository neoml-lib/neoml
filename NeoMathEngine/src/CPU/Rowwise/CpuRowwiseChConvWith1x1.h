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
#include "../CpuMathEngineDnnChannelwiseConv.h"

namespace NeoML {

class CCpuMathEngine::CRowwiseChConvWith1x1 : public IRowwiseCpuImpl, public CRowwiseOperationDesc {
public:
	CRowwiseChConvWith1x1( CCpuMathEngine& mathEngine, int stride, const float* chFilter, const float* chFreeTerm,
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
		desc( 1, 1, stride, stride, CBlobDesc(), CBlobDesc(), CBlobDesc() )
	{
	}

	int MinInputRowCount() const override { return 3; }
	CBlobDesc Reshape( const CBlobDesc& inputSize ) override;
	int InOperationBufferSize() const override
		{ return desc.Result.Channels() * desc.Result.Width() * maxOutputRowsPerStep(); }
	int OutputRowCount() const override { return desc.Result.ObjectCount() * desc.Result.Height(); }
	int OutputRowSize() const override { return desc.Result.Width() * outputChannels; }
	bool IsTrivial() const override { return false; }
	CProcessingReport Process( const float* input, int inputRowIndex, int inputRowsAvailable,
		float* output, int outputRowIndex, int outputRowsAvailable, float* buffer ) const override;

private:
	CCpuMathEngine& mathEngine;
	const float* chFilter;
	const float* chFreeTerm;
	TActivationFunction activation;
	float reluParam;
	const float* convFilter;
	const float* convFreeTerm;
	int outputChannels;
	bool residual;
	CCommonChannelwiseConvolutionDesc desc;

	int maxOutputRowsPerStep() const;
};

inline CBlobDesc CCpuMathEngine::CRowwiseChConvWith1x1::Reshape( const CBlobDesc& inputSize )
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
	return outputSize;
}

inline int CCpuMathEngine::CRowwiseChConvWith1x1::maxOutputRowsPerStep() const
{
	const int maxRowSize = std::max( desc.Result.Channels(), desc.Source.Channels() ) * desc.Result.Width();
	return std::min( std::max( RowwiseCacheSize / maxRowSize, 1 ), desc.Result.Height() );
}

inline IRowwiseCpuImpl::CProcessingReport CCpuMathEngine::CRowwiseChConvWith1x1::Process( const float* input,
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

	const int outputRowsThisCall = outputRowIndex + report.OutputRowsCalculated;
	const int maxRowsPerStep = maxOutputRowsPerStep();
	const int chOutputRowSize = desc.Result.Channels() * desc.Result.Width();
	const int outputWidth = desc.Result.Width();
	const int inputChannels = desc.Source.Channels();

	const float* residualInput = input + ( outputRowIndex - inputRowIndex ) * desc.Source.Width() * desc.Source.Channels();
	while( outputRowIndex < outputRowsThisCall ) {
		// Process channelwise output rows (while there are any)
		const int outputImageRowIndex = outputRowIndex % desc.Result.Height();
		const int outputRowsThisStep = std::min( maxRowsPerStep,
			std::min( desc.Result.Height() - outputImageRowIndex, outputRowsThisCall - outputRowIndex ) );

		ProcessChannelwise3x3( desc, outputRowsThisStep, input, inputRowIndex % desc.Source.Height(),
			chFilter, chFreeTerm, buffer, outputImageRowIndex );

		if( activation == AF_HSwish ) {
			vectorHSwish( buffer, buffer, outputRowsThisStep * chOutputRowSize );
		} else if( activation == AF_ReLU ) {
			if( reluParam > 0 ) {
				vectorReLU( buffer, buffer, outputRowsThisStep * chOutputRowSize, reluParam );
			} else {
				vectorReLU( buffer, buffer, outputRowsThisStep * chOutputRowSize );
			}
		}

		mathEngine.multiplyMatrixByTransposedWithFreeTerm( buffer, outputRowsThisStep * outputWidth, inputChannels,
			convFilter, outputChannels, convFreeTerm, output );
		if( residual ) {
			vectorAdd( output, residualInput, output, outputRowsThisStep * outputWidth * outputChannels );
			residualInput += outputRowsThisStep * desc.Source.Width() * inputChannels;
		}

		output += outputRowsThisStep * outputChannels * outputWidth;
		outputRowIndex += outputRowsThisStep;

		if( outputRowIndex % desc.Result.Height() == 0 && outputRowIndex < outputRowsThisCall ) {
			// Switch to the next image in batch
			const int imageIndex = outputRowIndex / desc.Result.Height();
			const int diff = imageIndex * desc.Source.Height() - inputRowIndex;
			PRESUME_EXPR( diff >= 0 );
			input += diff * desc.Source.Width() * inputChannels;
			inputRowIndex += diff;
			inputRowsAvailable -= diff;
		}
	}

	return report;
}

} // namespace NeoML
