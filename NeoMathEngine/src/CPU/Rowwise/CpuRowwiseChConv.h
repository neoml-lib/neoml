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

namespace NeoML {

class CRowwiseChConv : public IRowwiseCpuImpl, public CRowwiseOperationDesc {
public:
	CRowwiseChConv( int paddingHeight, int paddingWidth, int strideHeight, int strideWidth,
			const CBlobDesc& filterDesc, const float* filter, const float* freeTerm ) :
		desc( paddingHeight, paddingWidth, strideHeight, strideWidth, CBlobDesc(), filterDesc, CBlobDesc() ),
		processFunc( nullptr ),
		filter( filter ),
		freeTerm( freeTerm )
	{
	}

	int MinInputRowCount() const override { return desc.Filter.Height(); }
	CBlobDesc Reshape( const CBlobDesc& inputSize ) override;
	int InOperationBufferSize() const override { return 0; }
	int OutputRowCount() const override { return desc.Result.ObjectCount() * desc.Result.Height(); }
	int OutputRowSize() const override { return desc.Result.Width() * desc.Result.Depth() * desc.Result.Channels(); }
	bool IsTrivial() const override { return false; }
	CProcessingReport Process( const float* input, int inputRowIndex, int inputRowsAvailable,
		float* output, int outputRowIndex, int outputRowsAvailable, float* buffer ) const override;

private:
	CCommonChannelwiseConvolutionDesc desc;
	TChannelwiseProcessFunction processFunc;
	const float* filter;
	const float* freeTerm;
};

inline CBlobDesc CRowwiseChConv::Reshape( const CBlobDesc& inputSize )
{
	auto convOutputSize = [] ( int input, int filter, int padding, int stride ) -> int
	{
		return 1 + ( input + 2 * padding - filter ) / stride;
	};

	desc.Source = inputSize;
	desc.Result = desc.Source;
	desc.Result.SetDimSize( BD_Height, convOutputSize( inputSize.Height(), desc.Filter.Height(),
		desc.PaddingHeight, desc.StrideHeight ) );
	desc.Result.SetDimSize( BD_Width, convOutputSize( inputSize.Width(), desc.Filter.Width(),
		desc.PaddingWidth, desc.StrideWidth ) );
	processFunc = GetChannelwiseProcessFunction( desc );

	return desc.Result;
}

inline IRowwiseCpuImpl::CProcessingReport CRowwiseChConv::Process( const float* input, int inputRowIndex,
	int inputRowsAvailable, float* output, int outputRowIndex, int outputRowsAvailable, float* buffer ) const
{
	CProcessingReport report = RowwiseConvProcessingReport( inputRowIndex, inputRowsAvailable, outputRowIndex,
		outputRowsAvailable, desc.Source.Height(), desc.Result.Height(), desc.Filter.Height(), desc.PaddingHeight,
		desc.StrideHeight, 1 );
	if( report.OutputRowsCalculated == 0 ) {
		PRESUME_EXPR( report.InputRowsMayBeRemoved == 0 );
		return report;
	}

	int imageIndex = outputRowIndex / desc.Result.Height();
	const int outputRowsAfterThisCall = outputRowIndex + report.OutputRowsCalculated;
	const int inputRowSize = desc.Source.Width() * desc.Source.Depth() * desc.Source.Channels();
	const int outputRowSize = desc.Result.Width() * desc.Result.Depth() * desc.Result.Channels();
	const int inputHeight = desc.Source.Height();
	const int outputHeight = desc.Result.Height();

	while( outputRowIndex < outputRowsAfterThisCall ) {
		if( imageIndex * desc.Source.Height() > inputRowIndex ) {
			const int diff = imageIndex * desc.Source.Height() - inputRowIndex;
			inputRowIndex += diff;
			input += diff * inputRowSize;
		}

		const int outputRowsThisStep = std::min( outputRowsAfterThisCall, ( imageIndex + 1 ) * desc.Result.Height() )
			- outputRowIndex;
		processFunc( desc, outputRowsThisStep, input, inputRowIndex % inputHeight, filter, freeTerm,
			output, outputRowIndex % outputHeight );

		outputRowIndex += outputRowsThisStep;
		output += outputRowsThisStep * outputRowSize;
		imageIndex = outputRowIndex / outputHeight;
	}

	return report;
}

} // namespace NeoML
