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
#include <CpuMathEngineDnnPooling.h>
#include <CpuMathEngine.h>

namespace NeoML {

class CCpuMathEngine::CCpuRowwise2DPooling : public ICpuRowwiseImpl, public CRowwiseOperationDesc {
public:
	CCpuRowwise2DPooling( CCpuMathEngine& mathEngine, bool isMax, int filterHeight, int filterWidth,
			int strideHeight, int strideWidth ) :
		mathEngine( mathEngine ),
		isMax( isMax ),
		desc( CBlobDesc(), CBlobDesc(), filterHeight, filterWidth, strideHeight, strideWidth )
	{
	}

	CBlobDesc Reshape( const CBlobDesc& inputSize ) override;
	int InputRowRequirement() const override { return desc.FilterHeight; }
	int OutputRowRequirement() const override { return 0; }
	int InOperationBufferSize() const override { return desc.Source.Width() * desc.Source.Depth() * desc.Source.Channels(); }
	int OutputRowCount() const override { return desc.Result.ObjectCount() * desc.Result.Height(); }
	int OutputRowSize() const override { return desc.Result.Width() * desc.Result.Depth() * desc.Result.Channels(); }
	bool IsTrivial() const override { return false; }
	CProcessingReport Process( const float* input, int inputRowIndex, int inputRowsAvailable,
		float* output, int outputRowIndex, int outputRowsAvailable, float* buffer ) const override;

private:
	CCpuMathEngine& mathEngine;
	bool isMax;
	CCommon2DPoolingDesc desc;
};

inline CBlobDesc CCpuMathEngine::CCpuRowwise2DPooling::Reshape( const CBlobDesc& inputSize )
{
	auto poolOutputSize = [] ( int input, int filter, int stride ) -> int
	{
		return 1 + ( input - filter ) / stride;
	};

	desc.Source = inputSize;
	desc.Result = desc.Source;
	desc.Result.SetDimSize( BD_Height, poolOutputSize( inputSize.Height(), desc.FilterHeight, desc.StrideHeight ) );
	desc.Result.SetDimSize( BD_Width, poolOutputSize( inputSize.Width(), desc.FilterWidth, desc.StrideWidth ) );

	return desc.Result;
}

inline ICpuRowwiseImpl::CProcessingReport CCpuMathEngine::CCpuRowwise2DPooling::Process( const float* input,
	int inputRowIndex, int inputRowsAvailable, float* output, int outputRowIndex, int outputRowsAvailable,
	float* buffer ) const
{
	CProcessingReport report = RowwiseConvProcessingReport( inputRowIndex, inputRowsAvailable, outputRowIndex,
		outputRowsAvailable, desc.Source.Height(), desc.Result.Height(), desc.FilterHeight, 0,
		desc.StrideHeight, 1 );
	if( report.OutputRowsCalculated == 0 ) {
		return report;
	}

	if( isMax ) {
		mathEngine.blobMaxPoolingWithoutIndices( desc, report.OutputRowsCalculated,
			input, inputRowIndex, output, outputRowIndex, buffer );
	} else {
		blobMeanPooling( desc, report.OutputRowsCalculated, input, inputRowIndex,
			output, outputRowIndex, buffer );
	}

	return report;
}

} // namespace NeoML
