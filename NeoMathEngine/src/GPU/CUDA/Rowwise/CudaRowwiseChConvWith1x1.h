/* Copyright © 2023 ABBYY

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

#include "CudaRowwiseInterface.h"
#include "../CudaMathEngine.h"

namespace NeoML {

class CCudaRowwiseChConvWith1x1 : public ICudaRowwiseImpl, public CRowwiseOperationDesc {
public:
	CCudaRowwiseChConvWith1x1( int stride, const CConstFloatHandle& chFilter, const CConstFloatHandle* chFreeTerm,
			TActivationFunction activation, float reluParam, const CConstFloatHandle& convFilter,
			const CConstFloatHandle* convFreeTerm, int outputChannels, bool residual ) :
		stride( stride ),
		chFilter( chFilter ),
		chFreeTerm( chFreeTerm == nullptr ? CConstFloatHandle() : *chFreeTerm ),
		activation( activation ),
		reluParam( reluParam ),
		convFilter( convFilter ),
		convFreeTerm( convFreeTerm == nullptr ? CConstFloatHandle() : *convFreeTerm ),
		outputChannels( outputChannels ),
		residual( residual )
	{
	}

	// ICudaRowwiseImpl
	CBlobDesc Reshape( const CBlobDesc& inputSize ) override;
	int OutputSize() const override { return outputDesc.BlobSize(); }
	bool IsInPlace() const override { return false; }
	void Process( const CFloatHandle& input, const CFloatHandle& output ) const override;

private:
	const int stride;
	const CConstFloatHandle chFilter;
	const CConstFloatHandle chFreeTerm;
	const TActivationFunction activation;
	const float reluParam;
	const CConstFloatHandle convFilter;
	const CConstFloatHandle convFreeTerm;
	const int outputChannels;
	const bool residual;
	CBlobDesc inputDesc;
	CBlobDesc outputDesc;
	std::unique_ptr<CChannelwiseConvolutionDesc> chDesc;
};

//---------------------------------------------------------------------------------------------------------------------

inline CBlobDesc CCudaRowwiseChConvWith1x1::Reshape( const CBlobDesc& inputSize )
{
	inputDesc = inputSize;
	outputDesc = inputSize;
	outputDesc.SetDimSize( BD_Height, 1 + ( outputDesc.Height() - 1 ) / stride );
	outputDesc.SetDimSize( BD_Width, 1 + ( outputDesc.Width() - 1 ) / stride );
	CBlobDesc filterSize( { 3, 3, 1, inputSize.Channels() } );
	CBlobDesc freeTermSize( { inputSize.Channels() } );
	IMathEngine& mathEngine = *chFilter.GetMathEngine();
	chDesc.reset( mathEngine.InitBlobChannelwiseConvolution( inputSize, 1, 1, stride, stride,
		filterSize, chFreeTerm.IsNull() ? nullptr : &freeTermSize, outputDesc ) );
	outputDesc.SetDimSize( BD_Channels, outputChannels );
	return outputDesc;
}

inline void CCudaRowwiseChConvWith1x1::Process( const CFloatHandle& input, const CFloatHandle& output ) const
{
	chFilter.GetMathEngine()->ChannelwiseWith1x1( inputDesc, outputDesc, *chDesc, input, chFilter,
		chFreeTerm.IsNull() ? nullptr : &chFreeTerm, activation, reluParam, convFilter,
		convFreeTerm.IsNull() ? nullptr : &convFreeTerm, residual, output );
}

} // namespace NeoML
