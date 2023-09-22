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

class CCudaRowwiseMobileNetV2 : public ICudaRowwiseImpl, public CRowwiseOperationDesc {
public:
	CCudaRowwiseMobileNetV2( const CConstFloatHandle& expandFilter,
			const CConstFloatHandle* expandFreeTerm, int expandedChannels, TActivationFunction expandActivation,
			float expandReluParam, const CConstFloatHandle& channelwiseFilter,
			const CConstFloatHandle* channelwiseFreeTerm, int stride, TActivationFunction channelwiseActivation,
			float channelwiseReluParam, const CConstFloatHandle& downFilter, const CConstFloatHandle* downFreeTerm,
			int outputChannels, bool residual ) :
		expandFilter( expandFilter ),
		expandFreeTerm( expandFreeTerm == nullptr ? CConstFloatHandle() : *expandFreeTerm ),
		expandedChannels( expandedChannels ),
		expandActivation( expandActivation ),
		expandReluParam( expandReluParam ),
		channelwiseFilter( channelwiseFilter ),
		channelwiseFreeTerm( channelwiseFreeTerm == nullptr ? CConstFloatHandle() : *channelwiseFreeTerm  ),
		stride( stride ),
		channelwiseActivation( channelwiseActivation ),
		channelwiseReluParam( channelwiseReluParam ),
		downFilter( downFilter ),
		downFreeTerm( downFreeTerm == nullptr ? CConstFloatHandle() : *downFreeTerm ),
		outputChannels( outputChannels ),
		residual( residual )
	{
	}

	// ICudaRowwiseImpl
	CBlobDesc Reshape( const CBlobDesc& inputSize ) override;
	int OutputSize() const override { return outputDesc.BlobSize(); }
	bool IsInPlace() const override { return supportsInPlace; }
	void Process( const CFloatHandle& input, const CFloatHandle& output ) const override;

private:
	const CConstFloatHandle expandFilter;
	const CConstFloatHandle expandFreeTerm;
	const int expandedChannels;
	const TActivationFunction expandActivation;
	const float expandReluParam;
	const CConstFloatHandle channelwiseFilter;
	const CConstFloatHandle channelwiseFreeTerm;
	const int stride;
	const TActivationFunction channelwiseActivation;
	const float channelwiseReluParam;
	const CConstFloatHandle downFilter;
	const CConstFloatHandle downFreeTerm;
	const int outputChannels;
	const bool residual;
	// Reshape
	CBlobDesc inputDesc{};
	CBlobDesc outputDesc{};
	std::unique_ptr<CChannelwiseConvolutionDesc> chDesc{};
	bool supportsInPlace = false;

	friend class CCudaMathEngine;
};

//---------------------------------------------------------------------------------------------------------------------

inline CBlobDesc CCudaRowwiseMobileNetV2::Reshape( const CBlobDesc& inputSize )
{
	inputDesc = inputSize;
	const int inputChannels = inputDesc.Channels();
	inputDesc.SetDimSize( BD_Channels, expandedChannels );

	outputDesc = inputDesc;
	outputDesc.SetDimSize( BD_Height, 1 + ( outputDesc.Height() - 1 ) / stride );
	outputDesc.SetDimSize( BD_Width, 1 + ( outputDesc.Width() - 1 ) / stride );

	CBlobDesc filterDesc( { 3, 3, 1, expandedChannels } );
	CBlobDesc freeTermDesc( { expandedChannels } );
	IMathEngine& mathEngine = *expandFilter.GetMathEngine();
	chDesc.reset( mathEngine.InitBlobChannelwiseConvolution( inputDesc, 1, 1, stride, stride,
		filterDesc, channelwiseFreeTerm.IsNull() ? nullptr : &freeTermDesc, outputDesc ) );
	inputDesc.SetDimSize( BD_Channels, inputChannels );
	outputDesc.SetDimSize( BD_Channels, outputChannels );

	supportsInPlace = inputSize.HasEqualDimensions( outputDesc );

	return outputDesc;
}

inline void CCudaRowwiseMobileNetV2::Process( const CFloatHandle& input, const CFloatHandle& output ) const
{
	expandFilter.GetMathEngine()->MobileNetV2Block( inputDesc, outputDesc, *this, *chDesc, input, output );
}

} // namespace NeoML
