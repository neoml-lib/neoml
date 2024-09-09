/* Copyright © 2017-2024 ABBYY

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

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Dnn.h>
#include <NeoML/Dnn/Layers/ActivationLayers.h>

namespace NeoML {

// Emulates the part of the block which goes before Squeeze-and-Excite
//
// The only input accepts the input of the block
// The only output contains the output of channelwise convolution
//
// Replaces the following construction:
//     ExpandConv -> Activation -> Channelwise [-> Activation]
//
// Possible activations: ReLU and HSwish (or trivial Linear{mul=1, ft=0})
// Channelwise support stride 1 and 2, and filters 3x3 or 5x5 with paddings 1 and 2 correspondingly
class NEOML_API CMobileNetV3PreSEBlockLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CMobileNetV3PreSEBlockLayer )
public:
	CMobileNetV3PreSEBlockLayer( IMathEngine& mathEngine, const CPtr<CDnnBlob>& expandFilter,
		const CPtr<CDnnBlob>& expandFreeTerm, const CActivationDesc& expandActivation, int stride,
		const CPtr<CDnnBlob>& channelwiseFilter, const CPtr<CDnnBlob>& channelwiseFreeTerm,
		const CActivationDesc& channelwiseActivation );
	explicit CMobileNetV3PreSEBlockLayer( IMathEngine& mathEngine );
	~CMobileNetV3PreSEBlockLayer() override;

	// Expand convolution and activation parameters
	CPtr<CDnnBlob> ExpandFilter() const;
	CPtr<CDnnBlob> ExpandFreeTerm() const;
	CActivationDesc ExpandActivation() const { return expandActivation; }

	// Channelwise convolution
	int Stride() const { return stride; }
	CPtr<CDnnBlob> ChannelwiseFilter() const;
	CPtr<CDnnBlob> ChannelwiseFreeTerm() const;
	CActivationDesc ChannelwiseActivation() const { return channelwiseActivation; }

	void Serialize( CArchive& archive ) override;

private:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override { NeoAssert( false ); }
	// Specialization for transferParamsBlob
	bool ContainsNullParamBlob( int i ) const override
		{ return paramBlobs[i] == nullptr && ( i == P_ChannelwiseFreeTerm || i == P_ExpandFreeTerm ); }

private:
	// paramBlobs indices
	enum TParam {
		P_ExpandFilter,
		P_ExpandFreeTerm,
		P_ChannelwiseFilter,
		P_ChannelwiseFreeTerm,

		P_Count
	};

	CActivationDesc expandActivation; // activation applied after expand 1x1 convolution
	int stride; // stride of channelwise convolution
	CActivationDesc channelwiseActivation; // activation applied after channelwise convolution
	CChannelwiseConvolutionDesc* convDesc; // descriptor of channelwise convolution
};

// Emulates the part of the block which goes after Squeeze-and-Excite
//
// 2 or 3 inputs:
//     1. Channelwise conv result
//     2. Squeeze-and-Excite result
//     3. (optional) Residual input
//
// Replaces the following construction:
//     Mul [-> Activation] -> DownConv -> [Residual ->]
//
// Possible activations: ReLU and HSwish (or trivial Linear{mul=1, ft=0})
class NEOML_API CMobileNetV3PostSEBlockLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CMobileNetV3PostSEBlockLayer )
public:
	CMobileNetV3PostSEBlockLayer( IMathEngine& mathEngine, const CActivationDesc& activation,
		const CPtr<CDnnBlob>& downFilter, const CPtr<CDnnBlob>& downFreeTerm );
	explicit CMobileNetV3PostSEBlockLayer( IMathEngine& mathEngine );

	// Activation
	// Applied on the result of Mul(ChannelwiseConv, Squeeze-and-Excite)
	const CActivationDesc& Activation() { return activation; }
	void SetActivation( const CActivationDesc& newActivation ) { activation = newActivation; }

	// Down convolution parameters
	CPtr<CDnnBlob> DownFilter() const;
	CPtr<CDnnBlob> DownFreeTerm() const;

	// Serialization
	void Serialize( CArchive& archive ) override;

protected:
	// CBaseLayer methods
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override { NeoAssert( false ); }
	// Specialization for transferParamsBlob
	bool ContainsNullParamBlob( int i ) const override
		{ return paramBlobs[i] == nullptr && ( i == P_DownFreeTerm ); }

private:
	// paramBlobs indices
	enum TParam {
		P_DownFilter,
		P_DownFreeTerm,

		P_Count
	};

	// input indices
	enum TInput {
		I_Channelwise,
		I_SqueezeAndExcite,
		I_ResidualInput,

		I_Count
	};

	CActivationDesc activation; // activation applied after channelwise convolution
};

} // namespace NeoML
