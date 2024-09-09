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

// This layer computes a block in a MobileNetV2 architecture
//
// The block may be without residual connection
//     conv1x1 (expand) -> expandActivation -> channelwiseConv3x3 -> chanenlwiseActivation -> conv1x1 (down)
// or it may be with residual connection
//     -+--> block without residual ----> sum ->
//      |                                  |
//      +----------------------------------+
//
// This layer is faster and consumes less memory than the composite of layers but it has some restrictions:
//     - this layer is untrainable
//     - all 1x1 convolutions must have no paddings and stride == 1
//     - channelwise convolution must have stride 1 or 2, padding == 1 and dilation == 1
//     - only ReLU and HSwish activations are supported (upper thresholds in ReLU are supported)
//     - free terms are supported (but may be nullptr) for all the convolutions in nblock
class NEOML_API CMobileNetV2BlockLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CMobileNetV2BlockLayer )
public:
	CMobileNetV2BlockLayer( IMathEngine& mathEngine, const CPtr<CDnnBlob>& expandFilter,
		const CPtr<CDnnBlob>& expandFreeTerm, const CActivationDesc& expandActivation, int stride,
		const CPtr<CDnnBlob>& channelwiseFilter, const CPtr<CDnnBlob>& channelwiseFreeTerm,
		const CActivationDesc& channelwiseActivation, const CPtr<CDnnBlob>& downFilter,
		const CPtr<CDnnBlob>& downFreeTerm, bool residual );
	explicit CMobileNetV2BlockLayer( IMathEngine& mathEngine );
	~CMobileNetV2BlockLayer();

	// Expand convolution and activation parameters
	CPtr<CDnnBlob> ExpandFilter() const;
	CPtr<CDnnBlob> ExpandFreeTerm() const;
	CActivationDesc ExpandActivation() const { return expandActivation; }

	// Channelwise convolution and activation parameters
	int Stride() const { return stride; }
	CPtr<CDnnBlob> ChannelwiseFilter() const;
	CPtr<CDnnBlob> ChannelwiseFreeTerm() const;
	CActivationDesc ChannelwiseActivation() const { return channelwiseActivation; }

	// Down convolution parameters
	CPtr<CDnnBlob> DownFilter() const;
	CPtr<CDnnBlob> DownFreeTerm() const;

	// Residual connection
	bool Residual() const { return residual; }
	void SetResidual( bool newValue );

	// Serialization
	void Serialize( CArchive& archive ) override;

protected:
	// CBaseLayer methods
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override { NeoAssert( false ); }
	// Specialization for transferParamsBlob
	bool ContainsNullParamBlob( int i ) const override
		{ return !paramBlobs[i] && ( i == P_ChannelwiseFreeTerm || i == P_DownFreeTerm || i == P_ExpandFreeTerm ); }

private:
	// paramBlobs indices
	enum TParam {
		P_ExpandFilter,
		P_ExpandFreeTerm,
		P_ChannelwiseFilter,
		P_ChannelwiseFreeTerm,
		P_DownFilter,
		P_DownFreeTerm,

		P_Count
	};

	void recreateConvDesc();
	void recreateRowwiseDesc();

	bool residual; // Does block have residual connection?
	int stride; // stride of channnelwise convolution
	CActivationDesc expandActivation; // expand convolution activation
	CActivationDesc channelwiseActivation; // channelwise convolution activation
	CChannelwiseConvolutionDesc* convDesc = nullptr; // descriptor of channelwise convolution
	CRowwiseOperationDesc* rowwiseDesc = nullptr; // descriptor of rowwise operation
};

} // namespace NeoML
