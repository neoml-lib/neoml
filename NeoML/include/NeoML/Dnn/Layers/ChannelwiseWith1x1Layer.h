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

// This layer emulates the following chain of the layers
//     channelwise3x3p1s1/2 -> activation -> conv1x1p0s1
// or the construction above with residual connection
//     -+--> block without residual ----> sum ->
//      |                                  |
//      +----------------------------------+
class NEOML_API CChannelwiseWith1x1Layer : public CBaseLayer {
public:
	CChannelwiseWith1x1Layer( IMathEngine& mathEngine, int stride, const CPtr<CDnnBlob>& channelwiseFilter,
		const CPtr<CDnnBlob>& channelwiseFreeTerm, const CActivationDesc& activation,
		const CPtr<CDnnBlob>& convFilter, const CPtr<CDnnBlob>& convFreeTerm, bool residual );
	explicit CChannelwiseWith1x1Layer( IMathEngine& mathEngine );
	~CChannelwiseWith1x1Layer();

	// Channelwise convolution and activation parameters
	int Stride() const { return stride; }
	CPtr<CDnnBlob> ChannelwiseFilter() const;
	CPtr<CDnnBlob> ChannelwiseFreeTerm() const;
	CActivationDesc Activation() const { return activation; }

	// 1x1 convolution parameters
	CPtr<CDnnBlob> ConvFilter() const;
	CPtr<CDnnBlob> ConvFreeTerm() const;

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
		{ return paramBlobs[i] == nullptr && ( i == P_ChannelwiseFreeTerm || i == P_ConvFreeTerm ); }

private:
	// paramBlobs indices
	enum TParam {
		P_ChannelwiseFilter,
		P_ChannelwiseFreeTerm,
		P_ConvFilter,
		P_ConvFreeTerm,

		P_Count
	};

	int stride; // stride of channnelwise convolution
	CActivationDesc activation; // activation after channelwise convolution
	bool residual; // Does block have residual connection?
	CChannelwiseConvolutionDesc* convDesc = nullptr; // descriptor of channelwise convolution
	CRowwiseOperationDesc* rowwiseDesc = nullptr; // matrix multiplication optimization

	void recreateConvDesc();
	void recreateRowwiseDesc();
};

} // namespace NeoML
