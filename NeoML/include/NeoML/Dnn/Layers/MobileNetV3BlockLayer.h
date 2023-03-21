/* Copyright © 2017-2023 ABBYY Production LLC

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

// Emulates the part of the block which goes after Squeeze-and-Excite
//
// 2 or 3 inputs:
//     1. Channelwise conv result
//     2. Squeeze-and-Excite result
//     3. (optional) Residual input
//
// Replaces the follwoing construction:
//     Mul -> Activation -> DownConv -> [Residual ->]
// Possible activations: ReLU and HSwish
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
	CPtr<CDnnBlob> DownFilter() const { return getParamBlob( P_DownFilter ); }
	CPtr<CDnnBlob> DownFreeTerm() const { return getParamBlob( P_DownFreeTerm ); }
	void SetDownFilter( const CPtr<CDnnBlob>& value ) { setParamBlob( P_DownFilter, value ); }
	void SetDownFreeTerm( const CPtr<CDnnBlob>& value ) { setParamBlob( P_DownFreeTerm, value ); }

	// Serialization
	void Serialize( CArchive& archive ) override;

protected:
	// CBaseLayer methods
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override { NeoAssert( false ); }

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

	CActivationDesc activation;

	CPtr<CDnnBlob> getParamBlob( TParam param ) const;
	void setParamBlob( TParam param, const CPtr<CDnnBlob>& blob );
};

} // namespace NeoML
