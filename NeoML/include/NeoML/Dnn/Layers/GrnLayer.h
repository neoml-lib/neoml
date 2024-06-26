/* Copyright © 2023-2024 ABBYY

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
#include <NeoML/NeoMLCommon.h>
#include <NeoML/Dnn/Dnn.h>
#include <NeoMathEngine/MemoryHandle.h>

namespace NeoML {

// Global Responce Normalization (GRN) layer from https://arxiv.org/pdf/2301.00808.pdf
// The PyTorch like pseudocode is
// X: input of shape (N, H, W, C)
//
//    gx = torch.norm(X, p=2, dim=(1,2), keepdim=True)
//    nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
//    return scale * (X * nx) + bias + X
//
// where scale and bias are trainable vectors of C elements
// WARNING: for now training and backward are not supported, only inference
class NEOML_API CGrnLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CGrnLayer )
public:
	explicit CGrnLayer( IMathEngine& mathEngine );

	// Returns or sets epsilon, added before division
	// By default, epsilon is equal to 1e-6
	// This value must be positive
	void SetEpsilon( float newEpsilon );
	float GetEpsilon() const { return epsilon; }

	// Returns or sets the scale
	// The blob should be of Channels size
	// May be null if the scale has not been initialized (or must be reset)
	CPtr<CDnnBlob> GetScale() const { return getParam( PN_Scale ); }
	void SetScale( const CPtr<CDnnBlob>& newScale ) { setParam( PN_Scale, newScale ); }

	// Returns or sets the bias
	// The blob should be of Channels size
	// May be null if the bias has not been initialized (or must be reset)
	CPtr<CDnnBlob> GetBias() const { return getParam( PN_Bias ); }
	void SetBias( const CPtr<CDnnBlob>& newBias ) { setParam( PN_Bias, newBias ); }

	void Serialize( CArchive& archive ) override;

protected:
	// CBaseLayer methods
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override { NeoAssert( false ); }

private:
	// The trainable parameters names
	enum TParamName {
		PN_Scale = 0,
		PN_Bias,

		PN_Count
	};

	float epsilon = 1e-6f; // default value from the article
	float invChannels = 0;

	void setParam( TParamName name, const CPtr<CDnnBlob>& newValue );
	CPtr<CDnnBlob> getParam( TParamName name ) const;

	CPtr<CDnnBlob>& scale() { return paramBlobs[PN_Scale]; }
	CPtr<CDnnBlob>& bias() { return paramBlobs[PN_Bias]; }
};

} // namespace NeoML
