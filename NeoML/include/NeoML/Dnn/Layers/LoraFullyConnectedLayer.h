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

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Layers/CompositeLayer.h>
#include <NeoML/Dnn/Layers/FullyConnectedLayer.h>

namespace NeoML {

// Forward declarations
class CLinearLayer;
class CDropoutLayer;
class CEltwiseSumLayer;

// Fully Connected Layer with Low Rank Adaptation implements
// https://arxiv.org/pdf/2106.09685v2.pdf
// Substitute FullyConnectedLayer
//
//        loraSum
//           ^
//           |
//      +---------+
//      ^         ^
//      |         |
//      |     loraScaling
//      |         |
//      |      loraFcB
//   baseFc       |
//      |      loraFcA
//      |         |
//      |     (dropout) 
//      |         |
//      ^         ^
//      +---------+
//           ^
//           |
//       inputData
//
class NEOML_API CLoraFullyConnectedLayer : public CCompositeLayer {
	NEOML_DNN_LAYER( CLoraFullyConnectedLayer )
public:
	explicit CLoraFullyConnectedLayer( IMathEngine&, const char* name = nullptr );
	explicit CLoraFullyConnectedLayer( CFullyConnectedLayer& fc );

	void Serialize( CArchive& ) override;

	// Public interface of the FullyConnectedLayer to exchange these classes

	int GetNumberOfElements() const { return baseFc->GetNumberOfElements(); }
	void SetNumberOfElements( int numberOfElements ) { baseFc->SetNumberOfElements( numberOfElements ); }

	CPtr<CDnnBlob> GetWeightsData() const { return baseFc->GetWeightsData(); }
	void SetWeightsData( const CDnnBlob* weights ) { baseFc->SetWeightsData( weights ); }

	CPtr<CDnnBlob> GetFreeTermData() const { return baseFc->GetFreeTermData(); }
	void SetFreeTermData( const CDnnBlob* freeTerm ) { baseFc->SetFreeTermData( freeTerm ); }

	bool IsZeroFreeTerm() const { return baseFc->IsZeroFreeTerm(); }
	void SetZeroFreeTerm( bool zeroFreeTerm ) { baseFc->SetZeroFreeTerm( zeroFreeTerm ); }

	CPtr<CDnnBlob>& Weights() { return baseFc->Weights(); }
	CPtr<CDnnBlob>& FreeTerms() { return baseFc->FreeTerms(); }

	const CPtr<CDnnBlob>& Weights() const { return baseFc->Weights(); }
	const CPtr<CDnnBlob>& FreeTerms() const { return baseFc->FreeTerms(); }

	// Public interface of the LoRA

	// Return 0 if LoRA is not initialized
	int GetRankLoRA() const { return loraRank; }
	// Return 0 if LoRA is not initialized
	float GetAlphaLoRA() const { return loraAlpha; }
	// Return 0.f if LoRA dropout-layer is not initialized
	float GetDropoutRateLoRA() const;

	const CPtr<CDnnBlob> AWeightsLoRA() const { return ( loraFcA == nullptr ) ? nullptr : loraFcA->Weights(); }
	void SetAWeightsLoRAData( const CDnnBlob* weights );

	const CPtr<CDnnBlob> BWeightsLoRA() const { return ( loraFcB == nullptr ) ? nullptr : loraFcB->Weights(); }
	void SetBWeightsLoRAData( const CDnnBlob* weights );

	// Disable learning of the original weights of baseFC.
	// Alloc LoRA weight-matrices.
	// Append an optinal dropout-layer before LoRA full connected layers, if `dropoutRate` > 0.
	// The `rank` should be > 0 and `rank` << min(baseFC->Weights()->GetObjectSize(), baseFC->GetNumberOfElements())
	// Append linear-layer after LoRA full connected layers to set the scaling value == ( `alpha` / `rank` )
	// For more details see the original article
	void BuildLoRA( int rank, float alpha, float dropoutRate = 0.f );
	// Enable learing of the original weights of baseFC.
	// Add the LoRA weight matrices to the original weights of baseFC to speed-up inference.
	// Delete all the LoRA layers (it cannot be undone)
	void MergeWeightsLoRA();
	// Enable learing of the original weights of baseFC.
	// Delete (if exist) all the LoRA layers
	// Discards LoRA weights
	void DestroyUnmergedLoRA() { destroyLoRA(); }

protected:
	~CLoraFullyConnectedLayer() override = default;

	void Reshape() override;

private:
	CPtr<CFullyConnectedLayer> baseFc;
	// LoRA
	CPtr<CFullyConnectedLayer> loraFcA;
	CPtr<CFullyConnectedLayer> loraFcB;
	CPtr<CDropoutLayer> loraDropout;
	CPtr<CLinearLayer> loraScaling;
	CPtr<CEltwiseSumLayer> loraSum;

	int loraRank = 0; // by default LoRA isn't constructed
	float loraAlpha = 0; // by default set it to loraRank value
	float loraDropoutRate = 0; // by default dropout layer isn't constructed

	void buildLayer();
	void destroyLoRA();
};

//-------------------------------------------------------------------------------------------------

NEOML_API CLayerWrapper<CLoraFullyConnectedLayer> LoraFullyConnected(
	int numberOfElements, bool zeroFreeTerm = false );

} // namespace NeoML
