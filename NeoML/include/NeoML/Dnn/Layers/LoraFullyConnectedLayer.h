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
class CDistributedTraining;

// Common interface for all LoRA layers to apply commands
class ILowRankAdapted {
public:
	// Return 0 if LoRA is not initialized
	virtual int GetRankLoRA() const = 0;
	// Return 0 if LoRA is not initialized
	virtual float GetAlphaLoRA() const = 0;
	// Return 0.f if LoRA dropout-layer is not initialized
	virtual float GetDropoutRateLoRA() const = 0;

	virtual void BuildLoRA( int rank, float alpha, float dropoutRate = 0.f ) = 0;
	virtual void MergeWeightsLoRA() = 0;
	virtual void DestroyUnmergedLoRA() = 0;
	virtual bool GetStoreSeparateLoRA() const = 0;
	virtual void SetStoreSeparateLoRA( bool storeSeparate ) = 0;
	virtual void SerializeWeightsLoRA( CArchive& ) = 0;
};

//-------------------------------------------------------------------------------------------------


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
class NEOML_API CLoraFullyConnectedLayer : public CCompositeLayer, public ILowRankAdapted {
	NEOML_DNN_LAYER( CLoraFullyConnectedLayer )
public:
	explicit CLoraFullyConnectedLayer( IMathEngine&, const char* name = nullptr );
	explicit CLoraFullyConnectedLayer( CFullyConnectedLayer& fc );

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
	int GetRankLoRA() const override { return loraRank; }
	// Return 0 if LoRA is not initialized
	float GetAlphaLoRA() const override { return loraAlpha; }
	// Return 0.f if LoRA dropout-layer is not initialized
	float GetDropoutRateLoRA() const override;

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
	void BuildLoRA( int rank, float alpha, float dropoutRate = 0.f ) override;
	// Enable learing of the original weights of baseFC.
	// Add the LoRA weight matrices to the original weights of baseFC to speed-up inference.
	// Delete all the LoRA layers (it cannot be undone)
	void MergeWeightsLoRA() override;
	// Enable learing of the original weights of baseFC.
	// Delete (if exist) all the LoRA layers
	// Discards LoRA weights
	void DestroyUnmergedLoRA() override { destroyLoRA(); }

	// if storeSeparate == true, this layer is serialized as old CFullyConnected (baseFc only),
	//     SerializeWeightsLoRA method is enabled.
	// if storeSeparate == false, this layer is serialized as new CLoraFullyConnected,
	//     with or without lora as it's built or not, SerializeWeightsLoRA method is disabled.
	bool GetStoreSeparateLoRA() const override  { return loraStoreSeparate; }
	// Set storeSeparate to true, before call the SerializeWeightsLoRA
	void SetStoreSeparateLoRA( bool storeSeparate ) override { loraStoreSeparate = storeSeparate; }

	// Store separate A and B matrices as arrays of floats into the file.
	// Alse it stores ( rank, alpha, dropoutRate ).
	// When Load separate matrices and recreate LoRA layers
	void SerializeWeightsLoRA( CArchive& ) override;

	// if storeSeparate == true, it serializes baseFc weights only,
	//     LoRA weights should be merged or destoroyed before,
	//     use SerializeWeightsLoRA method to serialize LoRA separately.
	// if storeSeparate == false, full current internal state would be serialized.
	void Serialize( CArchive& ) override;

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
	bool loraStoreSeparate = true; // by default stores LoRA separate

	void buildLayer();
	void destroyLoRA();
};

//-------------------------------------------------------------------------------------------------

NEOML_API CLayerWrapper<CLoraFullyConnectedLayer> LoraFullyConnected(
	int numberOfElements, bool zeroFreeTerm = false );

//-------------------------------------------------------------------------------------------------

// Apply a command for all sub-layers in the current composite layer only
template <typename TLoraCommand>
void LoraApplyCommand( CDnnLayerGraph& composite, TLoraCommand command, int existedLayers )
{
	CArray<char const*> layerNames;
	composite.GetLayerList( layerNames );

	int appliedLayers = 0;
	for( char const* layerName : layerNames ) {
		CBaseLayer* layer = composite.GetLayer( layerName ).Ptr();

		auto* loraLayer = dynamic_cast<ILowRankAdapted*>( layer );
		if( loraLayer != nullptr ) {
			command( *loraLayer );
			++appliedLayers;
		}
	}
	NeoAssert( appliedLayers == existedLayers );
}

//-------------------------------------------------------------------------------------------------

// Switch on/off learning for all layers except ILowRankAdapted in the dnn
// Enable learning  - set enable = true
// Disable learning - set enable = false
void NEOML_API LoraExceptSwitchLearnings( CDnnLayerGraph& graph, bool enable );

//-------------------------------------------------------------------------------------------------

// This special mechanism allows to serialize LoRA weights of CDnn only
class NEOML_API CLoraSerializer final {
public:
	// Serialize only LoRA-params, A and B LoRA-weights in the CDnn
	// Returns the number of LoRA layers whose weights were stored/loaded
	static int Serialize( CDnn&, CArchive& );
	// Serialize only oRA-params, A and B LoRA-weights in the CDistributedTraining
	// Returns the number of LoRA layers whose weights were stored/loaded
	static int Serialize( CDistributedTraining&, CArchive& );

	// LoRA checkpoint is serialized LoRA weights + solver (same as CDnn)
	static int SerializeCheckpoint( CDnn&, CArchive& );

private:
	// Lora implementations layers types
	enum TLoraLayerType {
		LLT_FullyConnected, // CLoraFullyConnectedLayer

		LLT_End,  // special value, end of archive
		LLT_Count // special value, count of types
	};

	static constexpr int loraSerializerVersion = 0;

	static int storeLora( CDnn&, CArchive& );
	static int loadLora( CDnn&, CArchive& );
};

} // namespace NeoML
