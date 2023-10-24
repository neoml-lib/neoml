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
#include <NeoML/Dnn/Layers/ActivationLayers.h>
#include <NeoML/Dnn/Layers/CompositeLayer.h>
#include <NeoML/Dnn/Layers/DropoutLayer.h>
#include <NeoML/Dnn/Layers/EltwiseLayer.h>
#include <NeoML/Dnn/Layers/FullyConnectedLayer.h>

namespace NeoML {

struct NEOML_API CLoraParams {
	int Rank; // Size of vector in-between A and B matrices of LoRA
	float Alpha; // Coefficient, the output will be multiplied by Alpha / Rank
	float Dropout; // Dropout applied to input before matrix multiplications

	explicit CLoraParams( int rank = 1, float alpha = 1.f, float dropout = 0.f )
		: Rank( rank ), Alpha( alpha ), Dropout( dropout ) {}

	void Serialize( CArchive& archive );
};

// Fully Connected Layer with Low Rank Adaptation implements
// https://arxiv.org/pdf/2106.09685v2.pdf
// LoRA wrapper for CFullyConnectedLayer
//
// This layer lazily switches between 2 states
//
// 1st: "split" state
// In this state baseFc contains unmodified weights from the original network
// the LoRA part is present explicitly as a group of layers
// Used during training
//
//        loraSum
//           ^
//           |
//      +---------+
//      ^         ^
//      |         |
//      |      scaling
//      |         |
//      |        fcB
//   baseFc       |
//      |        fcA
//      |         |
//      |      dropout
//      |         |
//      ^         ^
//      +---------+
//           ^
//           |
//       inputData
//
// 2nd: "merged" state
// In this state layer baseFc contains weights which emulate full LoRA
// and no other layers are present in the composite
// Used during inference
//
//        loraSum
//           ^
//           |
//         baseFc
//           ^
//           |
//       inputData
//
// NOTE: even in the merged state this layer has to store A and B matrices
// in order to switch back to "split" state when needed
// If you need only inference then you can replace this layer with CFullyConnected with merged weights
// (this can be done via CLoraBuilder::MergeFcWrapper)
class NEOML_API CLoraFullyConnectedLayer : public CCompositeLayer {
	NEOML_DNN_LAYER( CLoraFullyConnectedLayer )
public:
	CLoraFullyConnectedLayer( CDnnBlob& baseWeights, CDnnBlob* baseFreeTerms, const CLoraParams& params );
	explicit CLoraFullyConnectedLayer( IMathEngine& mathEngine ); // used for loading serialized layer

	void Serialize( CArchive& ) override;

	void UpdateParams( const CLoraParams& newParams, CDnnBlob* newA, CDnnBlob* newB );

	int OutputSize() const { return baseFc->GetNumberOfElements(); }
	int Rank() const { return fcA->GetNumberOfElements(); }
	float Alpha() const { return scaling->GetMultiplier() * Rank(); }
	float Dropout() const { return dropout->GetDropoutRate(); }

	// Raw getters for weights
	// These getters do not copy weights which may lead to difficult-to-debug troubles
	// But they're necessary for making LoRA work without excessive copying
	// baseFc weights from "split" state
	CPtr<CDnnBlob> GetSplitWeightsNoCopy() { split(); return baseFc->Weights(); }
	// baseFc weights from "merged" state
	CPtr<CDnnBlob> GetMergedWeightsNoCopy() { merge(); return baseFc->Weights(); }
	// baseFc free terms
	CPtr<CDnnBlob> GetFreeTermsNoCopy() { return baseFc->FreeTerms(); }
	// A LoRA matrix
	CPtr<CDnnBlob> GetAWeightsNoCopy() { return fcA->Weights(); }
	// B LoRA matrix
	CPtr<CDnnBlob> GetBWeightsNoCopy() { return fcB->Weights(); }

	// Mostly for testing/debugging
	CPtr<CDnnBlob>& GetWeightsNoCopy() { return baseFc->Weights(); }
	bool IsMerged() const { return isMerged; }

protected:
	~CLoraFullyConnectedLayer() override = default;

	void Reshape() override;

private:
	bool isMerged = true;
	CPtr<CFullyConnectedLayer> baseFc;
	CPtr<CDropoutLayer> dropout;
	CPtr<CFullyConnectedLayer> fcA;
	CPtr<CFullyConnectedLayer> fcB;
	CPtr<CLinearLayer> scaling;
	CPtr<CEltwiseSumLayer> sum;

	void initialize( const CLoraParams& params );
	void merge();
	void split();
	void recalcBaseWeights();
};

} // namespace NeoML
