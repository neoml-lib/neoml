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
#include <NeoML/Dnn/DnnLora.h>
#include <NeoML/Dnn/Layers/ActivationLayers.h>
#include <NeoML/Dnn/Layers/CompositeLayer.h>
#include <NeoML/Dnn/Layers/DropoutLayer.h>
#include <NeoML/Dnn/Layers/EltwiseLayer.h>
#include <NeoML/Dnn/Layers/FullyConnectedLayer.h>

namespace NeoML {

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
class NEOML_API CLoraFullyConnectedLayer : public CCompositeLayer {
	NEOML_DNN_LAYER( CLoraFullyConnectedLayer )
public:
	CLoraFullyConnectedLayer( CDnnBlob& baseWeights, CDnnBlob* baseFreeTerms, const CLoraParams& params );
	explicit CLoraFullyConnectedLayer( IMathEngine& mathEngine ); // used for loading serialized layer

	void Serialize( CArchive& ) override;

	CPtr<CDnnBlob> GetSplitBaseWeightsNoCopy() { split(); return baseFc->Weights(); }
	CPtr<CDnnBlob> GetMergedBaseWeightsNoCopy() { merge(); return baseFc->Weights(); }
	CPtr<CDnnBlob> GetFreeTermsNoCopy() { return baseFc->FreeTerms(); }
	CPtr<CDnnBlob> GetAWeightsNoCopy() { return fcA->Weights(); }
	CPtr<CDnnBlob> GetBWeightsNoCopy() { return fcB->Weights(); }

	int OutputSize() const { return baseFc->GetNumberOfElements(); }
	int Rank() const { return fcA->GetNumberOfElements(); }
	float Alpha() const { return scaling->GetMultiplier() * Rank(); }
	float Dropout() const { return dropout->GetDropoutRate(); }

	void UpdateParams( const CLoraParams& newParams, CDnnBlob* newA, CDnnBlob* newB );

	// Mostly for testing/debugging
	CPtr<CDnnBlob>& GetRawBaseWeightsNoCopy() { return baseFc->Weights(); }
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
