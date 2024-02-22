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
#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoMathEngine/MathEngineDropout.h>

namespace NeoML {

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
class NEOML_API CLoraFullyConnectedLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CLoraFullyConnectedLayer )
public:
	CLoraFullyConnectedLayer( CDnnBlob& baseWeights, CDnnBlob* baseFreeTerms, const CLoraParams& params );
	explicit CLoraFullyConnectedLayer( IMathEngine& mathEngine, const char* name = nullptr ); // used for loading serialized layer

	void Serialize( CArchive& ) override;

	void UpdateParams( const CLoraParams& newParams, CDnnBlob* newA, CDnnBlob* newB );

	int OutputSize() const { return NumberOfElements(); }
	int Rank() const { return lora.Rank; }
	float Alpha() const { return lora.Alpha; }
	float Dropout() const { return lora.Dropout; }

	// Raw getters for weights
	// These getters do not copy weights which may lead to difficult-to-debug troubles
	// But they're necessary for making LoRA work without excessive copying
	// baseFc weights from "split" state
	CPtr<CDnnBlob> GetSplitWeightsNoCopy() { split(); return weightsBase; }
	// baseFc weights from "merged" state
	CPtr<CDnnBlob> GetMergedWeightsNoCopy() { merge(); return weightsBase; }
	// baseFc free terms
	CPtr<CDnnBlob> GetFreeTermsNoCopy() { return freeTermsBase; }
	// A LoRA matrix
	CPtr<CDnnBlob> GetAWeightsNoCopy() { return WeightsA(); }
	// B LoRA matrix
	CPtr<CDnnBlob> GetBWeightsNoCopy() { return WeightsB(); }

	// Mostly for testing/debugging
	CPtr<CDnnBlob>& GetWeightsNoCopy() { return weightsBase; }
	bool IsMerged() const { return isMerged; }

protected:
	~CLoraFullyConnectedLayer() override;

	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	void LearnOnce() override;

	int BlobsForBackward() const override { return 0; }
	int BlobsForLearn() const override { return TInputBlobs; }

	CPtr<CDnnBlob>& WeightsA() { return paramBlobs[0]; } // weights A transposed
	CPtr<CDnnBlob>& WeightsB() { return paramBlobs[1]; } // weights B transposed
	const CPtr<CDnnBlob>& WeightsA() const { return paramBlobs[0]; } // weights A transposed
	const CPtr<CDnnBlob>& WeightsB() const { return paramBlobs[1]; } // weights B transposed

	CPtr<CDnnBlob>& WeightsADiff() { return paramDiffBlobs[0]; } // weightsDiff A transposed
	CPtr<CDnnBlob>& WeightsBDiff() { return paramDiffBlobs[1]; } // weightsDiff B transposed
	
	int NumberOfElements() const { return weightsBase->GetObjectCount(); }

private:
	bool isMerged = true;
	CLoraParams lora;
	CBaseDropoutDesc* desc = nullptr; // dropout description
	CPtr<CDnnBlob> weightsBase; // weights Base transposed (as default)
	CPtr<CDnnBlob> freeTermsBase; // freeTerms Base as default
	CPtr<CDnnBlob> scaling; // scaling = lora.Alpha / lora.Rank

	void initialize( const CLoraParams& params );
	void merge();
	void split();
	void recalcBaseWeights();
	void disableDropoutDesc();
	void destroyDropoutDesc();
	void initDropoutDesc();
};

} // namespace NeoML
