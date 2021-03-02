/* Copyright © 2017-2021 ABBYY Production LLC

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
#include <NeoML/Dnn/Layers/RecurrentLayer.h>
#include <NeoML/Dnn/Layers/FullyConnectedLayer.h>

namespace NeoML {

// IRNN implementation from this article: https://arxiv.org/pdf/1504.00941.pdf
//
// It's a simple recurrent unit with the following formula:
//    Y_t = ReLU( FC_input( X_t ) + FC_recur( Y_t-1 ) )
// Where FC_* are fully-connected layers
//
// The crucial point of this layer is weights initialization
// The weight matrix of FC_input is initialized from N(0, inputWeightStd) where inputWeightStd is a layer setting
// The weight matrix of FC_recur is an identity matrix multiplied by identityScale setting
class NEOML_API CIrnnLayer : public CRecurrentLayer {
	NEOML_DNN_LAYER( CIrnnLayer )
public:
	explicit CIrnnLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// Layer settings

	// Sets the number of elements in fully connected layers
	int GetHiddenSize() const { return inputFc->GetNumberOfElements(); }
	void SetHiddenSize( int size );

	// Sets the scale of identity matrix, used for the initialization of recurrent weights
	float GetIdentityScale() const { return identityScale; }
	void SetIdentityScale( float scale ) { identityScale = scale; }

	// Sets the standard deviation for input weights
	float GetInputWeightStd() const { return inputWeightStd; }
	void SetInputWeightStd( float std ) { inputWeightStd = std; }

	// Trainable parameters

	// Input weights
	CPtr<CDnnBlob> GetInputWeightsData() const { return inputFc->GetWeightsData(); }
	CPtr<CDnnBlob> GetInputFreeTermData() const { return inputFc->GetFreeTermData(); }
	void SetInputWeightsData( const CPtr<CDnnBlob>& inputWeights ) { inputFc->SetWeightsData( inputWeights ); }
	void SetInputFreeTermData( const CPtr<CDnnBlob>& inputFreeTerm ) { inputFc->SetFreeTermData( inputFreeTerm ); }

	// Recurrent weights
	CPtr<CDnnBlob> GetRecurWeightsData() const { return inputFc->GetWeightsData(); }
	CPtr<CDnnBlob> GetRecurFreeTermData() const { return inputFc->GetFreeTermData(); }
	void SetRecurWeightsData( const CPtr<CDnnBlob>& recurWeights ) { inputFc->SetWeightsData( recurWeights ); }
	void SetRecurFreeTermData( const CPtr<CDnnBlob>& recurFreeTerm ) { inputFc->SetFreeTermData( recurFreeTerm ); }

protected:
	void Reshape() override;

private:
	float identityScale; // scale of identity matrix (recurrent weights initialization)
	float inputWeightStd; // standard deviartion (input weights initialization)

	CPtr<CFullyConnectedLayer> inputFc; // FC_input from formula above
	CPtr<CFullyConnectedLayer> recurFc; // FC_recur from formula above
	CPtr<CBackLinkLayer> backLink; // Back link for transferring Y_t-1 from formula above

	void buildLayer();

	void identityInitialization( CDnnBlob& blob );
	void normalInitialization( CDnnBlob& blob );
};

} // namespace NeoML
