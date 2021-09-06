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

#include <memory>
#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Dnn.h>
#include <NeoML/Dnn/Layers/CompositeLayer.h>
#include <NeoML/Dnn/Layers/DropoutLayer.h>
#include <NeoML/Dnn/Layers/FullyConnectedLayer.h>

namespace NeoML {

// Forward declaration(s)
class CIndRnnRecurrentLayer;

// Independently Recurrent Neural Network (IndRNN): https://arxiv.org/pdf/1803.04831.pdf
//
// It's a simple recurrent unit with the following formula:
//    Y_t = activation( W * X_t + B + U * Y_t-1 )
// Where:
//    W and B are weights and free terms of the fully-connected layer (W * X_t is a matrix multiplication)
//    U is a vector (U * Y_t-1 is an eltwise multiplication of 2 vectors of the same length)
//    activation is an activation function (sigmoid or ReLU)

class NEOML_API CIndRnnLayer : public CCompositeLayer {
	NEOML_DNN_LAYER( CIndRnnLayer )
public:
	explicit CIndRnnLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// Layer settings

	// Sets the number of element in hidden layer and vector U
	// 1 by default
	int GetHiddenSize() const { return fc->GetNumberOfElements(); }
	void SetHiddenSize( int hiddenSize );

	// Sets dropout rate
	// Dropout is applied to X_t and Y_t-1 before any other operations
	// 0 by default (no dropout)
	float GetDropoutRate() const { return inputDropout == nullptr ? 0 : inputDropout->GetDropoutRate(); }
	void SetDropoutRate( float rate );

	// Sets whether sequences must be processed in direct or reversed order
	bool IsReverseSequence() const;
	void SetReverseSequence( bool reverse );

	// Sets the activation function used in the recurrent part
	// AF_Sigmoid by default
	TActivationFunction GetActivation() const;
	void SetActivation( TActivationFunction activation );

	// Trainable parameters
	
	// Input weights (matrix W from formula)
	CPtr<CDnnBlob> GetInputWeights() const { return fc->GetWeightsData(); }
	void SetInputWeights( const CDnnBlob* inputWeights ) { fc->SetWeightsData( inputWeights ); }

	// Recurrent weights (vector U from formula)
	CPtr<CDnnBlob> GetRecurrentWeights() const;
	void SetRecurrentWeights( const CDnnBlob* recurrentWeights );

	// Bias (vector B from formula)
	CPtr<CDnnBlob> GetBias() const { return fc->GetFreeTermData(); }
	void SetBias( const CDnnBlob* bias ) { fc->SetFreeTermData( bias ); }

private:
	CPtr<CDropoutLayer> inputDropout; // input dropout (if needed)
	CPtr<CFullyConnectedLayer> fc; // Fully connected layer (W and B)
	CPtr<CIndRnnRecurrentLayer> recurrent; // Recurrent part

	void buildLayer();
};

// Recurrent part of IndRNN:
//    Y_t = activation( FC( X_t ) + U * Y_t-1 )
//
// For optimization purposes this class doesn't inherit CRecurrentLayer
// mathEngine 'emulates' recurrent part instead
class NEOML_API CIndRnnRecurrentLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CIndRnnRecurrentLayer )
public:
	explicit CIndRnnRecurrentLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// Layer settings

	// Sets whether input sequences must be processed in reversed order
	bool IsReverseSequence() const { return reverse; }
	void SetReverseSequence( bool value ) { reverse = value; }

	// Sets the dropout rate applied to Y_t-1
	float GetDropoutRate() const { return dropoutRate; }
	void SetDropoutRate( float rate );

	// Sets activation function
	TActivationFunction GetActivation() const { return activation; }
	void SetActivation( TActivationFunction activation );

	// Trainable parameters

	// Weights (vector U from formula)
	CPtr<CDnnBlob> GetWeights() const;
	void SetWeights( const CDnnBlob* weights );

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	void LearnOnce() override;

private:
	TActivationFunction activation; // Activation function
	bool reverse; // If true then sequences must be processed in reversed order
	float dropoutRate; // Dropout rate on recurrent link
	std::unique_ptr<CFloatHandleVar> dropoutMask; // Dropout mask

	CConstFloatHandle maskHandle() const;
};

NEOML_API CLayerWrapper<CIndRnnLayer> IndRnn( int hiddenSize, float dropoutRate = 0.f, bool reverse = false,
	TActivationFunction activation = AF_Sigmoid );

} // namespace NeoML
