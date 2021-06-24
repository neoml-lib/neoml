/* Copyright © 2017-2020 ABBYY Production LLC

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

#include "../LayerOperator.h"

namespace NeoOnnx {

// Base class for operators which are activation functions in NeoML
class CActivationOperatorBase : public CLayerOperator {
public:
	CActivationOperatorBase( const onnx::NodeProto& onnxNode, int opsetVersion,
		TActivationFunction activation );

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;
	void UserInputMask( CUserInputMask& mask ) const override { mask |= 0; }

	// Sets additional params for activation layer (e.g. negative slope coeff for CLeakyReLULayer)
	virtual void SetLayerParams( const CTensorArray& /* inputs */, CBaseLayer* /* layer */ ) const {}

private:
	// Activation function which is applied to the input by this operator
	TActivationFunction activation;
};

//---------------------------------------------------------------------------------------------------------------------
// Operators which are implementing activation functions

// Abs operator
class CAbsOperator : public CActivationOperatorBase {
public:
	CAbsOperator( const onnx::NodeProto& abs, int opsetVersion );
};

// Clip operator
class CClipOperator : public CActivationOperatorBase {
public:
	CClipOperator( const onnx::NodeProto& clip, int opsetVersion );

protected:
	// CActivationOperatorBase methods
	void SetLayerParams( const CTensorArray& inputs, CBaseLayer* layer ) const override;
};

// Elu operator
class CEluOperator : public CActivationOperatorBase {
public:
	CEluOperator( const onnx::NodeProto& elu, int opsetVersion );
};

// HardSigmoid operator
class CHardSigmoidOperator : public CActivationOperatorBase {
public:
	CHardSigmoidOperator( const onnx::NodeProto& hardSigmoid, int opsetVersion );

protected:
	// CActivationOperatorBase methods
	void SetLayerParams( const CTensorArray& inputs, CBaseLayer* layer ) const override;
};

// LeakyRelu operator
class CLeakyReluOperator : public CActivationOperatorBase {
public:
	CLeakyReluOperator( const onnx::NodeProto& leakyRelu, int opsetVersion );

protected:
	// CActivationOperatorBase methods
	void SetLayerParams( const CTensorArray& inputs, CBaseLayer* layer ) const override;
};

// Relu operator
class CReluOperator : public CActivationOperatorBase {
public:
	CReluOperator( const onnx::NodeProto& relu, int opsetVersion );
};

// Sigmoid operator
class CSigmoidOperator : public CActivationOperatorBase {
public:
	CSigmoidOperator( const onnx::NodeProto& sigmoid, int opsetVersion );
};

// Tanh operator
class CTanhOperator : public CActivationOperatorBase {
public:
	CTanhOperator( const onnx::NodeProto& tanh, int opsetVersion );
};

} // namespace NeoOnnx
