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

#include "../Node.h"

namespace NeoOnnx {

// Base class for operator nodes which are activation functions in NeoML
class CActivationNodeBase : public COpNode {
public:
	CActivationNodeBase( const onnx::NodeProto& onnxNode, int opsetVersion,
		TActivationFunction activation );

	// Checks if onnxNode is valid (opset version, number of inputs and outputs, required attributes)
	virtual void CheckOnnxNode() const = 0;
	// Sets additional params for activation layer (e.g. negative slope coeff for CLeakyReLULayer)
	virtual void SetLayerParams( const CObjectArray<const CTensorBase>& /* inputs */, CBaseLayer* /* layer */ ) const {}

	// CNode methods' realizations
	void AddLayers( const CObjectArray<const CTensorBase>& inputs,
		CObjectArray<const CTensorBase>& outputs, CDnn& dnn ) const override;
	void CalculateOutput( const CObjectArray<const CTensorBase>& inputs,
		CObjectArray<const CTensorBase>& outputs, IMathEngine& mathEngine ) const override;

private:
	// Activation function which is applied to the input by this node
	TActivationFunction activation;
};

//---------------------------------------------------------------------------------------------------------------------
// Operator nodes which are implementing activation functions

// Abs operator node
class CAbsNode : public CActivationNodeBase {
public:
	CAbsNode( const onnx::NodeProto& abs, int opsetVersion );

	// CActivationNodeBase methods' realizations
	void CheckOnnxNode() const override;
};

// Clip operator node
class CClipNode : public CActivationNodeBase {
public:
	CClipNode( const onnx::NodeProto& clip, int opsetVersion );

	// CActivationNodeBase methods' realizations
	void CheckOnnxNode() const override;
	void SetLayerParams( const CObjectArray<const CTensorBase>& inputs, CBaseLayer* layer ) const override;
};

// Elu operator node
class CEluNode : public CActivationNodeBase {
public:
	CEluNode( const onnx::NodeProto& elu, int opsetVersion );

	// CActivationNodeBase methods' realizations
	void CheckOnnxNode() const override;
};

// HardSigmoid operator node
class CHardSigmoidNode : public CActivationNodeBase {
public:
	CHardSigmoidNode( const onnx::NodeProto& hardSigmoid, int opsetVersion );

	// CActivationNodeBase methods' realizations
	void CheckOnnxNode() const override;
	void SetLayerParams( const CObjectArray<const CTensorBase>& inputs, CBaseLayer* layer ) const override;
};

// LeakyRelu operator node
class CLeakyReluNode : public CActivationNodeBase {
public:
	CLeakyReluNode( const onnx::NodeProto& leakyRelu, int opsetVersion );

	// CActivationNodeBase methods' realizations
	void CheckOnnxNode() const override;
	void SetLayerParams( const CObjectArray<const CTensorBase>& inputs, CBaseLayer* layer ) const override;
};

// Relu operator node
class CReluNode : public CActivationNodeBase {
public:
	CReluNode( const onnx::NodeProto& relu, int opsetVersion );

	// CActivationNodeBase methods' realizations
	void CheckOnnxNode() const override;
};

// Sigmoid operator node
class CSigmoidNode : public CActivationNodeBase {
public:
	CSigmoidNode( const onnx::NodeProto& sigmoid, int opsetVersion );

	// CActivationNodeBase methods' realizations
	void CheckOnnxNode() const override;
};

// Tanh operator node
class CTanhNode : public CActivationNodeBase {
public:
	CTanhNode( const onnx::NodeProto& tanh, int opsetVersion );

	// CActivationNodeBase methods' realizations
	void CheckOnnxNode() const override;
};

} // namespace NeoOnnx
