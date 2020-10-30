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
class CActivationNode : public COpNode {
public:
	CActivationNode( int nodeIndex, const onnx::NodeProto& onnxNode,
		int opsetVersion, TActivationFunction activation );

	// Checks if onnxNode is valid (opset version, number of inputs and outputs, required attributes)
	virtual void CheckOnnxNode() const = 0;
	// Sets additional params for activation layer (e.g. negative slope coeff for CLeakyReLULayer)
	virtual void SetLayerParams( const CTensorCache& /* tensors */, CBaseLayer* /* layer */ ) const {}

	// CNode methods' realizations
	void CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine ) override final;
	void LabelTensorDims( const CTensorCache& tensors, CDimCache& dims ) override final;
	void AddLayers( const CGraph& graph, const CTensorCache& tensors, const CDimCache& dims,
		CNeoMLLinkCache& neoMLLinks, CDnn& dnn ) override final;

private:
	// Activation function which is applied to the input by this node
	TActivationFunction activation;
};

//---------------------------------------------------------------------------------------------------------------------
// Operator nodes which are implementing activation functions

// Abs operator node
class CAbsNode : public CActivationNode {
public:
	CAbsNode( int nodeIndex, const onnx::NodeProto& abs, int opsetVersion );

	// CActivationNode methods' realizations
	void CheckOnnxNode() const override;
};

// Clip operator node
class CClipNode : public CActivationNode {
public:
	CClipNode( int nodeIndex, const onnx::NodeProto& clip, int opsetVersion );

	// CActivationNode methods' realizations
	void CheckOnnxNode() const override;
	void SetLayerParams( const CTensorCache& tensors, CBaseLayer* layer ) const override;
};

// Elu operator node
class CEluNode : public CActivationNode {
public:
	CEluNode( int nodeIndex, const onnx::NodeProto& elu, int opsetVersion );

	// CActivationNode methods' realizations
	void CheckOnnxNode() const override;
};

// LeakyRelu operator node
class CLeakyReluNode : public CActivationNode {
public:
	CLeakyReluNode( int nodeIndex, const onnx::NodeProto& leakyRelu, int opsetVersion );

	// CActivationNode methods' realizations
	void CheckOnnxNode() const override;
	void SetLayerParams( const CTensorCache& tensors, CBaseLayer* layer ) const override;
};

// Relu operator node
class CReluNode : public CActivationNode {
public:
	CReluNode( int nodeIndex, const onnx::NodeProto& relu, int opsetVersion );

	// CActivationNode methods' realizations
	void CheckOnnxNode() const override;
};

// Sigmoid operator node
class CSigmoidNode : public CActivationNode {
public:
	CSigmoidNode( int nodeIndex, const onnx::NodeProto& sigmoid, int opsetVersion );

	// CActivationNode methods' realizations
	void CheckOnnxNode() const override;
};

// Tanh operator node
class CTanhNode : public CActivationNode {
public:
	CTanhNode( int nodeIndex, const onnx::NodeProto& tanh, int opsetVersion );

	// CActivationNode methods' realizations
	void CheckOnnxNode() const override;
};

} // namespace NeoOnnx
