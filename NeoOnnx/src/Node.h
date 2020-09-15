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

#include "NodeAttributes.h"
#include "GraphCache.h"
#include "NeoOnnxCheck.h"

// Forward declaration(s).
namespace onnx {
class NodeProto;
} // namespace onnx;

namespace NeoOnnx {

// Node in the onnx calculation graph.
class CNode {
public:
	virtual ~CNode() = default;

	// Calculate output tensors' shape based on inputs' tensors' shape.
	virtual void CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine ) = 0;

	// Marks onnx tensors' dimensions as blob dimensions from NeoML.
	virtual void MarkTensorDims( const CTensorCache& tensors, CDimCache& dims ) = 0;

	// Adds layers, representing this node, to the dnn (if needed).
	virtual void AddLayers( const CGraph& graph, const CTensorCache& tensors, const CDimCache& dims,
		CNeoMLLinkCache& neoMLLinks, CDnn& dnn ) = 0;

	// Gets the number of inputs.
	int InputCount() const;

	// Gets the number of outputs.
	int OutputCount() const;
	
	// Connects index'th input of this node with the link.
	// inputInfo's content must be not null.
	// Must be called once for every used input.
	void Connect( int index, const CLink& link );

	// Gets the link connected to the inputIndex'th input
	const CLink& GetInput( int inputIndex ) const { return Input[inputIndex]; }

protected:
	CNode( int nodeIndex, int inputCount, int outputCount );

	// Links connected to inputs of this node.
	CArray<CLink> Input;

	// Links to outputs of this node.
	CArray<CLink> Output;

private:
	int nodeIndex;
};

//--------------------------------------------------------------------------------------------------------------------
// Opset versioning support
const int MaxOpsetVersion = 12;

// Registers the class as a NeoOnnx node for op_type == opName
#define REGISTER_OP_NODE( classType, opName ) \
	static CNodeClassRegistrar< classType > __merge__1( _RegisterLayer, __LINE__ )( opName );

class COpNode;

typedef COpNode* ( *TCreateOpNodeFunction )( int nodeIndex, const onnx::NodeProto& onnxNode, int opsetVersion );

void RegisterNode( const char* opName, TCreateOpNodeFunction function );

//---------------------------------------------------------------------------------------------------------------------

template<class T>
class CNodeClassRegistrar {
public:
	explicit CNodeClassRegistrar( const char* opName );

private:
	static COpNode* createObject( int nodeIndex, const onnx::NodeProto& onnxNode, int opsetVersion );
};

template<class T>
inline CNodeClassRegistrar<T>::CNodeClassRegistrar( const char* opName )
{
	RegisterNode( opName, createObject );
}

template<class T>
inline COpNode* CNodeClassRegistrar<T>::createObject( int nodeIndex, const onnx::NodeProto& onnxNode, int opsetVersion )
{
	return FINE_DEBUG_NEW T( nodeIndex, onnxNode, opsetVersion );
}

//---------------------------------------------------------------------------------------------------------------------
// Operator node
class COpNode : public CNode {
public:
	~COpNode() override = default;

	// Fabric method. Creates CNode's derivative for given onnx node.
	static COpNode* CreateOpNode( int nodeIndex, const onnx::NodeProto& onnxNode, int opsetVersion );

protected:
	COpNode( int nodeIndex, const onnx::NodeProto& node, int opsetVersion );

	const int OpsetVersion; // Opset version
	const CNodeAttributes Attributes; // Attributes of this node.
	const onnx::NodeProto OnnxNode; // Reference to onnx node. Used for diagnostics.
};

} // namespace NeoOnnx
