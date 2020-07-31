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
#include "Tensor.h"
#include "NeoOnnxCheck.h"

// Forward declaration(s).
namespace onnx {
class NodeProto;
class TensorProto;
class ValueInfoProto;
} // namespace onnx;

namespace NeoOnnx {

// Link to the OutputIndex'th output of the NeoML Layer.
struct CNeoMLLink {
	// Constructor for onnx node output, which doesn't with any NeoML layer's output.
	CNeoMLLink() : Layer( nullptr ), OutputIndex( NotFound ) {}

	// Constructor for onnx node output, matching it with layer's outputIndex'th output.
	CNeoMLLink( CBaseLayer* layer, int outputIndex ) : Layer( layer ), OutputIndex( outputIndex )
		{ CheckNeoOnnxInternal( layer != nullptr, "non empty output info with layer == nullptr" ); }

	CBaseLayer* Layer; // Used NeoML layer (nullptr if there is no layer mapped with this output)
	int OutputIndex; // NeoML layer's output index, mapped with this output
};

//--------------------------------------------------------------------------------------------------------------------

// Link to NodeIndex'th node OutputIndex'th output.
struct CLink
{
	CLink() : NodeIndex( NotFound ), OutputIndex( NotFound ) {}
	CLink( int nodeIndex, int outputIndex ) : NodeIndex( nodeIndex ), OutputIndex( outputIndex ) {}

	int NodeIndex; // Node connected to this input.
	int OutputIndex; // Node's output number connected to this input.
};

class CNode;

class CGraph {
public:
	CNode* operator[]( int nodeIndex ) { return nodes[nodeIndex]; }
	const CNode* operator[]( int nodeIndex ) const { return nodes[nodeIndex]; }
	const CNode* operator[]( const CLink& link ) const { return nodes[link.NodeIndex]; }

	void Add( CNode* newNode ) { nodes.Add( newNode ); }

	int NodeCount() const { return nodes.Size(); }

	void SetBufferSize( int nodeCount ) { nodes.SetBufferSize( nodeCount ); }

private:
	CPointerArray<CNode> nodes;
};

template<class T>
class CGraphCache {
public:
	explicit CGraphCache( const CGraph& graph );

	T& operator[]( const CLink& link );
	const T& operator[]( const CLink& link ) const;

private:
	CArray<CArray<T>> cache;
};

template<class T>
CGraphCache<T>::CGraphCache( const CGraph& graph )
{
	cache.SetSize( graph.NodeCount() );
	for( int i = 0; i < graph.NodeCount(); ++i ) {
		cache[i].SetSize( graph[i]->OutputCount() );
	}
}

template<class T>
T& CGraphCache<T>::operator[]( const CLink& link )
{
	return cache[link.NodeIndex][link.OutputIndex];
}

template<class T>
const T& CGraphCache<T>::operator[]( const CLink& link ) const
{
	return cache[link.NodeIndex][link.OutputIndex];
}

typedef CGraphCache<CTensor> CTensorCache;
typedef CGraphCache<CTensorDim> CDimCache;
typedef CGraphCache<CNeoMLLink> CNeoMLLinkCache;

// Node in the onnx calculation graph.
class CNode {
public:
	virtual ~CNode() = default;

	// Calculate output tensors' shape based on inputs' tensors' shape.
	virtual void CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine ) = 0;

	// Marks onnx tensors' dimensions as blob dimensions from NeoML.
	virtual void MarkTensorDims( const CTensorCache& tensors, CDimCache& dims ) = 0;

	// Adds layers, representing this node, to the dnn (if needed).
	virtual void AddLayers( const CGraph& graph, const CTensorCache& tensors, const CDimCache& dims, CNeoMLLinkCache& neoMLLinks, CDnn& dnn ) = 0;

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

	const int opsetVersion; // Opset version
	const CNodeAttributes attributes; // Attributes of this node.
	const onnx::NodeProto onnxNode; // Reference to onnx node. Used for diagnostics.
};

} // namespace NeoOnnx
