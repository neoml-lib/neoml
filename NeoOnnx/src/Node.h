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

// Node's outputs mapping with NeoML layers
struct CNeoMLMapping {
	// Constructor for onnx node output, which doesn't with any NeoML layer's output.
	CNeoMLMapping() : Layer( nullptr ), OutputIndex( NotFound ) {}

	// Constructor for onnx node output, matching it with layer's outputIndex'th output.
	CNeoMLMapping( CBaseLayer* layer, int outputIndex ) : Layer( layer ), OutputIndex( outputIndex )
		{ CheckNeoOnnxInternal( layer != nullptr, "non empty output info with layer == nullptr" ); }

	CBaseLayer* Layer; // Used NeoML layer (nullptr if there is no layer mapped with this output)
	int OutputIndex; // NeoML layer's output index, mapped with this output
};

//--------------------------------------------------------------------------------------------------------------------

class CNode;

typedef CPointerArray<CNode> CGraph;
typedef CArray<CArray<CTensor>> CGraphTensors;
typedef CArray<CArray<CTensorDim>> CGraphDims;
typedef CArray<CArray<CNeoMLMapping>> CGraphMappings;

// Node in the onnx calculation graph.
class CNode {
public:
	virtual ~CNode() = default;

	// Calculate output tensors' shape based on inputs' tensors' shape.
	virtual void CalcOutputTensors( CGraphTensors& tensors, IMathEngine& mathEngine ) = 0;

	// Marks onnx tensors' dimensions as blob dimensions from NeoML.
	virtual void MarkTensorDims( const CGraphTensors& tensors, CGraphDims& dims ) = 0;

	// Adds layers, representing this node, to the dnn (if needed).
	virtual void AddLayers( const CGraph& graph, const CGraphTensors& tensors, const CGraphDims& dims, CGraphMappings& mappings, CDnn& dnn ) = 0;

	// Gets the number of inputs.
	int InputCount() const;

	// Gets the number of outputs.
	int OutputCount() const;

	// Information about input.
	struct CInputInfo {
		CInputInfo() : NodeIndex( NotFound ), OutputIndex( NotFound ) {}
		CInputInfo( int nodeIndex, int outputIndex ) : NodeIndex( nodeIndex ), OutputIndex( outputIndex ) {}

		int NodeIndex; // Node connected to this input.
		int OutputIndex; // Node's output number connected to this input.
	};
	
	// Set index'th input of this node.
	// inputInfo's content must be not null.
	// Must be called once for every used input.
	void SetInput( int index, const CInputInfo& inputInfo );

	const CInputInfo& GetInput( int index ) const;

	// Different accessors
	const CNode* InputNode( const CGraph& graph, int inputIndex ) const { return graph[inputs[inputIndex].NodeIndex]; }

	const CTensor& InputTensor( const CGraphTensors& tensors, int inputIndex ) const { return tensors[inputs[inputIndex].NodeIndex][inputs[inputIndex].OutputIndex]; }
	const CTensor& OutputTensor( const CGraphTensors& tensors, int outputIndex ) const { return tensors[nodeIndex][outputIndex]; }
	CTensor& OutputTensor( CGraphTensors& tensors, int outputIndex ) const { return tensors[nodeIndex][outputIndex]; }

	const CTensorDim& InputDim( const CGraphDims& dims, int inputIndex ) const { return dims[inputs[inputIndex].NodeIndex][inputs[inputIndex].OutputIndex]; }
	CTensorDim& InputDim( CGraphDims& dims, int inputIndex ) const { return dims[inputs[inputIndex].NodeIndex][inputs[inputIndex].OutputIndex]; }
	const CTensorDim& OutputDim( const CGraphDims& dims, int outputIndex ) const { return dims[nodeIndex][outputIndex]; }
	CTensorDim& OutputDim( CGraphDims& dims, int outputIndex ) const { return dims[nodeIndex][outputIndex]; }

	const CNeoMLMapping& InputMapping( const CGraphMappings& mappings, int inputIndex ) { return mappings[inputs[inputIndex].NodeIndex][inputs[inputIndex].OutputIndex]; }
	const CNeoMLMapping& OutputMapping( const CGraphMappings& mappings, int outputIndex ) { return mappings[nodeIndex][outputIndex]; }
	CNeoMLMapping& OutputMapping( CGraphMappings& mappings, int outputIndex ) { return mappings[nodeIndex][outputIndex]; }

protected:
	CNode( int nodeIndex, int inputCount, int outputCount );

private:
	int nodeIndex;
	CArray<CInputInfo> inputs;
	int outputCount;
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
