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

#include "common.h"
#pragma hdrstop

#include "Node.h"
#include "Graph.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

#include <string>

#include "Nodes/AbsNode.h"
#include "Nodes/AddNode.h"
#include "Nodes/BatchNormalizationNode.h"
#include "Nodes/ClipNode.h"
#include "Nodes/ConcatNode.h"
#include "Nodes/ConstantNode.h"
#include "Nodes/ConstantOfShapeNode.h"
#include "Nodes/ConvNode.h"
#include "Nodes/EluNode.h"
#include "Nodes/FlattenNode.h"
#include "Nodes/GatherNode.h"
#include "Nodes/GemmNode.h"
#include "Nodes/GlobalAveragePoolNode.h"
#include "Nodes/LeakyReluNode.h"
#include "Nodes/LstmNode.h"
#include "Nodes/PoolNode.h"
#include "Nodes/ReduceMeanNode.h"
#include "Nodes/ReluNode.h"
#include "Nodes/ReshapeNode.h"
#include "Nodes/ShapeNode.h"
#include "Nodes/SigmoidNode.h"
#include "Nodes/SliceNode.h"
#include "Nodes/SqueezeNode.h"
#include "Nodes/TanhNode.h"
#include "Nodes/UnsqueezeNode.h"

namespace NeoOnnx {

// Registers the class as a NeoOnnx node for op_type == opName
#define REGISTER_OP_NODE( classType, opName ) \
	static CNodeClassRegistrar< classType > __merge__1( _RegisterOpNode, __LINE__ )( opName );

typedef COpNode* ( *TCreateOpNodeFunction )( int nodeIndex, const onnx::NodeProto& onnxNode, int opsetVersion );

// Returns reference to the map containing info about registered nodes
static CMap<CString, TCreateOpNodeFunction>& getRegisteredNodes()
{
	static CMap<CString, TCreateOpNodeFunction> registeredNodes;
	return registeredNodes;
}

// Registers function as a way to create operator node for NodeProto::op_type == opName
void registerNode( const char* opName, TCreateOpNodeFunction function )
{
	CheckNeoOnnxInternal( !getRegisteredNodes().Has( opName ), "Double-register node op: " + CString( opName ) );
	getRegisteredNodes().Add( opName, function );
}

//---------------------------------------------------------------------------------------------------------------------
// Class registers class T as an operator node
// Without this registration class will be inaccessible from COpNode::CreateOpNode
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
	registerNode( opName, createObject );
}

template<class T>
inline COpNode* CNodeClassRegistrar<T>::createObject( int nodeIndex, const onnx::NodeProto& onnxNode, int opsetVersion )
{
	return FINE_DEBUG_NEW T( nodeIndex, onnxNode, opsetVersion );
}

//---------------------------------------------------------------------------------------------------------------------

namespace {

// Register all nodes
REGISTER_OP_NODE( CAbsNode, "Abs" )
REGISTER_OP_NODE( CAddNode, "Add" )
REGISTER_OP_NODE( CAveragePoolNode, "AveragePool" )
REGISTER_OP_NODE( CBatchNormalizationNode, "BatchNormalization" )
REGISTER_OP_NODE( CClipNode, "Clip" )
REGISTER_OP_NODE( CConcatNode, "Concat" )
REGISTER_OP_NODE( CConstantNode, "Constant" )
REGISTER_OP_NODE( CConstantOfShapeNode, "ConstantOfShape" )
REGISTER_OP_NODE( CConvNode, "Conv" )
REGISTER_OP_NODE( CEluNode, "Elu" )
REGISTER_OP_NODE( CFlattenNode, "Flatten" )
REGISTER_OP_NODE( CGatherNode, "Gather" )
REGISTER_OP_NODE( CGemmNode, "Gemm" )
REGISTER_OP_NODE( CGlobalAveragePoolNode, "GlobalAveragePool" )
REGISTER_OP_NODE( CLeakyReluNode, "LeakyRelu" )
REGISTER_OP_NODE( CLstmNode, "LSTM" )
REGISTER_OP_NODE( CMaxPoolNode, "MaxPool" )
REGISTER_OP_NODE( CReduceMeanNode, "ReduceMean" )
REGISTER_OP_NODE( CReluNode, "Relu" )
REGISTER_OP_NODE( CReshapeNode, "Reshape" )
REGISTER_OP_NODE( CShapeNode, "Shape" )
REGISTER_OP_NODE( CSigmoidNode, "Sigmoid" )
REGISTER_OP_NODE( CSliceNode, "Slice" )
REGISTER_OP_NODE( CSqueezeNode, "Squeeze" )
REGISTER_OP_NODE( CTanhNode, "Tanh" )
REGISTER_OP_NODE( CUnsqueezeNode, "Unsqueeze" )

} // namespace

//---------------------------------------------------------------------------------------------------------------------

CNode::CNode( int _nodeIndex, int inputCount, int _outputCount ) :
	nodeIndex( _nodeIndex )
{
	Input.SetSize( inputCount );
	Output.SetBufferSize( _outputCount );
	for( int outputIndex = 0; outputIndex < _outputCount; ++outputIndex ) {
		Output.Add( CLink( nodeIndex, outputIndex ) );
	}
}

int CNode::InputCount() const
{
	return Input.Size();
}

int CNode::OutputCount() const
{
	return Output.Size();
}

void CNode::Connect( int index, const CLink& inputInfo )
{
	CheckNeoOnnxInternal( index >= 0 && index < InputCount(), "attempt to connect non-existing input" );
	CheckNeoOnnxInternal( Input[index].NodeIndex == NotFound && Input[index].OutputIndex == NotFound,
		"attempt to connect already-connected input" );
	CheckNeoOnnxInternal( inputInfo.OutputIndex >= 0, "attempt to connect an input with incorrect index" );
	Input[index] = inputInfo;
}

//---------------------------------------------------------------------------------------------------------------------

COpNode::COpNode( int nodeIndex, const onnx::NodeProto& _onnxNode, int _opsetVersion ) :
	CNode( nodeIndex, _onnxNode.input_size(), _onnxNode.output_size() ),
	OpsetVersion( _opsetVersion ),
	Attributes( _onnxNode ),
	OnnxNode( _onnxNode )
{
}

COpNode* COpNode::CreateOpNode( int nodeIndex, const onnx::NodeProto& onnxNode, int opsetVersion )
{
	TMapPosition pos = getRegisteredNodes().GetFirstPosition( onnxNode.op_type() );
	CheckNeoOnnxSupport( pos != NotFound, CString( "operator " ) + onnxNode.op_type().c_str() );
	return getRegisteredNodes().GetValue( pos )( nodeIndex, onnxNode, opsetVersion );
}

} // namespace NeoOnnx
