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

#include "Nodes/AddNode.h"
#include "Nodes/AveragePoolNode.h"
#include "Nodes/BatchNormalizationNode.h"
#include "Nodes/ClipNode.h"
#include "Nodes/ConcatNode.h"
#include "Nodes/ConstantNode.h"
#include "Nodes/ConstantOfShapeNode.h"
#include "Nodes/ConvNode.h"
#include "Nodes/FlattenNode.h"
#include "Nodes/GatherNode.h"
#include "Nodes/GemmNode.h"
#include "Nodes/GlobalAveragePoolNode.h"
#include "Nodes/LstmNode.h"
#include "Nodes/MaxPoolNode.h"
#include "Nodes/ReduceMeanNode.h"
#include "Nodes/ReluNode.h"
#include "Nodes/ShapeNode.h"
#include "Nodes/SliceNode.h"
#include "Nodes/SqueezeNode.h"
#include "Nodes/TanhNode.h"
#include "Nodes/UnsqueezeNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

#include <string>

namespace NeoOnnx {

static CMap<CString, TCreateNodeFunction>& getRegisteredNodes()
{
	static CMap<CString, TCreateNodeFunction> registeredNodes;
	return registeredNodes;
}

void RegisterNode( const char* opName, TCreateNodeFunction function )
{
	NeoAssert( !getRegisteredNodes().Has( opName ) );
	getRegisteredNodes().Add( opName, function );
}

namespace {

// Register all nodes
REGISTER_NEOONNX_NODE( CAddNode, "Add" )
REGISTER_NEOONNX_NODE( CAveragePoolNode, "AveragePool" )
REGISTER_NEOONNX_NODE( CBatchNormalizationNode, "BatchNormalization" )
REGISTER_NEOONNX_NODE( CClipNode, "Clip" )
REGISTER_NEOONNX_NODE( CConcatNode, "Concat" )
REGISTER_NEOONNX_NODE( CConstantNode, "Constant" )
REGISTER_NEOONNX_NODE( CConstantOfShapeNode, "ConstantOfShape" )
REGISTER_NEOONNX_NODE( CConvNode, "Conv" )
REGISTER_NEOONNX_NODE( CFlattenNode, "Flatten" )
REGISTER_NEOONNX_NODE( CGatherNode, "Gather" )
REGISTER_NEOONNX_NODE( CGemmNode, "Gemm" )
REGISTER_NEOONNX_NODE( CGlobalAveragePoolNode, "GlobalAveragePool" )
REGISTER_NEOONNX_NODE( CLstmNode, "LSTM" )
REGISTER_NEOONNX_NODE( CMaxPoolNode, "MaxPool" )
REGISTER_NEOONNX_NODE( CReduceMeanNode, "ReduceMean" )
REGISTER_NEOONNX_NODE( CReluNode, "Relu" )
REGISTER_NEOONNX_NODE( CShapeNode, "Shape" )
REGISTER_NEOONNX_NODE( CSliceNode, "Slice" )
REGISTER_NEOONNX_NODE( CSqueezeNode, "Squeeze" )
REGISTER_NEOONNX_NODE( CTanhNode, "Tanh" )
REGISTER_NEOONNX_NODE( CUnsqueezeNode, "Unsqueeze" )

} // namespace

//---------------------------------------------------------------------------------------------------------------------

CNode::CNode( const onnx::NodeProto& _onnxNode, int _opsetVersion ) :
	opsetVersion( _opsetVersion ),
	attributes( _onnxNode ),
	onnxNode( _onnxNode )
{
	input.SetSize( onnxNode.input_size() );
	output.SetSize( _onnxNode.output_size() );
}

CNode::CNode( int inputCount, int outputCount ) :
	opsetVersion( -1 )
{
	input.SetSize( inputCount );
	output.SetSize( outputCount );
}

int CNode::OutputCount() const
{
	return output.Size();
}

const CTensor& CNode::InputTensor( int index ) const
{
	CheckNeoOnnxInternal( index >= 0 && index < input.Size(), "attempt to access non-existing input" );
	CheckNeoOnnxInternal( input[index].InputNode != nullptr, "attempt to acces empty input" );
	return input[index].InputNode->output[input[index].OutputIndex];
}

CTensor& CNode::InputTensor( int index )
{
	CheckNeoOnnxInternal( index >= 0 && index < input.Size(), "attempt to access non-existing input" );
	CheckNeoOnnxInternal( input[index].InputNode != nullptr, "attempt to acces empty input" );
	return input[index].InputNode->output[input[index].OutputIndex];
}

void CNode::SetInput( int index, const CNode::CInputInfo& inputInfo )
{
	CheckNeoOnnxInternal( index >= 0 && index < input.Size(), "attempt to set non-existing input" );
	CheckNeoOnnxInternal( input[index].InputNode == nullptr && input[index].OutputIndex == NotFound,
		"attempt to set already-defined input" );
	CheckNeoOnnxInternal( inputInfo.InputNode != nullptr, "attempt to set an input to nullptr" );
	CheckNeoOnnxInternal( inputInfo.OutputIndex >= 0, "attempt to set an input with incorrect index" );
	input[index] = inputInfo;
}

const CNode::CNeoMLInputInfo& CNode::InputInfo( int index ) const
{
	CheckNeoOnnxInternal( index >= 0 && index < input.Size(), "attempt to access non-existing input" );
	CheckNeoOnnxInternal( input[index].InputNode != nullptr, "attempt to acces empty input" );
	const CNode& inputNode = *input[index].InputNode;
	const int inputNodeOutputIndex = input[index].OutputIndex;

	NeoAssert( inputNode.neoMLInputInfo.Size() == inputNode.output.Size() );
	NeoAssert( inputNodeOutputIndex >= 0 && inputNodeOutputIndex < inputNode.output.Size() );
	NeoAssert( inputNode.neoMLInputInfo[inputNodeOutputIndex].Layer != nullptr );
	return inputNode.neoMLInputInfo[inputNodeOutputIndex];
}

const CBaseLayer& CNode::InputLayer( int index ) const
{
	return *InputInfo( index ).Layer;
}

int CNode::InputLayerIndex( int index ) const
{
	return InputInfo( index ).OutputIndex;
}

CNode* CNode::CreateNode( const onnx::NodeProto& onnxNode, int opsetVersion, IMathEngine& mathEngine )
{
	TMapPosition pos = getRegisteredNodes().GetFirstPosition( onnxNode.op_type() );
	CheckNeoOnnxSupport( pos != NotFound, CString( "operator " ) + onnxNode.op_type().c_str() );
	return getRegisteredNodes().GetValue( pos )( onnxNode, opsetVersion, mathEngine );
}

} // namespace NeoOnnx
