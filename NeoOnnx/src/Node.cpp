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

#include "proto/onnx.pb.h"

#include <string>

namespace NeoOnnx {

CNode::CNode( const onnx::NodeProto& _onnxNode ) :
	attributes( _onnxNode ),
	onnxNode( _onnxNode )
{
	input.SetSize( onnxNode.input_size() );
	output.SetSize( _onnxNode.output_size() );
}

CNode::CNode( int inputCount, int outputCount )
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

CNode* CNode::CreateNode( const onnx::NodeProto& onnxNode, IMathEngine& mathEngine )
{
	if( onnxNode.op_type() == "Add" ) {
		return new CAddNode( onnxNode );
	} else if( onnxNode.op_type() == "AveragePool" ) {
		return new CAveragePoolNode( onnxNode );
	} else if( onnxNode.op_type() == "BatchNormalization" ) {
		return new CBatchNormalizationNode( onnxNode );
	} else if( onnxNode.op_type() == "Clip" ) {
		return new CClipNode( onnxNode );
	} else if( onnxNode.op_type() == "Concat" ) {
		return new CConcatNode( onnxNode );
	} else if( onnxNode.op_type() == "Constant" ) {
		return new CConstantNode( onnxNode, mathEngine );
	} else if( onnxNode.op_type() == "ConstantOfShape" ) {
		return new CConstantOfShapeNode( onnxNode );
	} else if( onnxNode.op_type() == "Conv" ) {
		return new CConvNode( onnxNode );
	} else if( onnxNode.op_type() == "Flatten" ) {
		return new CFlattenNode( onnxNode );
	} else if( onnxNode.op_type() == "Gather" ) {
		return new CGatherNode( onnxNode );
	} else if( onnxNode.op_type() == "Gemm" ) {
		return new CGemmNode( onnxNode );
	} else if( onnxNode.op_type() == "GlobalAveragePool" ) {
		return new CGlobalAveragePoolNode( onnxNode );
	} else if( onnxNode.op_type() == "LSTM" ) {
		return new CLstmNode( onnxNode );
	} else if( onnxNode.op_type() == "MaxPool" ) {
		return new CMaxPoolNode( onnxNode );
	} else if( onnxNode.op_type() == "ReduceMean" ) {
		return new CReduceMeanNode( onnxNode );
	} else if( onnxNode.op_type() == "Relu" ) {
		return new CReluNode( onnxNode );
	} else if( onnxNode.op_type() == "Shape" ) {
		return new CShapeNode( onnxNode, mathEngine );
	} else if( onnxNode.op_type() == "Slice" ) {
		return new CSliceNode( onnxNode );
	} else if( onnxNode.op_type() == "Squeeze" ) {
		return new CSqueezeNode( onnxNode );
	} else if( onnxNode.op_type() == "Tanh" ) {
		return new CTanhNode( onnxNode );
	} else if( onnxNode.op_type() == "Unsqueeze" ) {
		return new CUnsqueezeNode( onnxNode );
	}

	CheckNeoOnnxSupport( false, CString( "operator " ) + onnxNode.op_type().c_str() );
	return nullptr;
}

} // namespace NeoOnnx
