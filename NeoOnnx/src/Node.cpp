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

CNode::CNode( const onnx::NodeProto& _onnxNode, CMap<CString, CInputInfo>& nodeOutputs ) :
	attributes( _onnxNode ),
	onnxOutputCount( _onnxNode.output_size() ),
	onnxNode( _onnxNode )
{
	input.SetBufferSize( onnxNode.input_size() );
	for( const std::string& inputName : onnxNode.input() ) {
		if( inputName.size() > 0 ) {
			input.Add( nodeOutputs.Get( inputName.data() ) );
		} else {
			input.Add( CInputInfo( nullptr, 0 ) );
		}
	}

	// Adding this onnxNode's outputs to the map of onnxNode outputs.
	for( int outputIndex = 0; outputIndex < onnxNode.output_size(); ++outputIndex ) {
		nodeOutputs.Add( onnxNode.output( outputIndex ).c_str(), CInputInfo( this, outputIndex ) );
	}
}

int CNode::OutputCount() const
{
	return onnxOutputCount;
}

const CTensor& CNode::InputTensor( int index ) const
{
	CheckNeoOnnxInternal( index >= 0 && index < input.Size(),
		"attempt to access non-existing input" );
	CheckNeoOnnxInternal( input[index].InputNode != nullptr,
		"attempt to acces empty input" );
	return input[index].InputNode->outputData[input[index].OutputIndex];
}

CTensor& CNode::InputTensor( int index )
{
	CheckNeoOnnxInternal( index >= 0 && index < input.Size(),
		"attempt to access non-existing input" );
	CheckNeoOnnxInternal( input[index].InputNode != nullptr,
		"attempt to acces empty input" );
	return input[index].InputNode->outputData[input[index].OutputIndex];
}

const CNode::COutputInfo& CNode::InputInfo( int index ) const
{
	CheckNeoOnnxInternal( index >= 0 && index < input.Size(),
		"attempt to access non-existing input" );
	CheckNeoOnnxInternal( input[index].InputNode != nullptr,
		"attempt to access empty input" );
	const CNode& inputNode = *input[index].InputNode;
	const int inputNodeOutputIndex = input[index].OutputIndex;

	NeoAssert( inputNode.outputInfo.Size() == inputNode.outputData.Size() );
	NeoAssert( inputNodeOutputIndex >= 0 && inputNodeOutputIndex < inputNode.outputData.Size() );
	NeoAssert( inputNode.outputInfo[inputNodeOutputIndex].Layer != nullptr );
	return inputNode.outputInfo[inputNodeOutputIndex];
}

const CBaseLayer& CNode::InputLayer( int index ) const
{
	return *InputInfo( index ).Layer;
}

int CNode::InputLayerIndex( int index ) const
{
	return InputInfo( index ).OutputIndex;
}

CNode* CNode::CreateNode( const onnx::NodeProto& onnxNode, CMap<CString, CNode::CInputInfo>& nodeOutputs, IMathEngine& mathEngine )
{
	if( onnxNode.op_type() == "Add" ) {
		return new CAddNode( onnxNode, nodeOutputs );
	} else if( onnxNode.op_type() == "AveragePool" ) {
		return new CAveragePoolNode( onnxNode, nodeOutputs );
	} else if( onnxNode.op_type() == "BatchNormalization" ) {
		return new CBatchNormalizationNode( onnxNode, nodeOutputs );
	} else if( onnxNode.op_type() == "Clip" ) {
		return new CClipNode( onnxNode, nodeOutputs );
	} else if( onnxNode.op_type() == "Concat" ) {
		return new CConcatNode( onnxNode, nodeOutputs );
	} else if( onnxNode.op_type() == "Constant" ) {
		return new CConstantNode( onnxNode, nodeOutputs, mathEngine );
	} else if( onnxNode.op_type() == "ConstantOfShape" ) {
		return new CConstantOfShapeNode( onnxNode, nodeOutputs );
	} else if( onnxNode.op_type() == "Conv" ) {
		return new CConvNode( onnxNode, nodeOutputs );
	} else if( onnxNode.op_type() == "Flatten" ) {
		return new CFlattenNode( onnxNode, nodeOutputs );
	} else if( onnxNode.op_type() == "Gather" ) {
		return new CGatherNode( onnxNode, nodeOutputs );
	} else if( onnxNode.op_type() == "Gemm" ) {
		return new CGemmNode( onnxNode, nodeOutputs );
	} else if( onnxNode.op_type() == "GlobalAveragePool" ) {
		return new CGlobalAveragePoolNode( onnxNode, nodeOutputs );
	} else if( onnxNode.op_type() == "LSTM" ) {
		return new CLstmNode( onnxNode, nodeOutputs );
	} else if( onnxNode.op_type() == "MaxPool" ) {
		return new CMaxPoolNode( onnxNode, nodeOutputs );
	} else if( onnxNode.op_type() == "ReduceMean" ) {
		return new CReduceMeanNode( onnxNode, nodeOutputs );
	} else if( onnxNode.op_type() == "Relu" ) {
		return new CReluNode( onnxNode, nodeOutputs );
	} else if( onnxNode.op_type() == "Shape" ) {
		return new CShapeNode( onnxNode, nodeOutputs, mathEngine );
	} else if( onnxNode.op_type() == "Slice" ) {
		return new CSliceNode( onnxNode, nodeOutputs );
	} else if( onnxNode.op_type() == "Squeeze" ) {
		return new CSqueezeNode( onnxNode, nodeOutputs );
	} else if( onnxNode.op_type() == "Tanh" ) {
		return new CTanhNode( onnxNode, nodeOutputs );
	} else if( onnxNode.op_type() == "Unsqueeze" ) {
		return new CUnsqueezeNode( onnxNode, nodeOutputs );
	}

	CheckNeoOnnxSupport( false, CString( "operator " ) + onnxNode.op_type().c_str() );
	return nullptr;
}

} // namespace NeoOnnx
