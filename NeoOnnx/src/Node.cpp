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

#include "Nodes/ActivationNode.h"
#include "Nodes/BatchNormalizationNode.h"
//#include "Nodes/ConcatNode.h"
//#include "Nodes/ConstantNode.h"
//#include "Nodes/ConstantOfShapeNode.h"
#include "Nodes/ConvNode.h"
//#include "Nodes/DropoutNode.h"
#include "Nodes/EltwiseNode.h"
#include "Nodes/FlattenNode.h"
//#include "Nodes/GatherNode.h"
#include "Nodes/GemmNode.h"
#include "Nodes/GlobalPoolNode.h"
//#include "Nodes/IdentityNode.h"
//#include "Nodes/LrnNode.h"
//#include "Nodes/LstmNode.h"
//#include "Nodes/MatMulNode.h"
#include "Nodes/PadNode.h"
#include "Nodes/PoolNode.h"
//#include "Nodes/ReshapeNode.h"
//#include "Nodes/ShapeNode.h"
//#include "Nodes/SliceNode.h"
//#include "Nodes/SoftmaxNode.h"
//#include "Nodes/SqueezeNode.h"
//#include "Nodes/TransposeNode.h"
//#include "Nodes/UnsqueezeNode.h"

namespace NeoOnnx {

// Registers the class as a NeoOnnx node for op_type == opName
#define REGISTER_OP_NODE( classType, opName ) \
	static CNodeClassRegistrar< classType > __merge__1( _RegisterOpNode, __LINE__ )( opName );

typedef COpNode* ( *TCreateOpNodeFunction )( const onnx::NodeProto& onnxNode, int opsetVersion );

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
	static COpNode* createObject( const onnx::NodeProto& onnxNode, int opsetVersion );
};

template<class T>
inline CNodeClassRegistrar<T>::CNodeClassRegistrar( const char* opName )
{
	registerNode( opName, createObject );
}

template<class T>
inline COpNode* CNodeClassRegistrar<T>::createObject( const onnx::NodeProto& onnxNode, int opsetVersion )
{
	return FINE_DEBUG_NEW T( onnxNode, opsetVersion );
}

//---------------------------------------------------------------------------------------------------------------------

namespace {

// Register all nodes
REGISTER_OP_NODE( CAbsNode, "Abs" )
REGISTER_OP_NODE( CAddNode, "Add" )
REGISTER_OP_NODE( CAveragePoolNode, "AveragePool" )
REGISTER_OP_NODE( CBatchNormalizationNode, "BatchNormalization" )
REGISTER_OP_NODE( CClipNode, "Clip" )
//REGISTER_OP_NODE( CConcatNode, "Concat" )
//REGISTER_OP_NODE( CConstantNode, "Constant" )
//REGISTER_OP_NODE( CConstantOfShapeNode, "ConstantOfShape" )
REGISTER_OP_NODE( CConvNode, "Conv" )
REGISTER_OP_NODE( CDivNode, "Div" )
//REGISTER_OP_NODE( CDropoutNode, "Dropout" )
REGISTER_OP_NODE( CEluNode, "Elu" )
REGISTER_OP_NODE( CFlattenNode, "Flatten" )
//REGISTER_OP_NODE( CGatherNode, "Gather" )
REGISTER_OP_NODE( CGemmNode, "Gemm" )
REGISTER_OP_NODE( CGlobalAveragePoolNode, "GlobalAveragePool" )
REGISTER_OP_NODE( CGlobalMaxPoolNode, "GlobalMaxPool" )
REGISTER_OP_NODE( CHardSigmoidNode, "HardSigmoid" )
//REGISTER_OP_NODE( CIdentityNode, "Identity" )
REGISTER_OP_NODE( CLeakyReluNode, "LeakyRelu" )
//REGISTER_OP_NODE( CLrnNode, "LRN" )
//REGISTER_OP_NODE( CLstmNode, "LSTM" )
//REGISTER_OP_NODE( CMatMulNode, "MatMul" )
REGISTER_OP_NODE( CMaxPoolNode, "MaxPool" )
REGISTER_OP_NODE( CMulNode, "Mul" )
REGISTER_OP_NODE( CPadNode, "Pad" )
REGISTER_OP_NODE( CReduceMaxNode, "ReduceMax" )
REGISTER_OP_NODE( CReduceMeanNode, "ReduceMean" )
REGISTER_OP_NODE( CReduceMinNode, "ReduceMin" )
REGISTER_OP_NODE( CReluNode, "Relu" )
//REGISTER_OP_NODE( CReshapeNode, "Reshape" )
//REGISTER_OP_NODE( CShapeNode, "Shape" )
REGISTER_OP_NODE( CSigmoidNode, "Sigmoid" )
//REGISTER_OP_NODE( CSliceNode, "Slice" )
//REGISTER_OP_NODE( CSoftmaxNode, "Softmax" )
//REGISTER_OP_NODE( CSqueezeNode, "Squeeze" )
REGISTER_OP_NODE( CSubNode, "Sub" )
REGISTER_OP_NODE( CSumNode, "Sum" )
REGISTER_OP_NODE( CTanhNode, "Tanh" )
//REGISTER_OP_NODE( CTransposeNode, "Transpose" )
//REGISTER_OP_NODE( CUnsqueezeNode, "Unsqueeze" )

} // namespace

//---------------------------------------------------------------------------------------------------------------------

CNode::CNode( const CString& _name, const CArray<CString>& inputs, const CArray<CString>& outputs ) :
	name( _name )
{
	inputs.CopyTo( inputNames );
	outputs.CopyTo( outputNames );
}

CNode::CNode( const CString& _name, const ::google::protobuf::RepeatedPtrField<std::string>& inputs,
		const::google::protobuf::RepeatedPtrField<std::string>& outputs ) :
	name( _name )
{
	for( const std::string& input : inputs ) {
		inputNames.Add( CString( input ) );
	}

	for( const std::string& output : outputs ) {
		outputNames.Add( CString( output ) );
	}
}

const CString& CNode::InputName( int index ) const
{
	CheckNeoOnnxInternal( index >= 0 && index < InputCount(), "Access to non-existing input" );
	return inputNames[index];
}

const CString& CNode::OutputName( int index ) const
{
	CheckNeoOnnxInternal( index >= 0 && index < OutputCount(), "Access to non-existing output" );
	return outputNames[index];
}

bool CNode::CanCalculateOutput( const CObjectArray<const CTensorBase>& inputs ) const
{
	for( int inputIndex = 0; inputIndex < inputs.Size(); ++inputIndex ) {
		if( inputs[inputIndex] != nullptr && !inputs[inputIndex]->IsCalculated() ) {
			return false;
		}
	}

	return true;
}

//---------------------------------------------------------------------------------------------------------------------

COpNode::COpNode( const onnx::NodeProto& onnxNode, int opsetVersion ) :
	CNode( ( onnxNode.name().empty() ? onnxNode.output( 0 ) : onnxNode.name() ) + "_Op",
		onnxNode.input(), onnxNode.output() ),
	OpsetVersion( opsetVersion ),
	Attributes( onnxNode ),
	OnnxNode( onnxNode )
{
}

void COpNode::CalculateOutput( const CObjectArray<const CTensorBase>& inputs,
	CObjectArray<const CTensorBase>& outputs, IMathEngine& mathEngine )
{
	CRandom random( 0x1231 );
	CDnn internalDnn( random, mathEngine );

	// Add source layers for the operator
	CObjectArray<const CTensorBase> internalInputs;
	addInternalDnnSources( inputs, internalInputs, internalDnn );

	// Add operator layers
	CObjectArray<const CTensorBase> internalOutputs;
	internalOutputs.Add( nullptr, OutputCount() );
	AddLayers( internalInputs, internalOutputs, internalDnn );

	// Add sink layers for the operator
	CArray<CSinkLayer*> sinks;
	addInternalDnnSinks( internalOutputs, sinks, internalDnn );

	// Launch the dnn in order to calculate values
	internalDnn.RunOnce();

	// Extract values from the net
	extractOutputs( internalOutputs, sinks, outputs );
}

COpNode* COpNode::CreateOpNode( const onnx::NodeProto& onnxNode, int opsetVersion )
{
	TMapPosition pos = getRegisteredNodes().GetFirstPosition( onnxNode.op_type() );
	CheckNeoOnnxSupport( pos != NotFound, CString( "operator " ) + onnxNode.op_type().c_str() );
	return getRegisteredNodes().GetValue( pos )( onnxNode, opsetVersion );
}

bool COpNode::IsSupportedOperator( const CString& opType )
{
	TMapPosition pos = getRegisteredNodes().GetFirstPosition( opType );
	return pos != NotFound;
}

// Builds array of tensors related to the internal dnn
// Also adds required source layers to the internal dnn (with corresponding blobs)
void COpNode::addInternalDnnSources( const CObjectArray<const CTensorBase>& inputs,
	CObjectArray<const CTensorBase>& internalInputs, CDnn& internalDnn ) const
{
	IMathEngine& mathEngine = internalDnn.GetMathEngine();

	CUserInputMask isUserInput;
	UserInputMask( isUserInput );
	
	for( int inputIndex = 0; inputIndex < InputCount(); ++inputIndex ) {
		if( inputs[inputIndex] == nullptr || !inputs[inputIndex]->IsCalculated() ) {
			internalInputs.Add( nullptr );
		} else if( isUserInput[inputIndex] ) {
			CheckNeoOnnxInternal( inputs[inputIndex]->IsCalculated(), "Can't pass user input into internal net", OnnxNode );
			CPtr<CSourceLayer> source = new CSourceLayer( mathEngine );
			source->SetName( InputName( inputIndex ) );
			internalDnn.AddLayer( *source );
			source->SetBlob( dynamic_cast<const CDataTensor*>( inputs[inputIndex].Ptr() )->Data()->GetCopy() );
			internalInputs.Add( new CUserTensor( inputs[inputIndex]->Shape(), inputs[inputIndex]->Layout(),
				CLayerOutput( source, 0 ) ) );
		} else {
			internalInputs.Add( inputs[inputIndex] );
		}
	}
}

// Builds array of sinks (corresponding to the op outputs)
// Also adds those layers to the dnn
void COpNode::addInternalDnnSinks( const CObjectArray<const CTensorBase>& internalOutputs,
	CArray<CSinkLayer*>& sinks, CDnn& internalDnn ) const
{
	IMathEngine& mathEngine = internalDnn.GetMathEngine();

	for( int outputIndex = 0; outputIndex < OutputCount(); ++outputIndex ) {
		if( internalOutputs[outputIndex] == nullptr || internalOutputs[outputIndex]->IsCalculated() ) {
			sinks.Add( nullptr );
		} else {
			CPtr<CSinkLayer> sink = new CSinkLayer( mathEngine );
			sink->SetName( OutputName( outputIndex ) + "_Sink" );
			internalDnn.AddLayer( *sink );
			const CLayerOutput& connectedOutput = dynamic_cast<const CUserTensor*>( internalOutputs[outputIndex].Ptr() )->LayerOutput();
			sink->Connect( 0, *connectedOutput.Layer, connectedOutput.OutputIndex );
			sinks.Add( sink.Ptr() );
		}
	}
}

// Builds array of the operator outputs based on outputs of the internal dnn
void COpNode::extractOutputs( const CObjectArray<const CTensorBase>& internalOutputs,
	const CArray<CSinkLayer*>& sinks, CObjectArray<const CTensorBase>& outputs ) const
{
	for( int outputIndex = 0; outputIndex < OutputCount(); ++outputIndex ) {
		if( internalOutputs[outputIndex]->IsCalculated() ) {
			// This data was calculated prior to the net
			// TODO: is this possible?
			outputs[outputIndex] = internalOutputs[outputIndex];
		} else if( sinks[outputIndex] != nullptr ) {
			// Adding network result as data tensor
			// Shape and layout remains unchanged
			outputs[outputIndex] = new CDataTensor( internalOutputs[outputIndex]->Shape(),
				internalOutputs[outputIndex]->Layout(), *( sinks[outputIndex]->GetBlob() ) );
		} // otherwise leaving it as nullptr
	}
}

} // namespace NeoOnnx
