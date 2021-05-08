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

#include "Operator.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

#include <string>

#include "Operators/ActivationOperator.h"
#include "Operators/BatchNormalizationOperator.h"
#include "Operators/ConcatOperator.h"
#include "Operators/ConstantOperator.h"
#include "Operators/ConstantOfShapeOperator.h"
#include "Operators/ConvOperator.h"
#include "Operators/DropoutOperator.h"
#include "Operators/EltwiseOperator.h"
#include "Operators/FlattenOperator.h"
#include "Operators/GatherOperator.h"
#include "Operators/GemmOperator.h"
#include "Operators/GlobalPoolOperator.h"
#include "Operators/IdentityOperator.h"
#include "Operators/LrnOperator.h"
#include "Operators/LstmOperator.h"
#include "Operators/MatMulOperator.h"
#include "Operators/PadOperator.h"
#include "Operators/PoolOperator.h"
#include "Operators/ReshapeOperator.h"
#include "Operators/ShapeOperator.h"
#include "Operators/SliceOperator.h"
#include "Operators/SoftmaxOperator.h"
#include "Operators/SqueezeOperator.h"
#include "Operators/TransposeOperator.h"
#include "Operators/UnsqueezeOperator.h"

namespace NeoOnnx {

// Registers the class as a NeoOnnx operator for op_type == opName
#define REGISTER_OPERATOR( classType, opName ) \
	static COperatorClassRegistrar< classType > __merge__1( _RegisterOperator, __LINE__ )( opName );

typedef COperator* ( *TCreateOperatorFunction )( const onnx::NodeProto& onnxNode, int opsetVersion );

// Returns reference to the map containing info about registered operators
static CMap<CString, TCreateOperatorFunction>& getRegisteredOperators()
{
	static CMap<CString, TCreateOperatorFunction> registeredOperators;
	return registeredOperators;
}

// Registers function as a way to create operator for NodeProto::op_type == operatorName
void registerOperator( const char* operatorName, TCreateOperatorFunction function )
{
	NeoAssert( !getRegisteredOperators().Has( operatorName ) );
	getRegisteredOperators().Add( operatorName, function );
}

//---------------------------------------------------------------------------------------------------------------------
// Class registers class T as an operator
// Without this registration class will be inaccessible from COperator::CreateOperator
template<class T>
class COperatorClassRegistrar {
public:
	explicit COperatorClassRegistrar( const char* operatorName );

private:
	static COperator* createObject( const onnx::NodeProto& onnxNode, int opsetVersion );
};

template<class T>
inline COperatorClassRegistrar<T>::COperatorClassRegistrar( const char* operatorName )
{
	registerOperator( operatorName, createObject );
}

template<class T>
inline COperator* COperatorClassRegistrar<T>::createObject( const onnx::NodeProto& onnxNode, int opsetVersion )
{
	return FINE_DEBUG_NEW T( onnxNode, opsetVersion );
}

//---------------------------------------------------------------------------------------------------------------------

namespace {

// Register all operators
REGISTER_OPERATOR( CAbsOperator, "Abs" )
REGISTER_OPERATOR( CAddOperator, "Add" )
REGISTER_OPERATOR( CAveragePoolOperator, "AveragePool" )
REGISTER_OPERATOR( CBatchNormalizationOperator, "BatchNormalization" )
REGISTER_OPERATOR( CClipOperator, "Clip" )
REGISTER_OPERATOR( CConcatOperator, "Concat" )
REGISTER_OPERATOR( CConstantOperator, "Constant" )
REGISTER_OPERATOR( CConstantOfShapeOperator, "ConstantOfShape" )
REGISTER_OPERATOR( CConvOperator, "Conv" )
REGISTER_OPERATOR( CDivOperator, "Div" )
REGISTER_OPERATOR( CDropoutOperator, "Dropout" )
REGISTER_OPERATOR( CEluOperator, "Elu" )
REGISTER_OPERATOR( CFlattenOperator, "Flatten" )
REGISTER_OPERATOR( CGatherOperator, "Gather" )
REGISTER_OPERATOR( CGemmOperator, "Gemm" )
REGISTER_OPERATOR( CGlobalAveragePoolOperator, "GlobalAveragePool" )
REGISTER_OPERATOR( CGlobalMaxPoolOperator, "GlobalMaxPool" )
REGISTER_OPERATOR( CHardSigmoidOperator, "HardSigmoid" )
REGISTER_OPERATOR( CIdentityOperator, "Identity" )
REGISTER_OPERATOR( CLeakyReluOperator, "LeakyRelu" )
REGISTER_OPERATOR( CLrnOperator, "LRN" )
REGISTER_OPERATOR( CLstmOperator, "LSTM" )
REGISTER_OPERATOR( CMatMulOperator, "MatMul" )
REGISTER_OPERATOR( CMaxPoolOperator, "MaxPool" )
REGISTER_OPERATOR( CMulOperator, "Mul" )
REGISTER_OPERATOR( CPadOperator, "Pad" )
REGISTER_OPERATOR( CReduceMaxOperator, "ReduceMax" )
REGISTER_OPERATOR( CReduceMeanOperator, "ReduceMean" )
REGISTER_OPERATOR( CReduceMinOperator, "ReduceMin" )
REGISTER_OPERATOR( CReluOperator, "Relu" )
REGISTER_OPERATOR( CReshapeOperator, "Reshape" )
REGISTER_OPERATOR( CShapeOperator, "Shape" )
REGISTER_OPERATOR( CSigmoidOperator, "Sigmoid" )
REGISTER_OPERATOR( CSliceOperator, "Slice" )
REGISTER_OPERATOR( CSoftmaxOperator, "Softmax" )
REGISTER_OPERATOR( CSqueezeOperator, "Squeeze" )
REGISTER_OPERATOR( CSubOperator, "Sub" )
REGISTER_OPERATOR( CSumOperator, "Sum" )
REGISTER_OPERATOR( CTanhOperator, "Tanh" )
REGISTER_OPERATOR( CTransposeOperator, "Transpose" )
REGISTER_OPERATOR( CUnsqueezeOperator, "Unsqueeze" )

} // namespace

COperator::COperator( const onnx::NodeProto& onnxNode, int opsetVersion ) :
	OpsetVersion( opsetVersion ),
	Attributes( onnxNode, *this ),
	name( ( ( onnxNode.name().empty() ? onnxNode.output( 0 ) : onnxNode.name() ) + "_Op" ).c_str() ),
	type( onnxNode.op_type().c_str() )
{
	for( const std::string& inputName : onnxNode.input() ) {
		inputNames.Add( CString( inputName ) );
	}

	for( const std::string& outputName : onnxNode.output() ) {
		outputNames.Add( CString( outputName ) );
	}
}

const CString& COperator::InputName( int index ) const
{
	NeoAssert( index >= 0 && index < InputCount() );
	return inputNames[index];
}

const CString& COperator::OutputName( int index ) const
{
	NeoAssert( index >= 0 && index < OutputCount() );
	return outputNames[index];
}

bool COperator::CanCalculateOutput( const CObjectArray<const CTensorBase>& inputs ) const
{
	for( int inputIndex = 0; inputIndex < inputs.Size(); ++inputIndex ) {
		if( inputs[inputIndex] != nullptr && !inputs[inputIndex]->IsCalculated() ) {
			return false;
		}
	}

	return true;
}

COperator* COperator::CreateOperator( const onnx::NodeProto& onnxNode, int opsetVersion )
{
	TMapPosition pos = getRegisteredOperators().GetFirstPosition( onnxNode.op_type() );
	CheckNeoOnnxSupport( pos != NotFound, CString( "operator " ) + onnxNode.op_type().c_str() );
	return getRegisteredOperators().GetValue( pos )( onnxNode, opsetVersion );
}

bool COperator::IsSupportedOperator( const CString& operatorType )
{
	TMapPosition pos = getRegisteredOperators().GetFirstPosition( operatorType );
	return pos != NotFound;
}

//---------------------------------------------------------------------------------------------------------------------

void CLayerOperator::CalculateOutput( const CObjectArray<const CTensorBase>& inputs,
	IMathEngine& mathEngine, CObjectArray<const CTensorBase>& outputs )
{
	CRandom random( 0x1231 );
	CDnn internalDnn( random, mathEngine );

	// Add source layers for the operator
	CObjectArray<const CTensorBase> internalInputs;
	addInternalDnnSources( inputs, internalInputs, internalDnn );

	// Add operator layers
	CObjectArray<const CTensorBase> internalOutputs;
	internalOutputs.Add( nullptr, OutputCount() );
	AddLayers( internalInputs, internalDnn, internalOutputs );

	// Add sink layers for the operator
	CArray<CSinkLayer*> sinks;
	addInternalDnnSinks( internalOutputs, sinks, internalDnn );

	// Launch the dnn in order to calculate values
	internalDnn.RunOnce();

	// Extract values from the net
	extractOutputs( internalOutputs, sinks, outputs );
}

// Builds array of tensors related to the internal dnn
// Also adds required source layers to the internal dnn (with corresponding blobs)
void CLayerOperator::addInternalDnnSources( const CObjectArray<const CTensorBase>& inputs,
	CObjectArray<const CTensorBase>& internalInputs, CDnn& internalDnn ) const
{
	IMathEngine& mathEngine = internalDnn.GetMathEngine();

	CUserInputMask isUserInput;
	UserInputMask( isUserInput );

	for( int inputIndex = 0; inputIndex < InputCount(); ++inputIndex ) {
		if( inputs[inputIndex] == nullptr || !inputs[inputIndex]->IsCalculated() ) {
			internalInputs.Add( nullptr );
		} else if( isUserInput[inputIndex] ) {
			NeoAssert( inputs[inputIndex]->IsCalculated() );
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
void CLayerOperator::addInternalDnnSinks( const CObjectArray<const CTensorBase>& internalOutputs,
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
void CLayerOperator::extractOutputs( const CObjectArray<const CTensorBase>& internalOutputs,
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
