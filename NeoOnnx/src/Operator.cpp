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

#include <string>

#include "onnx.pb.h"

#include "Operator.h"
#include "NeoOnnxCheck.h"

#include "Operators/ActivationOperator.h"
#include "Operators/BatchNormalizationOperator.h"
#include "Operators/ConcatOperator.h"
#include "Operators/ConstantOfShapeOperator.h"
#include "Operators/ConstantOperator.h"
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

// Registers the given NeoOnnx class (e.g. CConvOperator) as an onnx operator with given opType ('Conv')
#define REGISTER_OPERATOR( neoOnnxClass, opType ) \
	static COperatorClassRegistrar<neoOnnxClass> __merge__1( _RegisterOperator, __LINE__ )( opType );

typedef COperator* ( *TCreateOperatorFunction )( const onnx::NodeProto& onnxNode, int opsetVersion );

// Returns reference to the map containing info about registered operators
static CMap<CString, TCreateOperatorFunction>& getRegisteredOperators()
{
	static CMap<CString, TCreateOperatorFunction> registeredOperators;
	return registeredOperators;
}

// Registers function as a way to create operator for NodeProto::op_type == operatorType
static void registerOperator( const char* operatorType, TCreateOperatorFunction function )
{
	NeoAssert( !getRegisteredOperators().Has( operatorType ) );
	getRegisteredOperators().Add( operatorType, function );
}

//---------------------------------------------------------------------------------------------------------------------
// Class registers class T as an operator
// Without this registration class will be inaccessible from COperator::CreateOperator
template<class T>
class COperatorClassRegistrar {
public:
	explicit COperatorClassRegistrar( const char* operatorType );

private:
	static COperator* createObject( const onnx::NodeProto& onnxNode, int opsetVersion );
};

template<class T>
inline COperatorClassRegistrar<T>::COperatorClassRegistrar( const char* operatorType )
{
	registerOperator( operatorType, createObject );
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
REGISTER_OPERATOR( CConstantOfShapeOperator, "ConstantOfShape" )
REGISTER_OPERATOR( CConstantOperator, "Constant" )
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

} // anonymous namespace

COperator::COperator( const onnx::NodeProto& onnxNode, int opsetVersion ) :
	OpsetVersion( opsetVersion ),
	name( ( ( onnxNode.name().empty() ? onnxNode.output( 0 ) : onnxNode.name() ) + "_Op" ).c_str() ),
	type( onnxNode.op_type().c_str() )
{
	for( const onnx::AttributeProto& attribute : onnxNode.attribute() ) {
		attributes.Add( attribute.name().c_str(), attribute );
	}

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

} // namespace NeoOnnx

