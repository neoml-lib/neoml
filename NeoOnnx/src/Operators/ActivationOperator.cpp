/* Copyright © 2017-2021 ABBYY Production LLC

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

#include "../common.h"
#pragma hdrstop

#include "onnx.pb.h"

#include "ActivationOperator.h"
#include "NeoOnnxCheck.h"

namespace NeoOnnx {

CActivationOperatorBase::CActivationOperatorBase( const onnx::NodeProto& onnxNode, int opsetVersion,
		TActivationFunction _activation ) :
	CLayerOperator( onnxNode, opsetVersion ),
	activation( _activation )
{
}

void CActivationOperatorBase::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckOnnxProtocol( inputs[0] != nullptr, "input can't be optional", *this );
	CPtr<const CUserTensor> userInput = AsUserTensor( *inputs[0], Name() + "_Source", dnn );

	CPtr<CBaseLayer> activationLayer = CreateActivationLayer( dnn.GetMathEngine(), activation );
	activationLayer->SetName( Name() );
	activationLayer->Connect( 0, *userInput->Layer(), userInput->OutputIndex() );
	dnn.AddLayer( *activationLayer );

	outputs.Add( new CUserTensor( userInput->Shape(), userInput->Layout(), CLayerOutput( activationLayer, 0 ) ) );
}

//---------------------------------------------------------------------------------------------------------------------

CAbsOperator::CAbsOperator( const onnx::NodeProto& abs, int opsetVersion ) :
	CActivationOperatorBase( abs, opsetVersion, AF_Abs )
{
	// v1 - original
	// v6 - legacy optimization attributes are removed and new data types are supported
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

//---------------------------------------------------------------------------------------------------------------------

CClipOperator::CClipOperator( const onnx::NodeProto& clip, int opsetVersion ) :
	CActivationOperatorBase( clip, opsetVersion, AF_ReLU )
{
	// v1 - original
	// v6 - legacy optimization attributes are removed
	// v11 - min/max values are moved from attributes to additional inputs
	// v12 - new data types are supported
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	if( OpsetVersion < 11 ) {
		CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	} else {
		CheckOnnxProtocol( InputCount() >= 1 && InputCount() <= 3, "operator must have from 1 up to 3 inputs", *this );
	}

	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CClipOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	float minValue = -FLT_MAX;
	float maxValue = FLT_MAX;
	if( OpsetVersion < 11 ) {
		GetAttribute( "min", minValue );
		GetAttribute( "max", maxValue );
	} else {
		if( InputCount() > 1 && inputs[1] != nullptr ) {
			const CDataTensor* minValueTensor = dynamic_cast<const CDataTensor*>( inputs[1].Ptr() );
			CheckNeoOnnxSupport( minValueTensor != nullptr, "user-provided clip min value", *this );
			const CDnnBlob* minValueBlob = minValueTensor->Data();
			if( minValueBlob->GetDataType() == CT_Float ) {
				minValue = minValueBlob->GetData<float>().GetValue();
			} else {
				minValue = static_cast<float>( minValueBlob->GetData<int>().GetValue() );
			}
		}

		if( InputCount() > 2 && inputs[2] != nullptr ) {
			const CDataTensor* maxValueTensor = dynamic_cast<const CDataTensor*>( inputs[2].Ptr() );
			CheckNeoOnnxSupport( maxValueTensor != nullptr, "user-provided clip max value", *this );
			const CDnnBlob* maxValueBlob = maxValueTensor->Data();
			if( maxValueBlob->GetDataType() == CT_Float ) {
				maxValue = maxValueBlob->GetData<float>().GetValue();
			} else {
				maxValue = static_cast<float>( maxValueBlob->GetData<int>().GetValue() );
			}
		}
	}

	CTensorArray currInputs;
	inputs.CopyTo( currInputs );
	if( minValue != 0 ) {
		CPtr<const CUserTensor> userInput = AsUserTensor( *currInputs[0], Name() + "_Source", dnn );
		CLinearLayer* preShift = Linear( 1.f, -minValue )( Name() + "_PreShift", CDnnLayerLink( userInput->Layer(), userInput->OutputIndex() ) );
		currInputs[0] = new CUserTensor( userInput->Shape(), userInput->Layout(), CLayerOutput( preShift, 0 ) );
	}

	CActivationOperatorBase::AddLayers( currInputs, dnn, outputs );
	CReLULayer* relu = dynamic_cast<CReLULayer*>( dnn.GetLayer( Name() ).Ptr() );
	NeoAssert( relu != nullptr );
	if( maxValue != FLT_MAX ) {
		relu->SetUpperThreshold( maxValue - minValue );
	}

	if( minValue != 0 ) {
		CLinearLayer* postShift = Linear( 1.f, minValue )( Name() + "_PostShift", relu );
		outputs[0] = new CUserTensor( outputs[0]->Shape(), outputs[0]->Layout(), CLayerOutput( postShift, 0 ) );
	}
}

//---------------------------------------------------------------------------------------------------------------------

CEluOperator::CEluOperator( const onnx::NodeProto& elu, int opsetVersion ) :
	CActivationOperatorBase( elu, opsetVersion, AF_ELU )
{
	// v1 - original
	// v6 - legacy optimization attributes are removed
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CEluOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CActivationOperatorBase::AddLayers( inputs, dnn, outputs );
	CELULayer* elu = dynamic_cast<CELULayer*>( dnn.GetLayer( Name() ).Ptr() );
	NeoAssert( elu != nullptr );

	float alpha = 1;
	GetAttribute( "alpha", alpha );
	elu->SetAlpha( alpha );
}

//---------------------------------------------------------------------------------------------------------------------

CLeakyReluOperator::CLeakyReluOperator( const onnx::NodeProto& leakyRelu, int opsetVersion ) :
	CActivationOperatorBase( leakyRelu, opsetVersion, AF_LeakyReLU )
{
	// v1 - original
	// v6 - legacy optimization attributes are removed
	// v16 - bfloat16 is supported
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CLeakyReluOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CActivationOperatorBase::AddLayers( inputs, dnn, outputs );
	CLeakyReLULayer* leakyReLU = dynamic_cast<CLeakyReLULayer*>( dnn.GetLayer( Name() ).Ptr() );
	NeoAssert( leakyReLU != nullptr );

	float alpha = 0.01f;
	GetAttribute( "alpha", alpha );
	leakyReLU->SetAlpha( alpha );
}

//---------------------------------------------------------------------------------------------------------------------

CHardSigmoidOperator::CHardSigmoidOperator( const onnx::NodeProto& hardSigmoid, int opsetVersion ) :
	CActivationOperatorBase( hardSigmoid, opsetVersion, AF_HardSigmoid )
{
	// v1 - original
	// v6 - legacy optimization attributes are removed
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CHardSigmoidOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CActivationOperatorBase::AddLayers( inputs, dnn, outputs );
	CHardSigmoidLayer* hardSigmoid = dynamic_cast<CHardSigmoidLayer*>( dnn.GetLayer( Name() ).Ptr() );
	NeoAssert( hardSigmoid != nullptr );

	float alpha = 0.2f;
	GetAttribute( "alpha", alpha );
	hardSigmoid->SetSlope( alpha );

	float beta = 0.5f;
	GetAttribute( "beta", beta );
	hardSigmoid->SetBias( beta );
}

//---------------------------------------------------------------------------------------------------------------------

CPowOperator::CPowOperator( const onnx::NodeProto& pow, int opsetVersion ) :
	CActivationOperatorBase( pow, opsetVersion, AF_Power )
{
	// v1 - original
	// v7 - broadcast attribute is removed
	// v12 - integer types are supported
	// v13 - bfloat16 as first argument is supported
	// v15 - bfloat16 as second argument is supported
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 2, "operator must have 2 inputs", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CPowOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	// The only scenario supported by NeoML is when the first input is a float tensor
	// and the second input is a constant float scalar (tensor of size 1)
	CheckOnnxProtocol( inputs[0] != nullptr, "input can't be optional", *this );

	CheckOnnxProtocol( inputs[1] != nullptr, "input can't be optional", *this );
	CheckNeoOnnxSupport( inputs[1]->IsCalculated(), "user-provided power of the exponent", *this );
	CPtr<const CDnnBlob> powerValue = dynamic_cast<const CDataTensor*>( inputs[1].Ptr() )->Data();
	CheckNeoOnnxSupport( powerValue->GetDataSize() == 1, "non-scalar power of the exponent", *this );
	CheckNeoOnnxSupport( powerValue->GetDataType() == CT_Float, "non-float power of the exponent", *this );

	CActivationOperatorBase::AddLayers( inputs, dnn, outputs );
	CPowerLayer* power = dynamic_cast<CPowerLayer*>( dnn.GetLayer( Name() ).Ptr() );
	NeoAssert( power != nullptr );
	power->SetExponent( powerValue->GetData().GetValue() );
}

//---------------------------------------------------------------------------------------------------------------------

CReluOperator::CReluOperator( const onnx::NodeProto& relu, int opsetVersion ) :
	CActivationOperatorBase( relu, opsetVersion, AF_ReLU )
{
	// v1 - original
	// v6 - legacy optimization attributes are removed
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

//---------------------------------------------------------------------------------------------------------------------

CSigmoidOperator::CSigmoidOperator( const onnx::NodeProto& sigmoid, int opsetVersion ) :
	CActivationOperatorBase( sigmoid, opsetVersion, AF_Sigmoid )
{
	// v1 - original
	// v6 - legacy optimization attributes are removed
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

//---------------------------------------------------------------------------------------------------------------------

CSqrtOperator::CSqrtOperator( const onnx::NodeProto& sqrt, int opsetVersion ) :
	CActivationOperatorBase( sqrt, opsetVersion, AF_Power )
{
	// v1 - original
	// v6 - legacy optimization attribute is removed
	// v13 - bfloat16 is supported
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CSqrtOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CActivationOperatorBase::AddLayers( inputs, dnn, outputs );
	CPowerLayer* power = dynamic_cast<CPowerLayer*>( dnn.GetLayer( Name() ).Ptr() );
	NeoAssert( power != nullptr );
	power->SetExponent( 0.5f );
}

//---------------------------------------------------------------------------------------------------------------------

CTanhOperator::CTanhOperator( const onnx::NodeProto& tanh, int opsetVersion ) :
	CActivationOperatorBase( tanh, opsetVersion, AF_Tanh )
{
	// v1 - original
	// v6 - legacy optimization attributes are removed
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

//---------------------------------------------------------------------------------------------------------------------

CExpOperator::CExpOperator( const onnx::NodeProto& exp, int opsetVersion ) :
	CActivationOperatorBase( exp, opsetVersion, AF_Exp )
{
	// v1 - original
	// v6 - legacy optimization attribute is removed
	// v13 - bfloat16 is supported
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

//---------------------------------------------------------------------------------------------------------------------

CLogOperator::CLogOperator( const onnx::NodeProto& log, int opsetVersion ) :
	CActivationOperatorBase( log, opsetVersion, AF_Log )
{
	// v1 - original
	// v6 - legacy optimization attribute is removed
	// v13 - bfloat16 is supported
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

//---------------------------------------------------------------------------------------------------------------------

CErfOperator::CErfOperator( const onnx::NodeProto& erf, int opsetVersion ) :
	CActivationOperatorBase( erf, opsetVersion, AF_Erf )
{
	// v9 - original
	// v13 - bfloat16 is supported
	CheckNeoOnnxSupport( OpsetVersion >= 9 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

//---------------------------------------------------------------------------------------------------------------------

CNegOperator::CNegOperator( const onnx::NodeProto& neg, int opsetVersion ) :
	CActivationOperatorBase( neg, opsetVersion, AF_Linear )
{
	// v1 - original
	// v6 - legacy optimization attribute is removed
	// v13 - bfloat16 is supported
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CNegOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CActivationOperatorBase::AddLayers( inputs, dnn, outputs );
	CLinearLayer* linear = dynamic_cast<CLinearLayer*>( dnn.GetLayer( Name() ).Ptr() );
	NeoAssert( linear != nullptr );
	linear->SetMultiplier( -1.f );
}

} // namespace NeoOnnx

