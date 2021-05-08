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

#include "../common.h"
#pragma hdrstop

#include "ActivationOperator.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CActivationOperatorBase::CActivationOperatorBase( const onnx::NodeProto& onnxNode, int opsetVersion,
		TActivationFunction _activation ) :
	CLayerOperator( onnxNode, opsetVersion ),
	activation( _activation )
{
}

void CActivationOperatorBase::AddLayers( const CObjectArray<const CTensorBase>& inputs,
	CDnn& dnn, CObjectArray<const CTensorBase>& outputs )
{
	const CUserTensor* userInput = dynamic_cast<const CUserTensor*>( inputs[0].Ptr() );
	NeoAssert( userInput != nullptr );
	CPtr<CBaseLayer> activationLayer = CreateActivationLayer( dnn.GetMathEngine(), activation );
	activationLayer->SetName( Name() );
	SetLayerParams( inputs, activationLayer );
	activationLayer->Connect( 0, *userInput->Layer(), userInput->OutputIndex() );
	dnn.AddLayer( *activationLayer );
	outputs[0] = new CUserTensor( userInput->Shape(), userInput->Layout(), CLayerOutput( activationLayer, 0 ) );
}

//---------------------------------------------------------------------------------------------------------------------

CAbsOperator::CAbsOperator( const onnx::NodeProto& abs, int opsetVersion ) :
	CActivationOperatorBase( abs, opsetVersion, AF_Abs )
{
}

void CAbsOperator::CheckOnnxNode() const
{
	// v1 - original
	// v6 - removed legacy optimization attributes and added new data types support
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

//---------------------------------------------------------------------------------------------------------------------

CClipOperator::CClipOperator( const onnx::NodeProto& clip, int opsetVersion ) :
	CActivationOperatorBase( clip, opsetVersion, AF_ReLU )
{
}

void CClipOperator::CheckOnnxNode() const
{
	// v1 - original
	// v6 - removed legacy optimization attributes
	// v11 - min/max values were moved from attributes to additional inputs
	// v12 - new data types supported
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	if( OpsetVersion < 11 ) {
		CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	} else {
		CheckOnnxProtocol( InputCount() >= 1 || InputCount() <= 3, "operator must have from 1 up to 3 inputs", *this );
	}

	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CClipOperator::SetLayerParams( const CObjectArray<const CTensorBase>& inputs, CBaseLayer* layer ) const
{
	CReLULayer* relu = dynamic_cast<CReLULayer*>( layer );
	NeoAssert( relu != nullptr );

	float minValue = -FLT_MAX;
	float maxValue = FLT_MAX;
	if( OpsetVersion < 11 ) {
		minValue = Attributes.GetOptionalFloat( "min", -FLT_MAX );
		maxValue = Attributes.GetOptionalFloat( "max", FLT_MAX );
	} else if( InputCount() > 1 ) {
		const CDataTensor* minValueTensor = dynamic_cast<const CDataTensor*>( inputs[1].Ptr() );
		CheckNeoOnnxSupport( minValueTensor != nullptr, "user-provided clip min value", *this );
		const CDnnBlob* minValueBlob = minValueTensor->Data();
		if( minValueBlob->GetDataType() == CT_Float ) {
			minValue = minValueBlob->GetData<float>().GetValue();
		} else {
			minValue = static_cast<float>( minValueBlob->GetData<int>().GetValue() );
		}

		if( InputCount() > 2 ) {
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

	CheckNeoOnnxSupport( minValue == 0, "Clip with non-zero min value", *this );
	if( maxValue != FLT_MAX ) {
		relu->SetUpperThreshold( maxValue );
	}
}

//---------------------------------------------------------------------------------------------------------------------

CEluOperator::CEluOperator( const onnx::NodeProto& elu, int opsetVersion ) :
	CActivationOperatorBase( elu, opsetVersion, AF_ELU )
{
}

void CEluOperator::CheckOnnxNode() const
{
	// v1 - original
	// v6 - removed legacy optimization attributes
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

//---------------------------------------------------------------------------------------------------------------------

CLeakyReluOperator::CLeakyReluOperator( const onnx::NodeProto& leakyRelu, int opsetVersion ) :
	CActivationOperatorBase( leakyRelu, opsetVersion, AF_LeakyReLU )
{
}

void CLeakyReluOperator::CheckOnnxNode() const
{
	// v1 - original
	// v6 - removed legacy optimization attributes
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CLeakyReluOperator::SetLayerParams( const CObjectArray<const CTensorBase>& /* inputs */, CBaseLayer* layer ) const
{
	CLeakyReLULayer* leakyReLU = dynamic_cast<CLeakyReLULayer*>( layer );
	NeoAssert( leakyReLU != nullptr );
	leakyReLU->SetAlpha( Attributes.GetOptionalFloat( "alpha", 0.f ) );
}

//---------------------------------------------------------------------------------------------------------------------

CHardSigmoidOperator::CHardSigmoidOperator( const onnx::NodeProto& hardSigmoid, int opsetVersion ) :
	CActivationOperatorBase( hardSigmoid, opsetVersion, AF_HardSigmoid )
{
}

void CHardSigmoidOperator::CheckOnnxNode() const
{
	// v1 - original
	// v6 - removed legacy optimization attributes
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CHardSigmoidOperator::SetLayerParams( const CObjectArray<const CTensorBase>& /* inputs */, CBaseLayer* layer ) const
{
	CHardSigmoidLayer* hardSigmoid = dynamic_cast<CHardSigmoidLayer*>( layer );
	NeoAssert( hardSigmoid != nullptr );

	const float alpha = Attributes.GetOptionalFloat( "alpha", 0.2f );
	const float beta = Attributes.GetOptionalFloat( "beta", 0.5f );

	hardSigmoid->SetSlope( alpha );
	hardSigmoid->SetBias( beta );
}

//---------------------------------------------------------------------------------------------------------------------

CReluOperator::CReluOperator( const onnx::NodeProto& relu, int opsetVersion ) :
	CActivationOperatorBase( relu, opsetVersion, AF_ReLU )
{
}

void CReluOperator::CheckOnnxNode() const
{
	// v1 - original
	// v6 - removed legacy optimization attributes
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

//---------------------------------------------------------------------------------------------------------------------

CSigmoidOperator::CSigmoidOperator( const onnx::NodeProto& sigmoid, int opsetVersion ) :
	CActivationOperatorBase( sigmoid, opsetVersion, AF_Sigmoid )
{
}

void CSigmoidOperator::CheckOnnxNode() const
{
	// v1 - original
	// v6 - removed legacy optimization attributes
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

//---------------------------------------------------------------------------------------------------------------------

CTanhOperator::CTanhOperator( const onnx::NodeProto& tanh, int opsetVersion ) :
	CActivationOperatorBase( tanh, opsetVersion, AF_Tanh )
{
}

void CTanhOperator::CheckOnnxNode() const
{
	// v1 - original
	// v6 - removed legacy optimization attributes
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

} // namespace NeoOnnx
