/* Copyright © 2017-2022 ABBYY Production LLC

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

#include "LogicalOperator.h"

namespace NeoOnnx {

CNotOperator::CNotOperator( const onnx::NodeProto& notNode, int opsetVersion ) :
	CLayerOperator( notNode, opsetVersion )
{
	// v1 - original
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CNotOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckOnnxProtocol( inputs[0] != nullptr, "input can't be optional", *this );
	CPtr<const CUserTensor> input = AsUserTensor( *inputs[0], Name() + "_Source", dnn );

	CNotLayer* notLayer = Not()( Name(), CDnnLayerLink( input->Layer(), input->OutputIndex() ) );
	outputs.Add( new CUserTensor( input->Shape(), input->Layout(), CLayerOutput( notLayer, 0 ) ) );
}

// --------------------------------------------------------------------------------------------------------------------

void CLessOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CPtr<CBaseLayer> layer( new CLessLayer( dnn.GetMathEngine() ) );
	CEltwiseOperatorBase::AddLayersImpl( Broadcast(), inputs, *layer, dnn, outputs );
	if( Type() == "Greater" || Type() == "LessOrEqual" ) {
		CString firstName = layer->GetInputName( 0 );
		int firstOutput = layer->GetInputOutputNumber( 0 );
		CString secondName = layer->GetInputName( 1 );
		int secondOutput = layer->GetInputOutputNumber( 1 );
		layer->Connect( 0, secondName, secondOutput );
		layer->Connect( 1, firstName, firstOutput );
	}
	if( Type() == "LessOrEqual" || Type() == "GreaterOrEqual" ) {
		NeoAssert( !outputs[0]->IsCalculated() );
		CPtr<const CUserTensor> currOutput = dynamic_cast<const CUserTensor*>( outputs[0].Ptr() );
		CNotLayer* notLayer = Not()( Name() + "_PostNot", CDnnLayerLink( currOutput->Layer(), currOutput->OutputIndex() ) );
		outputs[0] = new CUserTensor( currOutput->Shape(), currOutput->Layout(), CLayerOutput( notLayer, 0 ) );
	}
}

// --------------------------------------------------------------------------------------------------------------------

void CEqualOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CPtr<CBaseLayer> layer( new CEqualLayer( dnn.GetMathEngine() ) );
	CEltwiseOperatorBase::AddLayersImpl( Broadcast(), inputs, *layer, dnn, outputs );
}

// --------------------------------------------------------------------------------------------------------------------

CWhereOperator::CWhereOperator( const onnx::NodeProto& whereNode, int opsetVersion ) :
	CEltwiseOperatorBase( whereNode, opsetVersion, 3 )
{
	// v9 - original
	// v16 - bfloat16 data type supported
	CheckNeoOnnxSupport( OpsetVersion >= 9 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );
}

void CWhereOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CBroadcast broadcast( BT_Numpy );
	CPtr<CBaseLayer> layer( new CWhereLayer( dnn.GetMathEngine() ) );
	CEltwiseOperatorBase::AddLayersImpl( broadcast, inputs, *layer, dnn, outputs );
}

} // namespace NeoOnnx