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

#include "onnx.pb.h"

#include "EltwiseOperator.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

namespace NeoOnnx {

CEltwiseOperatorBase::CEltwiseOperatorBase( const onnx::NodeProto& eltwise, int opsetVersion, int _argsNum ) :
	CLayerOperator( eltwise, opsetVersion ),
	argsNum( _argsNum )
{
	if( argsNum > 0 ) {
		CheckOnnxProtocol( InputCount() == argsNum, "expected " + Str( argsNum ) + " arguments", *this );
	}
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CEltwiseOperatorBase::AddLayersImpl( const CBroadcast& broadcast, const CTensorArray& inputs,
	CBaseLayer& eltwiseLayer, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckOnnxProtocol( inputs[0] != nullptr, "input can't be optional", *this );

	// Corner case which doesn't violate Onnx protocol: operators with variable input count may have 1 input
	if( inputs.Size() == 1 && argsNum < 0 ) {
		outputs[0] = inputs[0];
		return;
	}

	// Calculate outputShape
	CTensorShape outputShape;
	inputs[0]->Shape().CopyTo( outputShape );
	for( int i = 1; i < inputs.Size(); ++i ) {
		CheckOnnxProtocol( inputs[i] != nullptr, "input can't be optional", *this );
		CTensorShape buff;
		CheckNeoOnnxSupport( BroadcastTensorShape( outputShape, inputs[i]->Shape(), broadcast, buff ),
			"Can't broadcast tensors shape", *this );
		buff.CopyTo( outputShape );
	}

	CTensorArray currInputs;
	inputs.CopyTo( currInputs );
	// Broadcast input to the final shape and set proper layout
	for( int i = 0; i < inputs.Size(); ++i ) {
		currInputs[i] = BroadcastTensor( *currInputs[i], broadcast, outputShape );
		currInputs[i] = ConvertTensor( *currInputs[i], currInputs[0]->Layout() );
	}

	// Put pre-calculated blobs to the source in the net
	for( int i = 0; i < currInputs.Size(); ++i ) {
		if( currInputs[i]->IsCalculated() ) {
			CPtr<CSourceLayer> source = new CSourceLayer( dnn.GetMathEngine() );
			source->SetName( Name() + "_input_" + Str( i ) );
			source->SetBlob( dynamic_cast<const CDataTensor*>( currInputs[i].Ptr() )->Data()->GetCopy() );
			// Save this pre-calculated data during serialization
			// Otherwise dnn won't be able to work correctly after serialization
			source->StoreBlob( true );
			dnn.AddLayer( *source );
			currInputs[i] = new CUserTensor( currInputs[i]->Shape(), currInputs[i]->Layout(), CLayerOutput( source, 0 ) );
		}
	}

	eltwiseLayer.SetName( Name() );
	for( int i = 0; i < currInputs.Size(); ++i ) {
		NeoAssert( !currInputs[i]->IsCalculated() );
		const CUserTensor* userInput = dynamic_cast<const CUserTensor*>( currInputs[i].Ptr() );
		eltwiseLayer.Connect( i, *userInput->Layer(), userInput->OutputIndex() );
	}
	dnn.AddLayer( eltwiseLayer );
	outputs.Add( new CUserTensor( outputShape, currInputs[0]->Layout(), CLayerOutput( &eltwiseLayer, 0 ) ) );
}

// --------------------------------------------------------------------------------------------------------------------

CBroadcast CEltwiseBinaryOperatorBase::Broadcast() const
{
	CBroadcast broadcast( BT_Numpy, NotFound );
	if( OpsetVersion < 7 ) {
		int broadcastAttr = 0;
		GetAttribute( "broadcast", broadcastAttr );
		if( broadcastAttr != 0 ) {
			broadcast.Type = BT_Onnx;
			GetAttribute( "axis", broadcast.Axis );
		} else {
			broadcast.Type = BT_None;
		}
	}

	return broadcast;
}

// --------------------------------------------------------------------------------------------------------------------

void CAddOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CPtr<CBaseLayer> layer( new CEltwiseSumLayer( dnn.GetMathEngine() ) );
	CEltwiseOperatorBase::AddLayersImpl( Broadcast(), inputs, *layer, dnn, outputs );
}

// --------------------------------------------------------------------------------------------------------------------

void CSubOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CTensorArray convertedInputs;
	inputs.CopyTo( convertedInputs );

	// a - b = a + (-1 * b)
	CheckOnnxProtocol( inputs[1] != nullptr, "input isn't connected", *this );
	if( inputs[1]->IsCalculated() ) {
		// Imitate by multiplying values in blob
		CPtr<const CDataTensor> secondInput = dynamic_cast<const CDataTensor*>( inputs[1].Ptr() );
		CPtr<CDnnBlob> newBlob = secondInput->Data()->GetClone();
		CFloatHandleStackVar minusOne( newBlob->GetMathEngine() );
		minusOne.SetValue( -1 );
		newBlob->GetMathEngine().VectorMultiply( secondInput->Data()->GetData(), newBlob->GetData(),
			newBlob->GetDataSize(), minusOne );
		convertedInputs[1] = new CDataTensor( secondInput->Shape(), secondInput->Layout(), *newBlob );
	} else {
		// Imitate by CLinearLayer with multiplier set to -1 and free term set to 0 
		CPtr<const CUserTensor> secondInput = dynamic_cast<const CUserTensor*>( inputs[1].Ptr() );
		CDnn& dnn = *secondInput->Layer()->GetDnn();
		CPtr<CLinearLayer> linear = new CLinearLayer( dnn.GetMathEngine() );
		linear->SetName( Name() + "_neg" );
		linear->SetMultiplier( -1 );
		linear->SetFreeTerm( 0 );
		linear->Connect( 0, *secondInput->Layer(), secondInput->OutputIndex() );
		convertedInputs[1] = new CUserTensor( secondInput->Shape(), secondInput->Layout(), CLayerOutput( linear, 0 ) );
	}

	CPtr<CBaseLayer> layer( new CEltwiseSumLayer( dnn.GetMathEngine() ) );
	CEltwiseOperatorBase::AddLayersImpl( Broadcast(), convertedInputs, *layer, dnn, outputs );
}

// --------------------------------------------------------------------------------------------------------------------

void CMulOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CPtr<CBaseLayer> layer( new CEltwiseMulLayer( dnn.GetMathEngine() ) );
	CEltwiseOperatorBase::AddLayersImpl( Broadcast(), inputs, *layer, dnn, outputs );
}

// --------------------------------------------------------------------------------------------------------------------

void CDivOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CTensorArray convertedInputs;
	inputs.CopyTo( convertedInputs );

	// a / b = a * (1 / b)
	// In that case it's impossible to imitate (1 / x) operation via layer
	// That's why only CDataTensor is supported
	CheckOnnxProtocol( inputs[1] != nullptr, "input isn't connected", *this );
	CheckNeoOnnxSupport( inputs[1]->IsCalculated(), "Div supports only data tensor as a second input", *this );
	CPtr<const CDataTensor> secondInput = dynamic_cast<const CDataTensor*>( inputs[1].Ptr() );
	CPtr<CDnnBlob> newBlob = secondInput->Data()->GetClone();
	newBlob->GetMathEngine().VectorInv( secondInput->Data()->GetData(), newBlob->GetData(), newBlob->GetDataSize() );
	convertedInputs[1] = new CDataTensor( secondInput->Shape(), secondInput->Layout(), *newBlob );

	CPtr<CBaseLayer> layer( new CEltwiseMulLayer( dnn.GetMathEngine() ) );
	CEltwiseOperatorBase::AddLayersImpl( Broadcast(), convertedInputs, *layer, dnn, outputs );
}

// --------------------------------------------------------------------------------------------------------------------

void CSumOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CBroadcast broadcast( BT_Numpy, NotFound );
	if( OpsetVersion < 8 ) {
		broadcast.Type = BT_None;
	}

	CPtr<CBaseLayer> layer( new CEltwiseSumLayer( dnn.GetMathEngine() ) );
	CEltwiseOperatorBase::AddLayersImpl( broadcast, inputs, *layer, dnn, outputs );
}

} // namespace NeoOnnx

