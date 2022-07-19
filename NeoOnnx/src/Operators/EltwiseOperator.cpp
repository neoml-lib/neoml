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
		outputs.Add( inputs[0] );
		return;
	}

	// Calculate outputShape
	CTensorShape outputShape;
	inputs[0]->Shape().CopyTo( outputShape );
	CTensorLayout outputLayout = inputs[0]->Layout();

	for( int i = 1; i < inputs.Size(); ++i ) {
		CheckOnnxProtocol( inputs[i] != nullptr, "input can't be optional", *this );
		CTensorShape buff;
		CheckNeoOnnxSupport( BroadcastTensorShape( outputShape, inputs[i]->Shape(), broadcast, buff ),
			"Can't broadcast tensors shape", *this );
		buff.CopyTo( outputShape );
		if( inputs[i]->DimCount() > outputLayout.Size() ) {
			outputLayout = inputs[i]->Layout();
		}
	}

	CPtr<CBroadcastLayer> broadcastLayer = new CBroadcastLayer( dnn.GetMathEngine() );
	broadcastLayer->SetName( Name() + "_Broadcast" );
	dnn.AddLayer( *broadcastLayer );
	eltwiseLayer.SetName( Name() );

	for( int i = 0; i < inputs.Size(); ++i ) {
		CPtr<const CTensorBase> tensor = PrepareForBroadcast( *inputs[i], broadcast, outputShape.Size() );
		tensor = ConvertTensor( *tensor, outputLayout );
		CPtr<const CUserTensor> userTensor = AsUserTensor( *tensor, Name() + "_input_" + Str( i ), dnn );
		broadcastLayer->Connect( i, *userTensor->Layer(), userTensor->OutputIndex() );
		eltwiseLayer.Connect( i, *broadcastLayer, i );
	}

	dnn.AddLayer( eltwiseLayer );
	outputs.Add( new CUserTensor( outputShape, outputLayout, CLayerOutput( &eltwiseLayer, 0 ) ) );
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
	CPtr<CBaseLayer> layer( new CEltwiseSubLayer( dnn.GetMathEngine() ) );
	CEltwiseOperatorBase::AddLayersImpl( Broadcast(), inputs, *layer, dnn, outputs );
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
	CPtr<CBaseLayer> layer( new CEltwiseDivLayer( dnn.GetMathEngine() ) );
	CEltwiseOperatorBase::AddLayersImpl( Broadcast(), inputs, *layer, dnn, outputs );
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

