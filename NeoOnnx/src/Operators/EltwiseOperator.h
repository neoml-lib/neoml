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

#pragma once

#include "../LayerOperator.h"

#include "TensorUtils.h"

#include <NeoML/Dnn/Layers/Onnx/OnnxEltwiseLayer.h>

namespace NeoOnnx {

// Base class for operators which perform eltwise operations
template<COnnxEltwiseLayer::TOperation Operation>
class CEltwiseOperator : public CLayerOperator {
public:
	CEltwiseOperator( const onnx::NodeProto& eltwise, int opsetVersion );

protected:
	// AddLayers implementation for the given broadcast and layer
	// The derivatives should call this method from their AddLayers
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;

private:
	int getArgsNum() const;
	CBroadcast getBroadcast() const;
	void getOutputShape( const CTensorArray& inputs, CTensorShape& outputShape ) const;
};

//---------------------------------------------------------------------------------------------------------------------

template<COnnxEltwiseLayer::TOperation Operation>
CEltwiseOperator<Operation>::CEltwiseOperator( const onnx::NodeProto& eltwise, int opsetVersion ) :
	CLayerOperator( eltwise, opsetVersion )
{
	const int argsNum = getArgsNum();
	if( argsNum > 0 ) {
		CheckOnnxProtocol( InputCount() == argsNum, "expected " + Str( argsNum ) + " arguments", *this );
	}
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

template<COnnxEltwiseLayer::TOperation Operation>
inline void CEltwiseOperator<Operation>::AddLayers( const CTensorArray& inputs, CDnn& dnn,
	CTensorArray& outputs ) const
{
	CheckNoNullInputs( inputs );
	const bool hasUserInput = HasUserInput( inputs );

	// Corner case which doesn't violate Onnx protocol: operators with variable input count may have 1 input
	if( inputs.Size() == 1 && getArgsNum() < 0 ) {
		outputs.Add( inputs[0] );
		return;
	}

	// Calculate output layout
	CTensorLayout outputLayout = inputs[0]->Layout();
	for( int i = 1; i < inputs.Size(); ++i ) {
		if( inputs[i]->DimCount() > outputLayout.Size() ) {
			outputLayout = inputs[i]->Layout();
		}
	}

	CPtr<COnnxEltwiseLayer> layer = new COnnxEltwiseLayer( dnn.GetMathEngine() );
	layer->SetName( Name() );
	layer->SetOperation( Operation );
	dnn.AddLayer( *layer );

	const CBroadcast broadcast = getBroadcast();
	for( int i = 0; i < inputs.Size(); ++i ) {
		CPtr<const CTensorBase> tensor = PrepareForBroadcast( *inputs[i], broadcast, outputLayout.Size() );
		tensor = ConvertTensor( *tensor, outputLayout );
		if( HasUserInput( inputs ) ) {
			CPtr<const CUserTensor> userTensor = AsUserTensor( *tensor, Name() + "_input_" + Str( i ), dnn );
			layer->Connect( i, *userTensor->Layer(), userTensor->OutputIndex() );
		} else {
			CPtr<const CShapeTensor> shapeTensor = AsShapeTensor( *tensor, Name() + "_input_" + Str( i ), dnn );
			layer->Connect( i, *shapeTensor->Layer(), shapeTensor->OutputIndex() );
		}
	}

	if( hasUserInput ) {
		outputs.Add( new CUserTensor( outputLayout, CLayerOutput( layer, 0 ) ) );
	} else {
		CTensorShape outputShape;
		getOutputShape( inputs, outputShape );
		outputs.Add( new CShapeTensor( outputLayout, outputShape, CLayerOutput( layer, 0 ) ) );
	}
}

// Expected number of arguments (-1 if any number is supported)
template<COnnxEltwiseLayer::TOperation Operation>
inline int CEltwiseOperator<Operation>::getArgsNum() const
{
	if( Type() == "Sum" ) {
		return -1;
	} else if( Type() == "Where" ) {
		return 3;
	}

	return 2;
}

// Broadcast rule according to operator type and opset version
template<COnnxEltwiseLayer::TOperation Operation>
inline CBroadcast CEltwiseOperator<Operation>::getBroadcast() const
{
	CBroadcast broadcast( BT_Numpy, NotFound );
	
	if( Type() == "Where" ) {
		return broadcast;
	} else if( Type() == "Sum" ) {
		if( OpsetVersion < 8 ) {
			broadcast.Type = BT_None;
		}
		return broadcast;
	}

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

// Calculates output shape based on shape of inputs and broadcast rule
template<COnnxEltwiseLayer::TOperation Operation>
inline void CEltwiseOperator<Operation>::getOutputShape( const CTensorArray& inputs, CTensorShape& outputShape ) const
{
	NeoPresume( !HasUserInput( inputs ) );
	GetTensorShape( *inputs[0], outputShape );

	const CBroadcast broadcast = getBroadcast();
	for( int i = 1; i < inputs.Size(); ++i ) {
		CTensorShape inputShape;
		GetTensorShape( *inputs[i], inputShape );
		CTensorShape buff;
		CheckNeoOnnxSupport( BroadcastTensorShape( outputShape, inputShape, broadcast, buff ),
			"Can't broadcast tensors shape", *this );
		buff.CopyTo( outputShape );
	}
}

} // namespace NeoOnnx

