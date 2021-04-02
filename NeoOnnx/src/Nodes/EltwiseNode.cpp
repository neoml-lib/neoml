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

#include "EltwiseNode.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

// Checks if tensor shapes are equal
bool isEqual( const CTensorShape& first, const CTensorShape& second )
{
	if( first.Size() != second.Size() ) {
		return false;
	}

	for( int i = 0; i < first.Size(); ++i ) {
		if( first[i] != second[i] ) {
			return false;
		}
	}

	return true;
}

// Generate unique layer name for dnn
static CString getUniqueLayerName( const CString& prefix, const CDnn& dnn )
{
	int currIndex = dnn.GetLayerCount();
	CString currName = prefix + Str( currIndex );
	while( dnn.HasLayer( currName ) ) {
		++currIndex;
		currName = prefix + Str( currIndex );
	}
	return currName;
}

// Converts tensor prior to imageResizeLayer
CPtr<const CUserTensor> convertTensorBeforeUpsample( const CUserTensor& input, int heightDimIndex, int widthDimIndex )
{
	const CTensorLayout& inputLayout = input.Layout();

	if( inputLayout[heightDimIndex] == BD_Height
		&& ( widthDimIndex == NotFound || inputLayout[widthDimIndex] == static_cast<int>( BD_Width ) ) )
	{
		return &input;
	}

	CTensorLayout newLayout;
	newLayout.SetBufferSize( input.DimCount() );
	for( int i = 0; i < input.DimCount(); ++i ) {
		if( i == heightDimIndex ) {
			newLayout.Add( BD_Height );
		} else if( i == widthDimIndex ) {
			newLayout.Add( BD_Width );
		} else if( widthDimIndex == NotFound ) {
			newLayout.Add( i < static_cast<int>( BD_Height ) ? static_cast<TBlobDim>( i )
				: static_cast<TBlobDim>( i + 1 ) );
		} else {
			newLayout.Add( i < static_cast<int>( BD_Height ) ? static_cast<TBlobDim>( i )
				: static_cast<TBlobDim>( i + 2 ) );
		}
	}

	return dynamic_cast<const CUserTensor*>( ConvertTensor( input, newLayout ).Ptr() );
}

CPtr<const CUserTensor> addUpsample2dLayer( CUpsampling2DLayer& upsample, CDnn& dnn, const CUserTensor& input,
	int heightDimIndex, int widthDimIndex )
{
	// Add imageResize layer
	CPtr<const CUserTensor> result = convertTensorBeforeUpsample( input, heightDimIndex, widthDimIndex );
	upsample.Connect( 0, *result->Layer(), result->OutputIndex() );
	dnn.AddLayer( upsample );

	// Calculate output shape
	CTensorShape outputShape;
	result->Shape().CopyTo( outputShape );
	// TODO: delete after debug
	CheckNeoOnnxInternal( outputShape[heightDimIndex] == 1, "Bug in convertTensorBeforeUpsample" );
	outputShape[heightDimIndex] = upsample.GetHeightCopyCount();
	if( widthDimIndex != NotFound ) {
		CheckNeoOnnxInternal( outputShape[widthDimIndex] == 1, "Bug in convertTensorBeforeUpsample" );
		outputShape[widthDimIndex] = upsample.GetWidthCopyCount();
	}

	// Construct new CUserTensor which is provided by imageResize layer
	return new CUserTensor( outputShape, result->Layout(), CLayerOutput( &upsample, 0 ) );
}

//---------------------------------------------------------------------------------------------------------------------

CEltwiseNodeBase::CEltwiseNodeBase( const onnx::NodeProto& eltwise, int opsetVersion,
	TOperation _operation, int _argsNum ) :
	COpNode( eltwise, opsetVersion ),
	operation( _operation ),
	argsNum( _argsNum )
{
	if( argsNum > 0 ) {
		CheckOnnxProtocol( InputCount() == argsNum, "expected " + Str( argsNum ) + " arguments", OnnxNode );
	}
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", OnnxNode );
}

void CEltwiseNodeBase::AddLayers( const CObjectArray<const CTensorBase>& inputs,
	CObjectArray<const CTensorBase>& outputs, CDnn& dnn )
{
	// Corner case which doesn't contradict Onnx protocol: nodes with variable input count can have 1 input
	if( inputs.Size() == 1 && argsNum < 0 ) {
		outputs[0] = inputs[0];
		return;
	}

	// Calculate outputShape
	CTensorShape outputShape;
	inputs[0]->Shape().CopyTo( outputShape );
	for( int i = 1; i < inputs.Size(); ++i ) {
		CTensorShape buff;
		CheckNeoOnnxSupport( broadcastShape( outputShape, inputs[i]->Shape(), BroadcastInfo(), buff ),
			"Can't broadcast tensors shape", OnnxNode );
		buff.CopyTo( outputShape );
	}

	CObjectArray<const CTensorBase> currInputs;
	inputs.CopyTo( currInputs );
	if( argsNum == 2 ) {
		// Preparing values in case of subtraction or division
		currInputs[1] = prepareSecondInput( currInputs );
	}
	// Broadcast input to the final shape and set proper layout
	for( int i = 0; i < inputs.Size(); ++i ) {
		currInputs[i] = broadcast( *currInputs[i], BroadcastInfo(), outputShape );
		currInputs[i] = ConvertTensor( *currInputs[i], currInputs[0]->Layout() );
	}

	// Put pre-calculated blobs to the source in the net
	for( int i = 0; i < currInputs.Size(); ++i ) {
		if( currInputs[i]->IsCalculated() ) {
			CPtr<CSourceLayer> source = new CSourceLayer( dnn.GetMathEngine() );
			source->SetName( Name() + "_input_" + Str( i ) );
			source->SetBlob( dynamic_cast<const CDataTensor*>( currInputs[i].Ptr() )->Data()->GetCopy() );
			dnn.AddLayer( *source );
			currInputs[i] = new CUserTensor( currInputs[i]->Shape(), currInputs[i]->Layout(), CLayerOutput( source, 0 ) );
		}
	}

	static_assert( O_Count == 4, "O_Count != 4" );
	CPtr<CBaseLayer> eltwise = nullptr;
	if( operation == O_Add || operation == O_Sub ) {
		eltwise = new CEltwiseSumLayer( dnn.GetMathEngine() );
	} else {
		eltwise = new CEltwiseMulLayer( dnn.GetMathEngine() );
	}
	eltwise->SetName( Name() );
	for( int i = 0; i < currInputs.Size(); ++i ) {
		// TODO: Debug-only, delete later
		CheckNeoOnnxInternal( !currInputs[i]->IsCalculated(), "By this moment all tensors must be user-provided", OnnxNode );
		const CUserTensor* userInput = dynamic_cast<const CUserTensor*>( currInputs[i].Ptr() );
		eltwise->Connect( i, *userInput->Layer(), userInput->OutputIndex() );
	}
	dnn.AddLayer( *eltwise );
	outputs[0] = new CUserTensor( outputShape, currInputs[0]->Layout(), CLayerOutput( eltwise, 0 ) );
}

// Calculates shape of the result of the broadcast operation
// If shapes can be broadcasted writes broadcasted shape to the result and returns true
// Returns false if shapes can't be broadcasted (in this case result will be empty)
bool CEltwiseNodeBase::broadcastShape( const CTensorShape& first, const CTensorShape& second,
	const CBroadcastInfo& broadcast, CTensorShape& result ) const
{
	if( broadcast.Type == BT_None ) {
		// No broadcast, the shape must match
		if( isEqual( first, second ) ) {
			first.CopyTo( result );
			return true;
		}
		return false;
	}

	int axis = NotFound;
	if( broadcast.Type == BT_Onnx ) {
		axis = broadcast.Axis;
		CheckNeoOnnxSupport( second.Size() <= first.Size(), "second tensor has more dimensions", OnnxNode );
		if( axis < 0 ) {
			axis = abs( first.Size() - second.Size() );
		}
	} else {
		// Numpy-style broadcast is similar to the Onnx-broadcast with axis equal to difference
		// in number of dimensions
		axis = abs( first.Size() - second.Size() );
	}

	// The shape with lesser number of dimensions must be padded with ones
	const CTensorShape& lesserShape = first.Size() <= second.Size() ? first : second;
	const CTensorShape& biggerShape = first.Size() > second.Size() ? first  : second;
	CTensorShape paddedShape;
	paddedShape.Add( 1, axis );
	paddedShape.Add( lesserShape );
	if( paddedShape.Size() > biggerShape.Size() ) {
		// Wrong braodcast parameters (axis value is too big)
		return false;
	}
	// TODO: Debug-only, delete later
	CheckNeoOnnxInternal( broadcast.Type == BT_Onnx || paddedShape.Size() == biggerShape.Size(),
		"something wrong with Numpy broadcast", OnnxNode );

	// This will add ones only in case of BT_Onnx and axis != abs( first.Size() - second.Size() )
	paddedShape.Add( 1, biggerShape.Size() - paddedShape.Size() );

	result.SetSize( paddedShape.Size() );
	for( int dim = 0; dim < result.Size(); ++dim ) {
		if( paddedShape[dim] == biggerShape[dim] || min( paddedShape[dim], biggerShape[dim] ) == 1 ) {
			result[dim] = max( paddedShape[dim], biggerShape[dim] );
		} else {
			result.DeleteAll();
			return false;
		}
	}

	return true;
}

// This method modifies second input for binary division or subtraction
CPtr<const CTensorBase> CEltwiseNodeBase::prepareSecondInput( const CObjectArray<const CTensorBase>& inputs ) const
{
	static_assert( O_Count == 4, "O_Count != 4" );
	CheckNeoOnnxInternal( inputs.Size() == 2, "Illegal CEltwiseNodeBase::prepareSecondInput call", OnnxNode );
	if( operation == O_Mul || operation == O_Add ) {
		// Add and Mul already have corresponding layers 
		return inputs[1];
	}

	if( operation == O_Sub ) {
		// a - b = a + (-1 * b)
		if( inputs[1]->IsCalculated() ) {
			// Imitating by multiplying values in blob
			CPtr<const CDataTensor> secondInput = dynamic_cast<const CDataTensor*>( inputs[1].Ptr() );
			CPtr<CDnnBlob> newBlob = secondInput->Data()->GetClone();
			CFloatHandleStackVar minusOne( newBlob->GetMathEngine() );
			minusOne.SetValue( -1 );
			newBlob->GetMathEngine().VectorMultiply( secondInput->Data()->GetData(), newBlob->GetData(),
				newBlob->GetDataSize(), minusOne );
			return new CDataTensor( secondInput->Shape(), secondInput->Layout(), *newBlob );
		} else {
			// Imitating by CLinearLayer with multiplier equal to -1
			CPtr<const CUserTensor> secondInput = dynamic_cast<const CUserTensor*>( inputs[1].Ptr() );
			CDnn& dnn = *secondInput->Layer()->GetDnn();
			CPtr<CLinearLayer> linear = new CLinearLayer( dnn.GetMathEngine() );
			linear->SetName( Name() + "_neg" );
			linear->SetMultiplier( -1 );
			linear->Connect( 0, *secondInput->Layer(), secondInput->OutputIndex() );
			return new CUserTensor( secondInput->Shape(), secondInput->Layout(), CLayerOutput( linear, 0 ) );
		}
	}

	// operation is O_Div
	// a / b = a * (1 / b)
	// in that case we can't imitate (1 / x) operation by layer that's why only CDataTensor is supported
	CheckNeoOnnxSupport( inputs[1]->IsCalculated(), "Div supports only data tensor as second input", OnnxNode );
	CPtr<const CDataTensor> secondInput = dynamic_cast<const CDataTensor*>( inputs[1].Ptr() );
	CPtr<CDnnBlob> newBlob = secondInput->Data()->GetClone();
	newBlob->GetMathEngine().VectorInv( secondInput->Data()->GetData(), newBlob->GetData(), newBlob->GetDataSize() );
	return new CDataTensor( secondInput->Shape(), secondInput->Layout(), *newBlob );
}

// Broadcasts tensor into outputShape via broadcastInfo
CPtr<const CTensorBase> CEltwiseNodeBase::broadcast( const CTensorBase& input, const CBroadcastInfo& broadcastInfo,
	const CTensorShape& outputShape ) const
{
	if( input.IsCalculated() ) {
		return broadcast( dynamic_cast<const CDataTensor&>( input ), broadcastInfo, outputShape ).Ptr();
	}
	return broadcast( dynamic_cast<const CUserTensor&>( input ), broadcastInfo, outputShape ).Ptr();
}

// Broadcasts data tensor into outputShape via broadcastInfo
CPtr<const CDataTensor> CEltwiseNodeBase::broadcast( const CDataTensor& input, const CBroadcastInfo& broadcastInfo,
	const CTensorShape& outputShape ) const
{
	if( isEqual( input.Shape(), outputShape ) ) {
		return &input;
	}

	CheckNeoOnnxInternal( broadcastInfo.Type != BT_None, "Cannot broadcast tensor", OnnxNode );

	IMathEngine& mathEngine = input.Data()->GetMathEngine();
	CRandom random( 0x32456 );

	CDnn internalDnn( random, mathEngine );
	CPtr<CSourceLayer> source = new CSourceLayer( mathEngine );
	source->SetBlob( input.Data()->GetCopy() );
	internalDnn.AddLayer( *source );
	
	CPtr<const CUserTensor> internalInput = new CUserTensor( input.Shape(), input.Layout(), CLayerOutput( source, 0 ) );
	CPtr<const CUserTensor> internalOutput = broadcast( *internalInput, broadcastInfo, outputShape );
	// TODO: delete after debug
	CheckNeoOnnxInternal( isEqual( internalOutput->Shape(), outputShape ), "User tensor broadcast bug!", OnnxNode );

	CPtr<CSinkLayer> sink = new CSinkLayer( mathEngine );
	sink->Connect( 0, *internalOutput->Layer(), internalOutput->OutputIndex() );
	internalDnn.AddLayer( *sink );

	internalDnn.RunOnce();

	return new CDataTensor( outputShape, internalOutput->Layout(), *sink->GetBlob() );
}

// Broadcasts user tensor into outputShape via broadcastInfo
CPtr<const CUserTensor> CEltwiseNodeBase::broadcast( const CUserTensor& input, const CBroadcastInfo& broadcastInfo,
	const CTensorShape& outputShape ) const
{
	if( isEqual( input.Shape(), outputShape ) ) {
		return &input;
	}

	CheckNeoOnnxInternal( broadcastInfo.Type != BT_None, "Wrong broadcast type", OnnxNode );
	CheckNeoOnnxInternal( input.DimCount() <= outputShape.Size(), "Broadcast cannot reduce dimension count" );

	// Prefix for upsmaple layer names
	const CString upsampleNamePrefix = input.Layer()->GetName() + CString( "_upsample_" );
	// Used network
	CDnn& dnn = *( input.Layer()->GetDnn() );
	// Used mathEngine
	IMathEngine& mathEngine = dnn.GetMathEngine();

	int axis = outputShape.Size() - input.DimCount();
	if( broadcastInfo.Type == BT_Onnx && broadcastInfo.Axis >= 0 ) {
		axis = broadcastInfo.Axis;
	}

	CPtr<const CUserTensor> currData = padTensorShape( input, outputShape.Size(), axis );
	CPtr<CUpsampling2DLayer> upsample = nullptr;
	int heightDimIndex = NotFound;
	int widthDimIndex = NotFound;
	CTensorShape inputShape;
	currData->Shape().CopyTo( inputShape );

	for( int i = 0; i < inputShape.Size(); ++i ) {
		if( inputShape[i] == outputShape[i] ) {
			continue;
		}
		// TODO: delete after debug
		CheckNeoOnnxInternal( inputShape[i] == 1, "User tensor broadcast bug!", OnnxNode );

		if( upsample == nullptr ) {
			upsample = new CUpsampling2DLayer( mathEngine );
			upsample->SetName( getUniqueLayerName( upsampleNamePrefix, dnn ) );
		}

		if( heightDimIndex == NotFound ) {
			heightDimIndex = i;
			upsample->SetHeightCopyCount( outputShape[i] );
		} else {
			widthDimIndex = i;
			upsample->SetWidthCopyCount( outputShape[i] );
			currData = addUpsample2dLayer( *upsample, dnn, *currData, heightDimIndex, widthDimIndex );
			upsample = nullptr;
			heightDimIndex = NotFound;
			widthDimIndex = NotFound;
		}
	}

	// Corner case: we need to broadcast odd number of dimensions
	// In that case by this moment upsample != nullptr
	// heightDimIndex will be defined but widthDimIndex will remain NotFound
	if( upsample != nullptr ) {
		upsample->SetWidthCopyCount( 1 ); // Default value is 0 which is invalid
		currData = addUpsample2dLayer( *upsample, dnn, *currData, heightDimIndex, widthDimIndex );
	}

	return currData;
}

// Pads shape with '1' without changing the data
CPtr<const CUserTensor> CEltwiseNodeBase::padTensorShape( const CUserTensor& input, int dimCount, int axis ) const
{
	const CTensorShape& inputShape = input.Shape();
	CheckNeoOnnxInternal( axis + inputShape.Size() <= dimCount, "Wrong axis in broadcast", OnnxNode );

	CTensorShape outputShape;
	outputShape.Add( 1, axis );
	outputShape.Add( inputShape );
	outputShape.Add( 1, dimCount - outputShape.Size() );

	const CTensorLayout& inputLayout = input.Layout();

	TBlobDim currDim = BD_BatchLength;
	CTensorLayout outputLayout;
	outputLayout.SetBufferSize( dimCount );
	// Adding unused blob dims to the new layout
	for( int i = 0; i < axis; ++i ) {
		while( inputLayout.Find( currDim ) != NotFound && currDim < BD_Count ) {
			++currDim;
		}
		CheckNeoOnnxInternal( currDim != BD_Count, "Wrong axis index", OnnxNode );
		outputLayout.Add( currDim );
		++currDim;
	}
	// Copying existing dims
	outputLayout.Add( inputLayout );
	// Adding unused blob dims to the new layout
	for( int i = outputLayout.Size(); i < dimCount; ++i ) {
		while( inputLayout.Find( currDim ) != NotFound && currDim < BD_Count ) {
			++currDim;
		}
		CheckNeoOnnxInternal( currDim != BD_Count, "Wrong axis index", OnnxNode );
		outputLayout.Add( currDim );
		++currDim;
	}

	return new CUserTensor( outputShape, outputLayout, input.LayerOutput() );
}

// --------------------------------------------------------------------------------------------------------------------

CEltwiseNodeBase::CBroadcastInfo CEltwiseBinaryOpNodeBase::BroadcastInfo() const
{
	if( OpsetVersion < 7 ) {
		if( Attributes.GetOptionalInt( "broadcast", 0 ) != 0 ) {
			return CBroadcastInfo( BT_Onnx,
				Attributes.GetOptionalInt( "axis", NotFound ) );
		} else {
			return CBroadcastInfo( BT_None );
		}
	}

	return CBroadcastInfo( BT_Numpy );
}

CEltwiseNodeBase::CBroadcastInfo CSumNode::BroadcastInfo() const
{
	if( OpsetVersion < 8 ) {
		return CBroadcastInfo( BT_None );
	}

	return CBroadcastInfo( BT_Numpy );
}

} // namespace NeoOnnx
