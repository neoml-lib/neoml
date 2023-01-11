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

#include "SliceOperator.h"
#include "LayerUtils.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

#include <NeoML/Dnn/Layers/Onnx/OnnxSliceLayer.h>

namespace NeoOnnx {

CSliceOperator::CSliceOperator( const onnx::NodeProto& slice, int opsetVersion ) :
	CLayerOperator( slice, opsetVersion )
{
	// v1 - original
	// v10 - attributes are replaced with additional inputs + 'step' support
	// v11 - backward slicing support is added
	// v13 - bfloat16 support is added
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	if( OpsetVersion < 10 ) {
		CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	} else {
		CheckOnnxProtocol( InputCount() >= 3 && InputCount() <= 5, "operator must have from 3 up to 5 inputs", *this );
	}
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CSliceOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNoNullInputs( inputs );

	for( int i = 1; i < inputs.Size(); ++i ) {
		CheckNeoOnnxSupport( inputs[i]->Type() != TTensorType::User, "user-provided slice param", *this );
		CheckNeoOnnxSupport( inputs[i]->DimCount() == 1, "non-1-dimensional slice param", *this );
	}

	const bool isInShapeMode = hasShapeOutput( inputs );

	CPtr<COnnxSliceLayer> sliceLayer = new COnnxSliceLayer( dnn.GetMathEngine() );
	sliceLayer->SetName( Name() );
	inputs[0]->Layout().CopyTo( sliceLayer->TensorLayout() );

	if( isInShapeMode ) {
		CPtr<const CShapeTensor> data = AsShapeTensor( *inputs[0], Name() + "_Data", dnn );
		sliceLayer->Connect( 0, *data->Layer(), data->OutputIndex() );
	} else {
		CPtr<const CUserTensor> data = AsUserTensor( *inputs[0], Name() + "_Data", dnn );
		sliceLayer->Connect( 0, *data->Layer(), data->OutputIndex() );
	}

	CPtr<const CShapeTensor> starts = getStarts( inputs, dnn );
	sliceLayer->Connect( 1, *starts->Layer(), starts->OutputIndex() );
	CPtr<const CShapeTensor> ends = getEnds( inputs, dnn );
	sliceLayer->Connect( 2, *ends->Layer(), ends->OutputIndex() );
	CPtr<const CShapeTensor> axes = getAxes( inputs, dnn );
	if( axes != nullptr ) {
		sliceLayer->Connect( 3, *axes->Layer(), axes->OutputIndex() );
	}
	CPtr<const CShapeTensor> steps = getSteps( inputs, dnn );
	if( steps != nullptr ) {
		sliceLayer->Connect( 4, *steps->Layer(), steps->OutputIndex() );
	}

	dnn.AddLayer( *sliceLayer );

	if( isInShapeMode ) {
		CTensorShape outputShape;
		calcOutputShape( inputs, outputShape );
		outputs.Add( new CShapeTensor( inputs[0]->Layout(), outputShape, CLayerOutput( sliceLayer, 0 ) ) );
	} else {
		outputs.Add( new CUserTensor( inputs[0]->Layout(), CLayerOutput( sliceLayer, 0 ) ) );
	}
}

// Checks whether output can be CShapeTensor or it has to be CUserTensor
bool CSliceOperator::hasShapeOutput( const CTensorArray& inputs ) const
{
	// If input data is a CUserTensor then output data has to be CUserTensor
	if( inputs[0]->Type() == TTensorType::User ) {
		return false;
	}

	// For the correct work of NeoOnnx each CShapeTensor must have a computable shape (the data may stay unknown)
	// Which is why for the CShapeTensor as output we need values from all other inputs (CDataTensor)
	for( int i = 1; i < inputs.Size(); ++i ) {
		if( inputs[i]->Type() != TTensorType::Data ) {
			return false;
		}
	}

	return true;
}

void CSliceOperator::calcOutputShape( const CTensorArray& inputs, CTensorShape& outputShape ) const
{
	NeoPresume( hasShapeOutput( inputs ) );

	CTensorShape inputShape;
	if( inputs[0]->Type() == TTensorType::Data ) {
		const CDataTensor* dataTensor = dynamic_cast<const CDataTensor*>( inputs[0].Ptr() );
		for( int dimIndex = 0; dimIndex < dataTensor->DimCount(); ++dimIndex ) {
			inputShape.Add( dataTensor->DimSize( dimIndex ) );
		}
	} else {
		NeoPresume( inputs[0]->Type() == TTensorType::Shape );
		const CShapeTensor* shapeTensor = dynamic_cast<const CShapeTensor*>( inputs[0].Ptr() );
		shapeTensor->Shape().CopyTo( inputShape );
	}

	CFastArray<int, 8> starts;
	getStarts( inputs, starts );
	CFastArray<int, 8> ends;
	getEnds( inputs, ends );
	CFastArray<int, 8> axes;
	getAxes( inputs, axes );
	CFastArray<int, 8> steps;
	getSteps( inputs, steps );

	inputShape.CopyTo( outputShape );
	NeoPresume( starts.Size() == ends.Size() );
	NeoPresume( starts.Size() <= axes.Size() );
	NeoPresume( starts.Size() <= steps.Size() );
	for( int i = 0; i < starts.Size(); ++i ) {
		const int dimSize = inputShape[axes[i]];
		const int start = std::min<int>( dimSize, starts[i] < 0 ? starts[i] + dimSize : starts[i] );
		const int end = std::min<int>( dimSize, ends[i] < 0 ? ends[i] + dimSize : ends[i] );
		CheckNeoOnnxSupport( steps[i] == 1, "Non-1 step", *this );
		NeoPresume( start <= end );
		outputShape[axes[i]] = end - start;
	}
}

// Fills array with slice start indices
void CSliceOperator::getStarts( const CTensorArray& inputs, CFastArray<int, 8>& starts ) const
{
	if( OpsetVersion < 10 ) {
		// Extracting from attributes
		CheckOnnxProtocol( GetAttribute( "starts", starts ), "'starts' attribute is missing", *this );
	} else {
		NeoPresume( inputs[1]->Type() == TTensorType::Data );
		const CDnnBlob* startsBlob = dynamic_cast<const CDataTensor*>( inputs[1].Ptr() )->Data();
		CheckOnnxProtocol( startsBlob->GetDataType() == CT_Int, "Non-integer starts", *this );
		starts.SetSize( startsBlob->GetDataSize() );
		startsBlob->CopyTo( starts.GetPtr() );
	}
}

// Fills array with slice end indices
void CSliceOperator::getEnds( const CTensorArray& inputs, CFastArray<int, 8>& ends ) const
{
	if( OpsetVersion < 10 ) {
		// Extracting from attributes
		CheckOnnxProtocol( GetAttribute( "ends", ends ), "'ends' attribute is missing", *this );
	} else {
		NeoPresume( inputs[2]->Type() == TTensorType::Data );
		const CDnnBlob* endsBlob = dynamic_cast<const CDataTensor*>( inputs[2].Ptr() )->Data();
		CheckOnnxProtocol( endsBlob->GetDataType() == CT_Int, "Non-integer ends", *this );
		ends.SetSize( endsBlob->GetDataSize() );
		endsBlob->CopyTo( ends.GetPtr() );
	}
}

// Fills array with axes, affected by slice
void CSliceOperator::getAxes( const CTensorArray& inputs, CFastArray<int, 8>& axes ) const
{
	if( OpsetVersion < 10 && GetAttribute( "axes", axes ) ) {
		// Successfully extracted from the attribute
		return;
	} else if( OpsetVersion >= 10 && inputs.Size() >= 4 && inputs[3] != nullptr ) {
		NeoPresume( inputs[3]->Type() == TTensorType::Data );
		const CDnnBlob* axesBlob = dynamic_cast<const CDataTensor*>( inputs[3].Ptr() )->Data();
		CheckOnnxProtocol( axesBlob->GetDataType() == CT_Int, "Non-integer axes", *this );
		axes.SetSize( axesBlob->GetDataSize() );
		axesBlob->CopyTo( axes.GetPtr() );
	} else {
		// Fill with default value
		axes.SetBufferSize( inputs[0]->DimCount() );
		for( int i = 0; i < inputs[0]->DimCount(); ++i ) {
			axes.Add( i );
		}
	}
}

// Fills array with slice steps
void CSliceOperator::getSteps( const CTensorArray& inputs, CFastArray<int, 8>& steps ) const
{
	if( OpsetVersion >= 10 && inputs.Size() >= 5 && inputs[4] != nullptr ) {
		// Extracting from the input
		NeoPresume( inputs[4]->Type() == TTensorType::Data );
		CheckNeoOnnxSupport( inputs[4]->Type() == TTensorType::Data, "User-provided steps", *this );
		const CDnnBlob* stepsBlob = dynamic_cast<const CDataTensor*>( inputs[4].Ptr() )->Data();
		CheckOnnxProtocol( stepsBlob->GetDataType() == CT_Int, "Non-integer steps", *this );
		steps.SetSize( stepsBlob->GetDataSize() );
		stepsBlob->CopyTo( steps.GetPtr() );
	} else {
		// Fill with default value
		steps.SetBufferSize( inputs[0]->DimCount() );
		steps.Add( 1, inputs[0]->DimCount() );
	}
}

CPtr<const CShapeTensor> CSliceOperator::getStarts( const CTensorArray& inputs, CDnn& dnn ) const
{
	if( OpsetVersion < 10 ) {
		CFastArray<int, 8> startsAttr;
		CheckOnnxProtocol( GetAttribute( "starts", startsAttr ), "'starts' attribute is missing", *this );
		return AsShapeTensor( startsAttr, Name() + "_Starts", dnn );
	}

	return AsShapeTensor( *inputs[1], Name() + "_Starts", dnn );
}

CPtr<const CShapeTensor> CSliceOperator::getEnds( const CTensorArray& inputs, CDnn& dnn ) const
{
	if( OpsetVersion < 10 ) {
		CFastArray<int, 8> endsAttr;
		CheckOnnxProtocol( GetAttribute( "ends", endsAttr ), "'ends' attribute is missing", *this );
		return AsShapeTensor( endsAttr, Name() + "_Ends", dnn );
	}

	return AsShapeTensor( *inputs[2], Name() + "_Ends", dnn );
}

CPtr<const CShapeTensor> CSliceOperator::getAxes( const CTensorArray& inputs, CDnn& dnn ) const
{
	if( OpsetVersion < 10 ) {
		CFastArray<int, 8> axesAttr;
		if( GetAttribute( "axes", axesAttr ) ) {
			return AsShapeTensor( axesAttr, Name() + "_Axes", dnn );
		}
		return nullptr;
	}

	if( inputs.Size() <= 3 ) {
		return nullptr;
	}

	return AsShapeTensor( *inputs[3], Name() + "_Axes", dnn );
}

CPtr<const CShapeTensor> CSliceOperator::getSteps( const CTensorArray& inputs, CDnn& dnn ) const
{
	if( OpsetVersion < 10 || inputs.Size() <= 4 ) {
		return nullptr;
	}

	return AsShapeTensor( *inputs[4], Name() + "_Steps", dnn );
}

} // namespace NeoOnnx
