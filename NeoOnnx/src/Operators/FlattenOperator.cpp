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

#include "FlattenOperator.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CTransformLayer* createTransformLayer( const CLayerOutput& link, const CString& name )
{
	CTransformLayer* transform = Transform( 1, 1, 1, 1, 1, 1, 1 )
		( name, CDnnLayerLink( link.Layer, link.OutputIndex ) );
	for( int dim = 0; dim < BD_Count; ++dim ) {
		transform->SetDimensionRule( static_cast<TBlobDim>( dim ),
			CTransformLayer::CDimensionRule( CTransformLayer::O_InputDim, dim ) );
	}
	return transform;
}

//---------------------------------------------------------------------------------------------------------------------

CFlattenOperator::CFlattenOperator( const onnx::NodeProto& flatten, int opsetVersion ) :
	CLayerOperator( flatten, opsetVersion ),
	axisAttr( 1 )
{
	// v1 - original
	// v9 - added different data types support
	// v11 - added negative axis index support
	// v13 - bfloat16 is supported
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );

	GetAttribute( "axis", axisAttr );
	if( opsetVersion < 11 ) {
		CheckOnnxProtocol( axisAttr >= 0, "negative axis index", *this );
	}
}

void CFlattenOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNoNullInputs( inputs );
	CheckNoShapeInputs( inputs );

	// Every operator which somehow changes Onnx tensor's shape or dimensions works only with Onnx dim type
	// Otherwise it'll lead to hardly fixable troubles with data-ordering
	CPtr<const CUserTensor> input;
	if( IsTransposedLayout( inputs[0]->Layout() ) ) {
		input = AsUserTensor( *ConvertTensor( *inputs[0], CTensorLayout( inputs[0]->DimCount() ) ),
			Name() + "_Source", dnn );
	} else {
		input = AsUserTensor( *inputs[0], Name() + "_Source", dnn );
	}

	// Flatten operator reshapes tensor into 2-dimensional matrix of size
	// [ dim_0 * ... * dim_(axis-1) ; dim_axis * ... * dim_(n-1) ]
	// Corner case: if axis == 0 then output shape is [ 1 ; tensorSize ]
	const int axis = axisAttr < 0 ? axisAttr + input->DimCount() : axisAttr;
	NeoPresume( axis >= 0 && axis < inputs[0]->DimCount() );

	CTensorLayout layout = input->Layout();
	CLayerOutput output = input->LayerOutput();

	// Merge all dimension after axis'th into axis'th (if needed)
	if( axis != 0 && axis < input->DimCount() - 1 ) {
		CTransformLayer* secondTransform = createTransformLayer( output, Name() + "_SecondAxis" );
		secondTransform->SetDimensionRule( layout[axis],
			CTransformLayer::CDimensionRule( CTransformLayer::O_Remainder, 1 ) );
		for( int i = axis + 1; i < input->DimCount(); ++i ) {
			secondTransform->SetDimensionRule( layout[i],
				CTransformLayer::CDimensionRule( CTransformLayer::O_SetSize, 1 ) );
		}
		layout.DeleteAt( axis + 1, layout.Size() - axis - 1 );
		output = CLayerOutput( secondTransform, 0 );
	}

	// Merge all dimension before axis'th into 0'th (if needed)
	if( axis > 1 ) {
		CTransformLayer* firstTransform = createTransformLayer( output, Name() + "_FirstAxis" );
		firstTransform->SetDimensionRule( layout[0],
			CTransformLayer::CDimensionRule( CTransformLayer::O_Remainder, 1 ) );
		for( int i = 1; i < axis; ++i ) {
			firstTransform->SetDimensionRule( layout[i],
				CTransformLayer::CDimensionRule( CTransformLayer::O_SetSize, 1 ) );
		}
		output = CLayerOutput( firstTransform, 0 );
		layout.DeleteAt( 1, axis - 1 );
	}

	// Corner-case when axis == 0
	if( axis == 0 ) {
		CTransformLayer* transform = createTransformLayer( output, Name() );
		for( int i = 0; i < layout.Size(); ++i ) {
			CTransformLayer::TOperation operation = i == 1 ? CTransformLayer::O_Remainder : CTransformLayer::O_SetSize;
			transform->SetDimensionRule( layout[i], CTransformLayer::CDimensionRule( operation, 1 ) );
		}
		output = CLayerOutput( transform, 0 );
		layout.SetSize( 2 );
	}

	outputs.Add( new CUserTensor( layout, output ) );
}

} // namespace NeoOnnx
