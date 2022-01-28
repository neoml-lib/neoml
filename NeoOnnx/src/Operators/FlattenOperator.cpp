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

CFlattenOperator::CFlattenOperator( const onnx::NodeProto& flatten, int opsetVersion ) :
	CLayerOperator( flatten, opsetVersion ),
	axis( 1 )
{
	// v1 - original
	// v9 - added different data types support
	// v11 - added negative axis index support
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );

	GetAttribute( "axis", axis );
	if( opsetVersion < 11 ) {
		CheckOnnxProtocol( axis >= 0, "negative axis index", *this );
	}
}

void CFlattenOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckOnnxProtocol( inputs[0] != nullptr, "input can't be optional", *this );

	// Every operator which somehow changes Onnx tensor's shape or dimensions works only with Onnx dim type
	// Otherwise it'll lead to hardly fixable troubles with data-ordering
	CPtr<const CUserTensor> input;
	if( IsTransposedLayout( inputs[0]->Layout() ) ) {
		input = AsUserTensor( *ConvertTensor( *inputs[0], CTensorLayout( inputs[0]->DimCount() ) ), Name() + "_Source", dnn );
	} else {
		input = AsUserTensor( *inputs[0], Name() + "_Source", dnn );
	}

	// Flatten operator reshapes tensor into 2-dimensional matrix of size
	// [ dim_0 * ... * dim_(axis-1) ; dim_axis * ... * dim_(n-1) ]
	// Corner case: if axis == 0 then output shape is [ 1 ; tensorSize ]
	const int axisIndex = axis < 0 ? axis + input->DimCount() : axis;
	CTensorShape outputShape( { 1, 1 } );
	for( int dimIndex = 0; dimIndex < input->DimCount(); ++dimIndex ) {
		outputShape[dimIndex < axisIndex ? 0 : 1] *= input->Shape()[dimIndex];
	}

	CTensorLayout outputLayout( 2 );
	CPtr<CTransformLayer> transform = new CTransformLayer( dnn.GetMathEngine() );
	transform->SetName( Name() );
	transform->SetDimensionRule( outputLayout[0], 
		CTransformLayer::CDimensionRule( CTransformLayer::O_SetSize, outputShape[0] ) );
	transform->SetDimensionRule( outputLayout[1], 
		CTransformLayer::CDimensionRule( CTransformLayer::O_SetSize, outputShape[1] ) );

	for( TBlobDim dim = BD_BatchLength; dim < BD_Count; ++dim ) {
		// Other dimensions must be 1
		if( outputLayout.Find( dim ) == NotFound ) {
			transform->SetDimensionRule( static_cast< TBlobDim >( dim ),
				CTransformLayer::CDimensionRule( CTransformLayer::O_SetSize, 1 ) );
		}
	}

	transform->Connect( 0, *input->Layer(), input->OutputIndex() );
	dnn.AddLayer( *transform );

	outputs.Add( new CUserTensor( outputShape, outputLayout, CLayerOutput( transform, 0 ) ) );
}

} // namespace NeoOnnx
