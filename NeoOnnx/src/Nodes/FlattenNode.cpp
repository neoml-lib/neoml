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

#include "FlattenNode.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CFlattenNode::CFlattenNode( const onnx::NodeProto& flatten, int opsetVersion ) :
	COpNode( flatten, opsetVersion ),
	axis( Attributes.GetOptionalInt( "axis", 1 ) )
{
	// v1 - original
	// v9 - added different data types support
	// v11 - added negative axis index support
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", flatten );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", flatten );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", flatten );

	if( opsetVersion < 11 ) {
		CheckOnnxProtocol( axis >= 0, "negative axis index", flatten );
	}
}

void CFlattenNode::AddLayers( const CObjectArray<const CTensorBase>& inputs,
	CObjectArray<const CTensorBase>& outputs, CDnn& dnn )
{
	CheckNeoOnnxInternal( inputs[0] != nullptr && !inputs[0]->IsCalculated(), "user data expected as input", OnnxNode );

	// Every operator which somehow changes Onnx tensor's shape or dimensions works only with Onnx dim type
	// Otherwise it'll lead to hardly fixable troubles with data-packing (Onnx's channel-first vs NeoML's channel-last)
	CPtr<const CUserTensor> input = dynamic_cast<const CUserTensor*>( ConvertTensor( *inputs[0], CTensorLayout() ).Ptr() );

	// Flatten operator reshapes tensor into 2-dimensional matrix of size
	// [ dim_0 * ... * dim_(axis-1) ; dim_axis * ... * dim_(n-1) ]
	// Corner case: if axis == 0 then output shape is [ 1 ; tensorSize ]
	const int axisIndex = axis < 0 ? axis + input->Shape().Size() : axis;
	CTensorShape outputShape( { 1, 1 } );
	for( int dimIndex = 0; dimIndex < input->Shape().Size(); ++dimIndex ) {
		outputShape[dimIndex < axisIndex ? 0 : 1] *= input->Shape()[dimIndex];
	}

	CPtr<CTransformLayer> transform = new CTransformLayer( dnn.GetMathEngine() );
	transform->SetName( Name() );
	transform->SetDimensionRule( static_cast<TBlobDim>( 0 ), 
		CTransformLayer::CDimensionRule( CTransformLayer::O_SetSize, outputShape[0] ) );
	transform->SetDimensionRule( static_cast<TBlobDim>( 1 ), 
		CTransformLayer::CDimensionRule( CTransformLayer::O_SetSize, outputShape[1] ) );

	for( int dim = 2; dim < BD_Count; ++dim ) {
		// Other dimensions must be 1
		transform->SetDimensionRule( static_cast<TBlobDim>( dim ), 
			CTransformLayer::CDimensionRule( CTransformLayer::O_SetSize, 1 ) );
	}

	transform->Connect( 0, *input->Layer(), input->OutputIndex() );
	dnn.AddLayer( *transform );

	outputs[0] = new CUserTensor( outputShape, CTensorLayout(), CLayerOutput( transform, 0 ) );
}

} // namespace NeoOnnx
