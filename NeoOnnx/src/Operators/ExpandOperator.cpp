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

#include "onnx.pb.h"

#include "ExpandOperator.h"
#include "NeoOnnxCheck.h"
#include <NeoML/Dnn/Layers/Onnx/OnnxExpandLayer.h>

namespace NeoOnnx {

CExpandOperator::CExpandOperator( const onnx::NodeProto& expand, int opsetVersion ) :
	CLayerOperator( expand, opsetVersion )
{
	// v8 - original
	// v13 - new data types are supported
	CheckOnnxProtocol( InputCount() == 2, "operator must have 2 inputs", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CExpandOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNoNullInputs( inputs );
	CheckNeoOnnxSupport( inputs[1]->Type() != TTensorType::User, "user-provided shape", *this );

	CPtr<const CUserTensor> input = AsUserTensor( *inputs[0], Name() + "_InputSource", dnn );

	CPtr<const CShapeTensor> shape = AsShapeTensor( *inputs[1], Name() + "_ShapeSource", dnn );
	CheckNeoOnnxSupport( shape->DimCount() == 1, "shape must have 1 dimension", *this );

	CPtr<COnnxExpandLayer> expandLayer = new COnnxExpandLayer( dnn.GetMathEngine() );
	expandLayer->SetName( Name() );
	expandLayer->Connect( 0, *input->Layer(), input->OutputIndex() );
	expandLayer->Connect( 1, *shape->Layer(), shape->OutputIndex() );
	dnn.AddLayer( *expandLayer );

	CTensorLayout outputLayout;
	outputLayout.SetBufferSize( shape->Shape()[0] );
	TBlobDim newDim = BD_BatchLength;
	while( outputLayout.Size() < shape->Shape()[0] - input->DimCount() ) {
		while( input->Layout().Find( newDim ) != NotFound ) {
			++newDim;
		}
		outputLayout.Add( newDim );
		++newDim;
	}
	outputLayout.Add( input->Layout() );
	CheckNeoOnnxSupport( newDim < BD_Count, "Too many dimensions", *this );
	outputLayout.CopyTo( expandLayer->TensorLayout() );

	outputs.Add( new CUserTensor( outputLayout, CLayerOutput( expandLayer.Ptr(), 0 ) ) );
}

} // namespace NeoOnnx
