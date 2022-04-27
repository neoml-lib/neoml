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

namespace NeoOnnx {

CExpandOperator::CExpandOperator( const onnx::NodeProto& expand, int opsetVersion ) :
	CLayerOperator( expand, opsetVersion )
{
	// v8 - original
	// v13 - new data types are supported
	CheckOnnxProtocol( InputCount() == 2, "operator must have 2 inputs", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CExpandOperator::AddLayers( const CTensorArray& inputs, CDnn& /* dnn */, CTensorArray& outputs ) const
{
	CheckOnnxProtocol( inputs[0] != nullptr && inputs[1] != nullptr, "inputs can't be optional", *this );
	CheckNeoOnnxSupport( inputs[1]->IsCalculated(), "user-provided shape", *this );

	const CDnnBlob* shapeBlob = dynamic_cast<const CDataTensor&>( *inputs[1] ).Data();
	CTensorShape outputShape;
	CheckOnnxProtocol( inputs[0]->DimCount() >= shapeBlob->GetDataSize(),
		"Number of input dims is less than length of the shape", *this );
	const int preservedInputDims = inputs[0]->DimCount() - shapeBlob->GetDataSize();
	for( int i = 0; i < preservedInputDims; ++i ) {
		outputShape.Add( inputs[0]->Shape()[i]);
	}

	outputShape.SetSize( outputShape.Size() + shapeBlob->GetDataSize() );
	shapeBlob->CopyTo( outputShape.GetPtr() + preservedInputDims );

	outputs.Add( BroadcastTensor( *inputs[0], CBroadcast( BT_Numpy ), outputShape ) );
}

} // namespace NeoOnnx
