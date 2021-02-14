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

#include "GraphInput.h"
#include "GraphCache.h"
#include "TensorUtils.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CGraphInput::CGraphInput( const onnx::ValueInfoProto& input ) :
	CNode( input.name(), {}, { input.name() } ),
	valueInfo( input )
{
}

bool CGraphInput::CanCalculateOutput( const CObjectArray<const CTensorBase>& /* inputs */ ) const
{
	// This node's purpose is providing user data to the net
	return false;
}

void CGraphInput::AddLayers( const CObjectArray<const CTensorBase>& /* inputs */,
	CObjectArray<const CTensorBase>& outputs, CDnn& dnn ) const
{
	CPtr<CSourceLayer> source = new CSourceLayer( dnn.GetMathEngine() );
	source->SetName( Name() );

	CTensorShape outputShape;
	outputShape.SetBufferSize( valueInfo.type().tensor_type().shape().dim_size() );
	for( const onnx::TensorShapeProto_Dimension& dim : valueInfo.type().tensor_type().shape().dim() ) {
		outputShape.Add( static_cast<int>( dim.dim_value() ) );
		// Replacing 'None' shape with 1
		if( outputShape.Last() == 0 ) {
			outputShape.Last() = 1;
		}
	}
	CheckNeoOnnxSupport( outputShape.Size() < BD_Count, "Tensor has too many dimensions" );

	CheckNeoOnnxSupport( valueInfo.type().has_tensor_type(), "Only tensors supported for graph input values" );
	CBlobDesc outputBlobDesc(
		GetBlobType( static_cast<onnx::TensorProto_DataType>( valueInfo.type().tensor_type().elem_type() ) ) );

	for( int dim = 0; dim < outputShape.Size(); ++dim ) {
		outputBlobDesc.SetDimSize( dim, outputShape[dim] );
	}
	CPtr<CDnnBlob> inputBlob = CDnnBlob::CreateBlob( dnn.GetMathEngine(), outputBlobDesc.GetDataType(), outputBlobDesc );
	source->SetBlob( inputBlob );

	dnn.AddLayer( *source );
	outputs[0] = new CUserTensor( outputShape, CTensorLayout(), CLayerOutput( source, 0 ) );
}

void CGraphInput::CalculateOutput( const CObjectArray<const CTensorBase>& /* inputs */,
	CObjectArray<const CTensorBase>& /* outputs */, IMathEngine& /* mathEngine */ ) const
{
	CheckNeoOnnxInternal( false, "Illegal call: CGraphInput::CalculateOutput" );
}

} // namespace NeoOnnx
