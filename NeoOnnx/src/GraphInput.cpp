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

#include "onnx.pb.h"

#include "GraphInput.h"
#include "TensorUtils.h"
#include "NeoOnnxCheck.h"

namespace NeoOnnx {

CGraphInput::CGraphInput( const onnx::ValueInfoProto& input ) :
	name( input.name().c_str() ),
	valueInfo( input )
{
}

CPtr<const CUserTensor> CGraphInput::AddSourceLayer( CDnn& dnn ) const
{
	CPtr<CSourceLayer> source = new CSourceLayer( dnn.GetMathEngine() );
	source->SetName( Name() );
	// Store blob in order to save input sizes through serialization
	source->StoreBlob( true );

	CTensorShape outputShape;
	outputShape.SetBufferSize( valueInfo.type().tensor_type().shape().dim_size() );
	for( const onnx::TensorShapeProto_Dimension& dim : valueInfo.type().tensor_type().shape().dim() ) {
		outputShape.Add( static_cast<int>( dim.dim_value() ) );
		// Replace 'None' shape with 1
		if( outputShape.Last() == 0 ) {
			outputShape.Last() = 1;
		}
	}
	const int dimCount = outputShape.Size();
	CheckNeoOnnxSupport( dimCount <= BD_Count, "Tensor has too many dimensions" );

	CheckNeoOnnxSupport( valueInfo.type().has_tensor_type(), "Only tensors supported for graph input values" );
	CBlobDesc outputBlobDesc(
		GetBlobType( static_cast<onnx::TensorProto_DataType>( valueInfo.type().tensor_type().elem_type() ) ) );

	CTensorLayout outputLayout = CTensorLayout::IOLayout( dimCount );
	for( int dimIndex = 0; dimIndex < dimCount; ++dimIndex ) {
		outputBlobDesc.SetDimSize( outputLayout[dimIndex], outputShape[dimIndex] );
	}
	CPtr<CDnnBlob> inputBlob = CDnnBlob::CreateBlob( dnn.GetMathEngine(), outputBlobDesc.GetDataType(), outputBlobDesc );
	source->SetBlob( inputBlob );

	dnn.AddLayer( *source );
	return new CUserTensor( outputShape, outputLayout, CLayerOutput( source, 0 ) );
}

} // namespace NeoOnnx

