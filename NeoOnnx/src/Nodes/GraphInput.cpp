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
#include "../TensorUtils.h"
#include "../NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CGraphInput::CGraphInput( const onnx::ValueInfoProto& _input, CMap<CString, CInputInfo>& nodeOutputs ) :
	CNode( onnx::NodeProto(), nodeOutputs ),
	name( _input.name().c_str() ),
	valueInfo( _input )
{
	nodeOutputs.Add( name, CInputInfo( this, 0 ) );
}

void CGraphInput::OnnxReshape()
{
	CTensorShape shape;
	shape.SetBufferSize( valueInfo.type().tensor_type().shape().dim_size() );
	for( const onnx::TensorShapeProto_Dimension dim : valueInfo.type().tensor_type().shape().dim() ) {
		shape.Add( static_cast<int>( dim.dim_value() ) );
	}

	outputData.Add( CTensor( TT_DataTensor, shape ) );
}

void CGraphInput::AddLayers( CDnn& net )
{
	CPtr<CSourceLayer> source = new CSourceLayer( net.GetMathEngine() );
	source->SetName( name );

	CheckNeoOnnxSupport( valueInfo.type().has_tensor_type(), "Only tensors supported for graph input values" );
	CBlobDesc outputBlobDesc(
		GetBlobType( static_cast<onnx::TensorProto_DataType>( valueInfo.type().tensor_type().elem_type() ) ) );

	NeoOnnxCheck( outputData[0].GetTensorDim().Size() == outputData[0].GetShape().Size(),
		"Graph input tensor's dimensions weren't marked with NeoML blob dimensions" );
	for( int i = 0; i < outputData[0].GetTensorDim().Size(); ++i ) {
		outputBlobDesc.SetDimSize( outputData[0].GetTensorDim()[i], outputData[0].GetShape()[i] );
	}
	CPtr<CDnnBlob> inputBlob = CDnnBlob::CreateBlob( net.GetMathEngine(), outputBlobDesc.GetDataType(), outputBlobDesc );
	source->SetBlob( inputBlob );

	net.AddLayer( *source );

	outputInfo.Add( COutputInfo( source, 0 ) );
}

} // namespace NeoOnnx
