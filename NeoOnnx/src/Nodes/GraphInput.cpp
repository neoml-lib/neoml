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

CGraphInput::CGraphInput( int nodeIndex, const onnx::ValueInfoProto& _input ) :
	CNode( nodeIndex, 0, 1 ),
	name( _input.name().c_str() ),
	valueInfo( _input )
{
}

void CGraphInput::CalcOutputTensors( CGraphTensors& tensors, IMathEngine& mathEngine )
{
	CTensorShape& outputShape = OutputTensor( tensors, 0 ).Shape;
	outputShape.SetBufferSize( valueInfo.type().tensor_type().shape().dim_size() );
	for( const onnx::TensorShapeProto_Dimension dim : valueInfo.type().tensor_type().shape().dim() ) {
		outputShape.Add( static_cast<int>( dim.dim_value() ) );
	}

	// The OutputTensor( tensors, 0 ).Data was already set to nullptr in default constructor.
}

void CGraphInput::AddLayers( const CGraph& graph, const CGraphTensors& tensors, const CGraphDims& dims, CGraphMappings& mappings, CDnn& dnn )
{
	CPtr<CSourceLayer> source = new CSourceLayer( dnn.GetMathEngine() );
	source->SetName( name );

	CheckNeoOnnxSupport( valueInfo.type().has_tensor_type(), "Only tensors supported for graph input values" );
	CBlobDesc outputBlobDesc(
		GetBlobType( static_cast<onnx::TensorProto_DataType>( valueInfo.type().tensor_type().elem_type() ) ) );

	NeoOnnxCheck( OutputDim( dims, 0 ).Size() == OutputTensor( tensors, 0 ).Shape.Size(),
		"Graph input tensor's dimensions weren't marked with NeoML blob dimensions" );
	for( int i = 0; i < OutputDim( dims, 0 ).Size(); ++i ) {
		outputBlobDesc.SetDimSize( OutputDim( dims, 0 )[i], OutputTensor( tensors, 0 ).Shape[i] );
	}
	CPtr<CDnnBlob> inputBlob = CDnnBlob::CreateBlob( dnn.GetMathEngine(), outputBlobDesc.GetDataType(), outputBlobDesc );
	source->SetBlob( inputBlob );

	dnn.AddLayer( *source );

	OutputMapping( mappings, 0 ) = CNeoMLMapping( source, 0 );
}

} // namespace NeoOnnx
