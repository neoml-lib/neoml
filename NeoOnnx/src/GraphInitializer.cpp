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

#include "GraphInitializer.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CGraphInitializer::CGraphInitializer( const onnx::TensorProto& _initializer ) :
	name( _initializer.name() ),
	initializer( _initializer )
{
}

CPtr<const CDataTensor> CGraphInitializer::GetDataTensor( IMathEngine& mathEngine )
{
	CTensorShape outputShape;
	outputShape.SetBufferSize( initializer.dims_size() );
	for( int dimIndex = 0; dimIndex < initializer.dims_size(); ++dimIndex ) {
		outputShape.Add( static_cast<int>( initializer.dims( dimIndex ) ) );
	}

	if( outputShape.IsEmpty() ) {
		// Tensor without dims is a scalar
		outputShape.Add( 1 );
	}

	CTensorLayout outputLayout( outputShape.Size() );
	CBlobDesc blobDesc;
	blobDesc.SetDataType( GetBlobType( static_cast<onnx::TensorProto_DataType>( initializer.data_type() ) ) );
	for( int dimIndex = 0; dimIndex < initializer.dims_size(); ++dimIndex ) {
		blobDesc.SetDimSize( outputLayout[dimIndex], outputShape[dimIndex] );
	}

	CPtr<CDnnBlob> outputBlob = CDnnBlob::CreateBlob( mathEngine, blobDesc.GetDataType(), blobDesc );
	if( blobDesc.GetDataType() == CT_Float ) {
		LoadBlobData<float>( initializer, *outputBlob );
	} else {
		LoadBlobData<int>( initializer, *outputBlob );
	}

	return new CDataTensor( outputShape, outputLayout, *outputBlob );
}

} // namespace NeoOnnx
