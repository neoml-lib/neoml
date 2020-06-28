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
#include "../TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CGraphInitializer::CGraphInitializer( const onnx::TensorProto& _initializer, CMap<CString, CInputInfo>& nodeOutputs, IMathEngine& _mathEngine ) :
	CNode( onnx::NodeProto(), nodeOutputs ),
	mathEngine( _mathEngine ),
	initializer( _initializer )
{
	assert( initializer.dims_size() > 0 );
	nodeOutputs.Add( initializer.name().c_str(), CInputInfo( this, 0 ) );
}

void CGraphInitializer::OnnxReshape()
{
	CTensorShape shape;
	shape.SetBufferSize( initializer.dims_size() );

	CBlobDesc blobDesc;
	blobDesc.SetDataType( GetBlobType( static_cast<onnx::TensorProto_DataType>( initializer.data_type() ) ) );
	for( int dimIndex = 0; dimIndex < initializer.dims_size(); ++dimIndex ) {
		shape.Add( static_cast<int>( initializer.dims( dimIndex ) ) );
		blobDesc.SetDimSize( dimIndex, shape.Last() );
	}

	CPtr<CDnnBlob> blob = CDnnBlob::CreateBlob( mathEngine, blobDesc.GetDataType(), blobDesc );
	if( blobDesc.GetDataType() == CT_Float ) {
		LoadBlobData<float>( initializer, *blob );
	} else {
		LoadBlobData<int>( initializer, *blob );
	}

	outputData.Add( CTensor( TT_ConstantTensor, shape, blob ) );
}

} // namespace NeoOnnx
