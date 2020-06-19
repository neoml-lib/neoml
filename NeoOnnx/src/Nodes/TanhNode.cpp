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

#include "../common.h"
#pragma hdrstop

#include "TanhNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CTanhNode::CTanhNode( const onnx::NodeProto& tanh, CMap<CString, CInputInfo>& nodeOutputs ) :
	CNode( tanh, nodeOutputs )
{
	CheckOnnxProtocol( input.Size() == 1, "node must have 1 input", tanh );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", tanh );
}

void CTanhNode::OnnxReshape()
{
	CheckNeoOnnxSupport( InputTensor( 0 ).GetType() == TT_DataTensor,
		"constant input", onnxNode );

	outputData.Add( InputTensor( 0 ) );
}

void CTanhNode::MarkTensorDims()
{
	if( !InputTensor( 0 ).GetTensorDim().IsEmpty() ) {
		CheckNeoOnnxInternal( outputData[0].SetTensorDim( InputTensor( 0 ).GetTensorDim() ),
			"marking output dimensions failed", onnxNode );
	}

	if( !outputData[0].GetTensorDim().IsEmpty() ) {
		CheckNeoOnnxInternal( InputTensor( 0 ).SetTensorDim( outputData[0].GetTensorDim() ),
			"marking input dimensions failed", onnxNode );
	}
}

void CTanhNode::AddLayers( CDnn& dnn )
{
	CPtr<CTanhLayer> tanh = new CTanhLayer( dnn.GetMathEngine() );
	tanh->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	tanh->Connect( 0, InputLayer( 0 ), InputLayerIndex( 0 ) );
	
	dnn.AddLayer( *tanh );

	outputInfo.Add( COutputInfo( tanh, 0 ) );
}

} // namespace NeoOnnx
