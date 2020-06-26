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

#include "ClipNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CClipNode::CClipNode( const onnx::NodeProto& clip, CMap<CString, CInputInfo>& nodeOutputs ) :
	CNode( clip, nodeOutputs ),
	minValue( attributes.GetOptionalFloat( "min", -FLT_MAX ) ),
	maxValue( attributes.GetOptionalFloat( "max", FLT_MAX ) )
{
	CheckOnnxProtocol( input.Size() == 1, "node must have 1 input", clip );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", clip );
}

void CClipNode::OnnxReshape()
{
	CheckNeoOnnxSupport( InputTensor( 0 ).GetType() == TT_DataTensor, "constant input", onnxNode );

	outputData.Add( InputTensor( 0 ) );
}

void CClipNode::MarkTensorDims()
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

void CClipNode::AddLayers( CDnn& dnn )
{
	CheckNeoOnnxSupport( minValue == 0.f, "'min' value must be equal to 0", onnxNode );

	CPtr<CReLULayer> relu = new CReLULayer( dnn.GetMathEngine() );
	relu->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	if( maxValue < FLT_MAX ) {
		relu->SetUpperThreshold( maxValue );
	}

	relu->Connect( 0, InputLayer( 0 ), InputLayerIndex( 0 ) );
	dnn.AddLayer( *relu );

	outputInfo.Add( COutputInfo( relu, 0 ) );
}

} // namespace NeoOnnx
