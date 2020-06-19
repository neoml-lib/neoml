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

#include "GraphOutput.h"
#include "../NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CGraphOutput::CGraphOutput( const onnx::ValueInfoProto& output, CMap<CString, CInputInfo>& nodeOutputs ) :
	CNode( onnx::NodeProto(), nodeOutputs ),
	name( output.name().c_str() )
{
	input.Add( nodeOutputs.Get( output.name().c_str() ) );
}

void CGraphOutput::MarkTensorDims()
{
	CheckNeoOnnxInternal( !InputTensor( 0 ).GetTensorDim().IsEmpty(),
		"Graph output tensor's dimensions weren't marked with NeoML blob dimensions" );
}

void CGraphOutput::AddLayers( CDnn& dnn )
{
	CPtr<CSinkLayer> sink = new CSinkLayer( dnn.GetMathEngine() );
	sink->SetName( name );

	sink->Connect( 0, InputLayer( 0 ), InputLayerIndex( 0 ) );

	dnn.AddLayer( *sink );
}

} // namespace NeoOnnx
