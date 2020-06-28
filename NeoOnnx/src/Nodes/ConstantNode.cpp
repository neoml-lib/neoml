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

#include "ConstantNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CConstantNode::CConstantNode( const onnx::NodeProto& constant, CMap<CString, CInputInfo>& nodeOutputs, IMathEngine& mathEngine ) :
	CNode( constant, nodeOutputs ),
	value( attributes.GetRequiredTensor( "value", mathEngine ) )
{
	CheckOnnxProtocol( input.Size() == 0, "node must have no inputs", constant );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", constant );
	
	CheckNeoOnnxSupport( value->GetDataSize() == 1, "'value' must be tensor of size 1", constant );
}

void CConstantNode::OnnxReshape()
{
	outputData.Add( CTensor( TT_ConstantTensor, { 1 }, value ) );
}

} // namespace NeoOnnx
