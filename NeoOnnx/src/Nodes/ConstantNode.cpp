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

#include "proto/onnx.pb.h"

namespace NeoOnnx {

CConstantNode::CConstantNode( const onnx::NodeProto& constant, IMathEngine& mathEngine ) :
	CNode( constant ),
	value( attributes.GetRequiredTensor( "value", mathEngine ) )
{
	CheckOnnxProtocol( input.Size() == 0, "node must have no inputs", constant );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", constant );
	
	// TODO: add other values support?
	CheckNeoOnnxSupport( value->GetDataSize() == 1, "'value' must be tensor of size 1", constant );
}

void CConstantNode::CalcOutputShape()
{
	output[0].Shape = { 1 };
}

void CConstantNode::CalcOutputData()
{
	output[0].Data = value;
}

} // namespace NeoOnnx
