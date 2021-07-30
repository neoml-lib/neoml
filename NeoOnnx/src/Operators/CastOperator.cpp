/* Copyright © 2017-2021 ABBYY Production LLC

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

#include "CastOperator.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CCastOperator::CCastOperator( const onnx::NodeProto& cast, int opsetVersion ) :
	COperator( cast, opsetVersion )
{
	// v1 - original
	// v6 - to attrbiute converted to integer instead of string
	// v9 - string type support is added
	// v13 - bloaf16 support is added
	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CCastOperator::ProcessTensors( const CTensorArray& inputs, CDnn& /* dnn */, CTensorArray& outputs ) const
{
	// TODO: add more detailed impl
	inputs.CopyTo( outputs );
}

} // namespace NeoOnnx
