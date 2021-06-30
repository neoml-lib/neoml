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

#include "ConstantOperator.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CConstantOperator::CConstantOperator( const onnx::NodeProto& constant, int opsetVersion ) :
	COperator( constant, opsetVersion )
{
	// v1 - original
	// v9 - supported new data types
	// v11 - added "sparse_value" attribute
	// v12 - added new attributes: "value_*"
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 0, "operator must have no inputs", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );

	if( OpsetVersion >= 11 ) {
		// In ONNX value can be passed through various attributes
		// like 'sparse_value', 'value_float', 'value_ints' etc.
		// NeoOnnx supports only "value" attribute
		CheckNeoOnnxSupport( HasAttribute( "value" ), "'*_value' or 'value_*' attributes", *this );
	}
}

void CConstantOperator::ProcessTensors( const CTensorArray& /* inputs */, CDnn& dnn, CTensorArray& outputs ) const
{
	CPtr<CDataTensor> value( new CDataTensor( dnn.GetMathEngine() ) );
	CheckOnnxProtocol( GetAttribute( "value", value ), "'value' attribute is missing" );
	outputs[0] = value.Ptr();
}

} // namespace NeoOnnx
