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

#include "GatherOperator.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CGatherOperator::CGatherOperator( const onnx::NodeProto& gather, int opsetVersion ) :
	COperator( gather, opsetVersion )
{
	// v1 - original
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 2, "operator must have 2 inputs", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CGatherOperator::ProcessTensors( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	// This is a stub for a specific case: integer 1-dimensional data
	CheckNeoOnnxSupport( inputs[0] != nullptr && inputs[0]->IsCalculated(), "User-provided data", *this );
	CheckNeoOnnxSupport( inputs[0]->DimCount() == 1, "2+ dimensional data", *this );
	const CDnnBlob* dataBlob = dynamic_cast<const CDataTensor*>( inputs[0].Ptr() )->Data();
	CheckNeoOnnxSupport( dataBlob->GetDataType() == CT_Int, "non-integer data", *this );

	CheckNeoOnnxSupport( inputs[1] != nullptr && inputs[1]->IsCalculated(), "User-provided indices", *this );
	CheckNeoOnnxSupport( inputs[1]->DimCount() <= 1, "2+ dimensional indices", *this );
	const CDnnBlob* indicesBlob = dynamic_cast<const CDataTensor*>( inputs[1].Ptr() )->Data();
	NeoAssert( indicesBlob->GetDataType() == CT_Int );

	CPtr<CDnnBlob> resultBlob = CDnnBlob::CreateBlob( dnn.GetMathEngine(), CT_Int, indicesBlob->GetDesc() );
	
	// const_cast in order to avoid copying (becasuse we won't change dataBlob or indicesBlob contents anyway)
	int* data = const_cast<CDnnBlob*>( dataBlob )->GetBuffer<int>( 0, dataBlob->GetDataSize(), true );
	int* indices = const_cast<CDnnBlob*>( indicesBlob )->GetBuffer<int>( 0, indicesBlob->GetDataSize(), true );

	int* result = resultBlob->GetBuffer<int>( 0, resultBlob->GetDataSize(), false );

	for( int i = 0; i < indicesBlob->GetDataSize(); ++i ) {
		result[i] = data[indices[i]];
	}

	resultBlob->ReleaseBuffer( result, true );
	const_cast<CDnnBlob*>( indicesBlob )->ReleaseBuffer( indices, false );
	const_cast<CDnnBlob*>( dataBlob )->ReleaseBuffer( data, false );

	outputs.Add( new CDataTensor( inputs[1]->Shape(), inputs[1]->Layout(), *resultBlob ) );
}

} // namespace NeoOnnx
