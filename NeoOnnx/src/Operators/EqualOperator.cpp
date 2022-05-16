/* Copyright © 2017-2022 ABBYY Production LLC

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

#include "onnx.pb.h"

#include "EqualOperator.h"
#include "NeoOnnxCheck.h"

namespace NeoOnnx {

template<class T>
static void equalOpImpl( const CDataTensor& a, const CDataTensor& b, CTensorArray& outputs )
{
	CPtr<CDnnBlob> resultBlob = CDnnBlob::CreateBlob( a.Data()->GetMathEngine(), CT_Int, a.Data()->GetDesc() );

	CDnnBlobBuffer<T> aBuffer( const_cast<CDnnBlob&>( *a.Data() ), 0, a.Data()->GetDataSize(), TDnnBlobBufferAccess::Read );
	CDnnBlobBuffer<T> bBuffer( const_cast<CDnnBlob&>( *b.Data() ), 0, b.Data()->GetDataSize(), TDnnBlobBufferAccess::Read );
	CDnnBlobBuffer<int> resultBuffer( *resultBlob, 0, resultBlob->GetDataSize(), TDnnBlobBufferAccess::Write );

	for( int i = 0; i < resultBlob->GetDataSize(); ++i ) {
		resultBuffer[i] = ( aBuffer[i] == bBuffer[i] ) ? 1 : 0;
	}

	outputs.Add( new CDataTensor( a.Shape(), a.Layout(), *resultBlob ) );
}

static CPtr<const CDataTensor> prepareEqualOpTensor( const CTensorBase& input, const CBroadcast& broadcast,
	const CTensorShape& outputShape )
{
	CPtr<const CTensorBase> currInput = BroadcastTensor( input, broadcast, outputShape );
	if( IsTransposedLayout( currInput->Layout() ) ) {
		currInput = ConvertTensor( *currInput, CTensorLayout( currInput->DimCount() ) );
	}
	return dynamic_cast<const CDataTensor*>( currInput.Ptr() );
}

// --------------------------------------------------------------------------------------------------------------------

CEqualOperator::CEqualOperator( const onnx::NodeProto& equal, int opsetVersion ) :
	COperator( equal, opsetVersion )
{
	// v1 - original
	// v7 - changes in broadcast rules
	// v11 - new data types supported
	// v13 - bfloat16 supported
	CheckOnnxProtocol( InputCount() == 2, "operator must have 2 inputs", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CEqualOperator::ProcessTensors( const CTensorArray& inputs, CDnn& /* dnn */, CTensorArray& outputs ) const
{
	CheckOnnxProtocol( inputs[0] != nullptr, "A can't be optional", *this );
	CheckOnnxProtocol( inputs[1] != nullptr, "B can't be optional", *this );

	CheckNeoOnnxSupport( inputs[0]->IsCalculated(), "user-provided A", *this );
	CheckNeoOnnxSupport( inputs[1]->IsCalculated(), "user-provided B", *this );

	CBroadcast broadcast( BT_None, NotFound );
	if( OpsetVersion < 7 ) {
		int broadcastAttr = 0;
		GetAttribute( "broadcast", broadcastAttr );
		if( broadcastAttr != 0 ) {
			broadcast.Type = BT_Onnx;
			GetAttribute( "axis", broadcast.Axis );
		}
	} else {
		broadcast.Type = BT_Numpy;
	}

	CTensorShape outputShape;
	CheckNeoOnnxSupport( BroadcastTensorShape( inputs[0]->Shape(), inputs[1]->Shape(), broadcast, outputShape ),
		"unbroadcastable A and B", *this );

	CPtr<const CDataTensor> a = prepareEqualOpTensor( *inputs[0], broadcast, outputShape );
	CPtr<const CDataTensor> b = prepareEqualOpTensor( *inputs[1], broadcast, outputShape );

	CheckNeoOnnxSupport( a->Data()->GetDataType() == b->Data()->GetDataType(), "A and B data type mismatch", *this );
	if( a->Data()->GetDataType() == CT_Float ) {
		equalOpImpl<float>( *a, *b, outputs );
	} else if( a->Data()->GetDataType() == CT_Int ) {
		equalOpImpl<int>( *a, *b, outputs );
	} else {
		CheckNeoOnnxSupport( false, "unsupported data type", *this );
	}
}

} // namespace NeoOnnx

