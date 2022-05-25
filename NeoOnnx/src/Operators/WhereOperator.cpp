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

#include "WhereOperator.h"
#include "NeoOnnxCheck.h"

namespace NeoOnnx {

template<class TCondition, class TData>
static void whereOpImpl( const CDataTensor& condition, const CDataTensor& x, const CDataTensor& y,
	CTensorArray& outputs )
{
	CPtr<CDnnBlob> resultBlob = CDnnBlob::CreateBlob( x.Data()->GetMathEngine(), x.Data()->GetDataType(), x.Data()->GetDesc() );

	CDnnBlobBuffer<TCondition> conditionBuffer( const_cast<CDnnBlob&>( *condition.Data() ), 0, condition.Data()->GetDataSize(),
		TDnnBlobBufferAccess::Read );
	CDnnBlobBuffer<TData> xBuffer( const_cast<CDnnBlob&>( *x.Data() ), 0, x.Data()->GetDataSize(), TDnnBlobBufferAccess::Read );
	CDnnBlobBuffer<TData> yBuffer( const_cast<CDnnBlob&>( *y.Data() ), 0, y.Data()->GetDataSize(), TDnnBlobBufferAccess::Read );
	CDnnBlobBuffer<TData> resultBuffer( *resultBlob, 0, resultBlob->GetDataSize(), TDnnBlobBufferAccess::Write );

	for( int i = 0; i < resultBlob->GetDataSize(); ++i ) {
		resultBuffer[i] = ( conditionBuffer[i] != 0 ) ? xBuffer[i] : yBuffer[i];
	}

	outputs.Add( new CDataTensor( x.Shape(), x.Layout(), *resultBlob ) );
}

static CPtr<const CDataTensor> prepareWhereOpTensor( const CTensorBase& input, const CTensorShape& resultShape )
{
	CPtr<const CTensorBase> currInput = BroadcastTensor( input, CBroadcast( BT_Numpy ), resultShape );
	if( IsTransposedLayout( currInput->Layout() ) ) {
		currInput = ConvertTensor( *currInput, CTensorLayout( currInput->DimCount() ) );
	}
	return dynamic_cast<const CDataTensor*>( currInput.Ptr() );
}

// --------------------------------------------------------------------------------------------------------------------

CWhereOperator::CWhereOperator( const onnx::NodeProto& where, int opsetVersion ) :
	COperator( where, opsetVersion )
{
	// v9 - original
	// v16 - support bfloat16 data type
	CheckOnnxProtocol( InputCount() == 3, "operator must have 3 inputs", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CWhereOperator::ProcessTensors( const CTensorArray& inputs, CDnn& /* dnn */, CTensorArray& outputs ) const
{
	CheckOnnxProtocol( inputs[0] != nullptr, "condition can't be optional", *this );
	CheckOnnxProtocol( inputs[1] != nullptr, "X can't be optional", *this );
	CheckOnnxProtocol( inputs[2] != nullptr, "Y can't be optional", *this );

	CheckNeoOnnxSupport( inputs[0]->IsCalculated(), "user-provided condition", *this );
	CheckNeoOnnxSupport( inputs[1]->IsCalculated(), "user-provided X", *this );
	CheckNeoOnnxSupport( inputs[2]->IsCalculated(), "user-provided Y", *this );

	CBroadcast broadcast( BT_Numpy );
	CTensorShape intermediateShape;
	CTensorShape resultShape;

	CheckNeoOnnxSupport( BroadcastTensorShape( inputs[0]->Shape(), inputs[1]->Shape(), broadcast, intermediateShape ),
		"unbroadcastable condition and X", *this );
	CheckNeoOnnxSupport( BroadcastTensorShape( intermediateShape, inputs[2]->Shape(), broadcast, resultShape ),
		"unbroadcastable Y", *this );

	CPtr<const CDataTensor> condition = prepareWhereOpTensor( *inputs[0], resultShape );
	CPtr<const CDataTensor> x = prepareWhereOpTensor( *inputs[1], resultShape );
	CPtr<const CDataTensor> y = prepareWhereOpTensor( *inputs[2], resultShape );

	CheckNeoOnnxSupport( x->Data()->GetDataType() == y->Data()->GetDataType(), "X and Y data type mismatch", *this );
	const TBlobType conditionType = condition->Data()->GetDataType();
	const TBlobType xType = x->Data()->GetDataType();
	if( conditionType == CT_Float && xType == CT_Float ) {
		whereOpImpl<float, float>( *condition, *x, *y, outputs );
	} else if( conditionType == CT_Float && xType == CT_Int ) {
		whereOpImpl<float, int>( *condition, *x, *y, outputs );
	} else if( conditionType == CT_Int && xType == CT_Float ) {
		whereOpImpl<int, float>( *condition, *x, *y, outputs );
	} else if( conditionType == CT_Int && xType == CT_Int ) {
		whereOpImpl<int, int>( *condition, *x, *y, outputs );
	} else {
		CheckNeoOnnxSupport( false, "unsupported data type", *this );
	}
}

} // namespace NeoOnnx

