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

#include "RangeOperator.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

// Range operator inputs
enum TRangeOpInput
{
	ROI_Start, // First value of range
	ROI_Limit, // Limit value of range (exclusive)
	ROI_Delta, // Step value of range

	ROI_Count
};

CRangeOperator::CRangeOperator( const onnx::NodeProto& range, int opsetVersion ) :
	COperator( range, opsetVersion )
{
	// v11 - original
	CheckOnnxProtocol( OpsetVersion >= 11, "Range operator was introduced in opset v11", *this );
	CheckNeoOnnxSupport( OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	static_assert( ROI_Count == 3, "ROI_Count != 3" );
	CheckOnnxProtocol( InputCount() == ROI_Count, "operator must have 3 inputs", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

template<class T>
static CPtr<const CDataTensor> generateRange( const CObjectArray<const CDnnBlob>& inputBlobs )
{
	const T start = inputBlobs[ROI_Start]->GetData<T>().GetValue();
	const T limit = inputBlobs[ROI_Limit]->GetData<T>().GetValue();
	const T delta = inputBlobs[ROI_Delta]->GetData<T>().GetValue();
	const int numberOfElements = static_cast<int>( max( T( ceil( ( limit - start ) / delta ) ), T( 0 ) ) );

	if( numberOfElements == 0 ) {
		// ONNX doesn't clarify what to do in this case
		// But tensor without shape is considered to be a scalar
		// That's why return start value
		return new CDataTensor( {}, {}, *inputBlobs[ROI_Start] );
	}

	TBlobType blobType = CBlobType<T>::GetType();
	CTensorLayout resultLayout( 1 );
	CBlobDesc resultBlobDesc( blobType );
	resultBlobDesc.SetDimSize( resultLayout[0], numberOfElements );
	CPtr<CDnnBlob> resultBlob = CDnnBlob::CreateBlob( inputBlobs[0]->GetMathEngine(), blobType, resultBlobDesc );
	T* buffer = resultBlob->GetBuffer<T>( 0, resultBlob->GetDataSize(), false );
	T currValue = start;
	for( int i = 0; i < numberOfElements; ++i ) {
		buffer[i] = currValue;
		currValue += delta;
	}
	resultBlob->ReleaseBuffer( buffer, true );
	return new CDataTensor( { numberOfElements }, resultLayout, *resultBlob );
}

void CRangeOperator::ProcessTensors( const CTensorArray& inputs, CDnn& /* dnn */, CTensorArray& outputs ) const
{
	// NeoML doesn't have alternative to Range operator
	// That's why the only scenario is supported: when all of the input arguments are calculated
	CObjectArray<const CDnnBlob> inputBlobs;
	inputBlobs.SetBufferSize( ROI_Count );
	for( int i = 0; i < ROI_Count; ++i ) {
		CheckOnnxProtocol( inputs[i] != nullptr, "input can't be optional", *this );
		CheckNeoOnnxSupport( inputs[i]->IsCalculated(), "user-provided input", *this );
		inputBlobs.Add( dynamic_cast<const CDataTensor*>( inputs[i].Ptr() )->Data() );
		CheckOnnxProtocol( inputBlobs[i]->GetDataSize() == 1, "input must be a scalar", *this );
	}

	outputs.Add( inputBlobs[0]->GetDataType() == CT_Float
		? generateRange<float>( inputBlobs ).Ptr()
		: generateRange<int>( inputBlobs ).Ptr() );
}

} // namespace NeoOnnx
