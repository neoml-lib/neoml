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

#include "NonZeroOperator.h"
#include "NeoOnnxCheck.h"

namespace NeoOnnx {

template<class T>
static void nonZeroImpl( const CDataTensor& input, CTensorArray& outputs )
{
	CArray<T> inputBuffer;
	inputBuffer.SetSize( input.Data()->GetDataSize() );
	input.Data()->CopyTo( inputBuffer.GetPtr() );

	int nonZeroElements = 0;
	for( int i = 0; i < inputBuffer.Size(); ++i ) {
		if( inputBuffer[i] != 0 ) {
			++nonZeroElements;
		}
	}

	IMathEngine& mathEngine = input.Data()->GetMathEngine();
	CPtr<CDnnBlob> outputBlob = CDnnBlob::CreateTensor( mathEngine, CT_Int, { input.DimCount(), nonZeroElements});
	CDnnBlobBuffer<int> outputBuffer( *outputBlob, 0, outputBlob->GetDataSize(), TDnnBlobBufferAccess::Write );
	int outIndex = 0;
	for( int i = 0; i < inputBuffer.Size(); ++i ) {
		if( inputBuffer[i] != 0 ) {
			int flatIndex = i;
			for( int dim = input.DimCount() - 1; dim >= 0; --dim ) {
				outputBuffer[outIndex + dim * nonZeroElements] = flatIndex % input.Shape()[dim];
				flatIndex /= input.Shape()[dim];
			}
			outIndex++;
		}
	}

	NeoAssert( outIndex == nonZeroElements );
	outputs.Add( new CDataTensor( CTensorShape{ input.DimCount(), nonZeroElements },
		CTensorLayout{ BD_BatchLength, BD_BatchWidth }, *outputBlob ) );
}

// --------------------------------------------------------------------------------------------------------------------

CNonZeroOperator::CNonZeroOperator( const onnx::NodeProto& nonZero, int opsetVersion ) :
	COperator( nonZero, opsetVersion )
{
	// v9 - original
	// v13 - new data types are supported
	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CNonZeroOperator::ProcessTensors( const CTensorArray& inputs, CDnn& /* dnn */, CTensorArray& outputs ) const
{
	CheckOnnxProtocol( inputs[0] != nullptr, "input can't be optional", *this );
	CheckNeoOnnxSupport( inputs[0]->IsCalculated(), "user-provided input", *this );

	CPtr<const CDataTensor> currInput = dynamic_cast<const CDataTensor*>( inputs[0].Ptr() );
	if( IsTransposedLayout( currInput->Layout() ) ) {
		currInput = ConvertTensor( *currInput, CTensorLayout( currInput->DimCount() ) );
	}

	CPtr<CDnnBlob> outBlob = nullptr;
	if( currInput->Data()->GetDataType() == CT_Float ) {
		nonZeroImpl<float>( *currInput, outputs );
	} else {
		nonZeroImpl<int>( *currInput, outputs );
	}
}

} // namespace NeoOnnx
