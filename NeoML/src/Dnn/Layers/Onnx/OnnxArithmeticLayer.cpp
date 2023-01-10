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

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/Layers/Onnx/OnnxArithmeticLayer.h>

namespace NeoML {

static CBlobDesc getBroadcastedSize( const CArray<CBlobDesc>& inputDescs )
{
	NeoPresume( !inputDescs.IsEmpty() );
	CBlobDesc resultDesc = inputDescs[0];
	NeoPresume( resultDesc.GetDataType() != CT_Invalid );
	for( int i = 1; i < inputDescs.Size(); ++i ) {
		NeoPresume( inputDescs[i].GetDataType() == resultDesc.GetDataType() );
		for( int dim = 0; dim < static_cast<int>( BD_Count ); ++dim ) {
			const int inputDimSize = inputDescs[i].DimSize( dim );
			const int resultDimSize = resultDesc.DimSize( dim );
			if( inputDimSize != resultDimSize ) {
				NeoPresume( std::min<int>( inputDimSize, resultDimSize ) == 1 );
				resultDesc.SetDimSize( dim, std::max<int>( inputDimSize, resultDimSize ) );
			}
		}
	}
	return resultDesc;
}

template<class T>
static void onnxArithmeticLayerImpl( COnnxArithmeticLayer::TOperation operation,
	CObjectArray<CDnnBlob>& inputs, CDnnBlob& output )
{
	IMathEngine& mathEngine = output.GetMathEngine();
	mathEngine.BroadcastCopy( output.GetData<T>(), inputs[0]->GetData<T>(),
		output.GetDesc(), inputs[0]->GetDesc(), 1 );
	CPtr<CDnnBlob> buff = nullptr;

	for( int inputIndex = 1; inputIndex < inputs.Size(); ++inputIndex ) {
		CPtr<CDnnBlob> buff = nullptr;
		CPtr<CDnnBlob> broadcastedInput = inputs[inputIndex];
		if( !inputs[inputIndex]->HasEqualDimensions( &output ) ) {
			if( buff == nullptr ) {
				buff = output.GetClone();
			}
			mathEngine.BroadcastCopy( buff->GetData<T>(), inputs[inputIndex]->GetData<T>(),
				buff->GetDesc(), inputs[inputIndex]->GetDesc(), 1 );
			broadcastedInput = buff;
		}
		switch( operation ) {
			case COnnxArithmeticLayer::TOperation::Add:
				mathEngine.VectorAdd( output.GetData<T>(), broadcastedInput->GetData<T>(),
					output.GetData<T>(), output.GetDataSize() );
				break;
			case COnnxArithmeticLayer::TOperation::Sub:
				mathEngine.VectorSub( output.GetData<T>(), broadcastedInput->GetData<T>(),
					output.GetData<T>(), output.GetDataSize() );
				break;
			case COnnxArithmeticLayer::TOperation::Mul:
				mathEngine.VectorEltwiseMultiply( output.GetData<T>(), broadcastedInput->GetData<T>(),
					output.GetData<T>(), output.GetDataSize() );
				break;
			case COnnxArithmeticLayer::TOperation::Div:
				mathEngine.VectorEltwiseDivide( output.GetData<T>(), broadcastedInput->GetData<T>(),
					output.GetData<T>(), output.GetDataSize() );
				break;
			default:
				NeoAssert( false );
		}
	}
}

static const int ArithmeticReshaperVersion = 0;

void COnnxArithmeticLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( ArithmeticReshaperVersion );
	CBaseReshaper::Serialize( archive );
	archive.SerializeEnum( operation );
}

void COnnxArithmeticLayer::CalculateShapes()
{
	CheckInputs();
	CheckOutputs();
	CheckArchitecture( GetOutputCount() == 1, GetPath(), "arithmetic operator with multiple outputs" );

	if( HasShapeInputs() ) {
		CArray<CBlobDesc> inputShapeDescs;
		for( int i = 0; i < inputShapeBlobs.Size(); ++i ) {
			CheckArchitecture( inputShapeBlobs[i] != nullptr, GetPath(), "Missing shape blob" );
			inputShapeDescs.Add( inputShapeBlobs[i]->GetDesc() );
		}
		CBlobDesc outputDesc = getBroadcastedSize( inputShapeDescs );
		outputShapeBlobs[0] = CDnnBlob::CreateBlob( inputShapeBlobs[0]->GetMathEngine(),
			outputDesc.GetDataType(), outputDesc );
		if( outputDesc.GetDataType() == CT_Float ) {
			onnxArithmeticLayerImpl<float>( operation, inputShapeBlobs, *outputShapeBlobs[0] );
		} else {
			onnxArithmeticLayerImpl<int>( operation, inputShapeBlobs, *outputShapeBlobs[0] );
		}
	} else {
		outputDescs[0] = getBroadcastedSize( inputDescs );
	}
}

void COnnxArithmeticLayer::RunOnce()
{
	if( !HasShapeInputs() ) {
		if( outputBlobs[0]->GetDataType() == CT_Float ) {
			onnxArithmeticLayerImpl<float>( operation, inputBlobs, *outputBlobs[0] );
		} else {
			onnxArithmeticLayerImpl<int>( operation, inputBlobs, *outputBlobs[0] );
		}
	}
}

} // namespace NeoML
