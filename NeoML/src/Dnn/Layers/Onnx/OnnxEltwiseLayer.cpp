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

#include <NeoML/Dnn/Layers/Onnx/OnnxEltwiseLayer.h>

namespace NeoML {

// Checks whether operation is arithmetics or logical
// Where operation is considered logical
static bool isOnnxEltwiseLogicalOperation( COnnxEltwiseLayer::TOperation operation )
{
	static_assert( static_cast<int>( COnnxEltwiseLayer::TOperation::Count ) == 10, "TOperation::Count != 10" );
	return operation == COnnxEltwiseLayer::TOperation::Less
		|| operation == COnnxEltwiseLayer::TOperation::Greater
		|| operation == COnnxEltwiseLayer::TOperation::Equal
		|| operation == COnnxEltwiseLayer::TOperation::LessOrEqual
		|| operation == COnnxEltwiseLayer::TOperation::GreaterOrEqual
		|| operation == COnnxEltwiseLayer::TOperation::Where;
}

// Returns the type of inputs during the calculation
static TBlobType getOnnxEltwiseOperationType( const CObjectArray<CDnnBlob>& inputs )
{
	NeoPresume( !inputs.IsEmpty() );
	if( inputs.Size() == 1 ) {
		return inputs[0]->GetDataType();
	}
	// We try to not use the first input as an indicator because of Where operation (mask in Where is always integer)
	return inputs[1]->GetDataType();
}

// Calculates the output desc of eltwise operation including all the broadcasts required
static CBlobDesc getOnnxEltwiseOutputDesc( COnnxEltwiseLayer::TOperation operation, const CArray<CBlobDesc>& inputDescs )
{
	NeoPresume( !inputDescs.IsEmpty() );
	CBlobDesc resultDesc = inputDescs.Size() == 1 ? inputDescs[0] : inputDescs[1];
	NeoPresume( resultDesc.GetDataType() != CT_Invalid );
	for( int i = 0; i < inputDescs.Size(); ++i ) {
		for( int dim = 0; dim < static_cast<int>( BD_Count ); ++dim ) {
			const int inputDimSize = inputDescs[i].DimSize( dim );
			const int resultDimSize = resultDesc.DimSize( dim );
			if( inputDimSize != resultDimSize ) {
				NeoAssert( std::min<int>( inputDimSize, resultDimSize ) == 1 );
				resultDesc.SetDimSize( dim, std::max<int>( inputDimSize, resultDimSize ) );
			}
		}
	}

	static_assert( static_cast<int>( COnnxEltwiseLayer::TOperation::Count ) == 10, "TOperation::Count != 10" );
	if( isOnnxEltwiseLogicalOperation( operation ) && operation != COnnxEltwiseLayer::TOperation::Where ) {
		resultDesc.SetDataType( CT_Int );
	}

	return resultDesc;
}

// Performs arithmetic operation over the blobs of data type T
template<class T>
static void onnxArithmeticOperationImpl( COnnxEltwiseLayer::TOperation operation,
	CObjectArray<CDnnBlob>& inputs, CDnnBlob& output )
{
	IMathEngine& mathEngine = output.GetMathEngine();
	mathEngine.BroadcastCopy( output.GetData<T>(), inputs[0]->GetData<T>(),
		output.GetDesc(), inputs[0]->GetDesc(), 1 );
	CPtr<CDnnBlob> buff = nullptr;

	for( int inputIndex = 1; inputIndex < inputs.Size(); ++inputIndex ) {
		CPtr<CDnnBlob> buff = nullptr;
		CPtr<CDnnBlob> broadcastedInput = inputs[inputIndex];
		// Allocate broadcast buffef only if needed
		if( !inputs[inputIndex]->HasEqualDimensions( &output ) ) {
			if( buff == nullptr ) {
				buff = output.GetClone();
			}
			mathEngine.BroadcastCopy( buff->GetData<T>(), inputs[inputIndex]->GetData<T>(),
				buff->GetDesc(), inputs[inputIndex]->GetDesc(), 1 );
			broadcastedInput = buff;
		}
		static_assert( static_cast<int>( COnnxEltwiseLayer::TOperation::Count ) == 10, "TOperation::Count != 10" );
		switch( operation ) {
			case COnnxEltwiseLayer::TOperation::Add:
				mathEngine.VectorAdd( output.GetData<T>(), broadcastedInput->GetData<T>(),
					output.GetData<T>(), output.GetDataSize() );
				break;
			case COnnxEltwiseLayer::TOperation::Sub:
				mathEngine.VectorSub( output.GetData<T>(), broadcastedInput->GetData<T>(),
					output.GetData<T>(), output.GetDataSize() );
				break;
			case COnnxEltwiseLayer::TOperation::Mul:
				mathEngine.VectorEltwiseMultiply( output.GetData<T>(), broadcastedInput->GetData<T>(),
					output.GetData<T>(), output.GetDataSize() );
				break;
			case COnnxEltwiseLayer::TOperation::Div:
				mathEngine.VectorEltwiseDivide( output.GetData<T>(), broadcastedInput->GetData<T>(),
					output.GetData<T>(), output.GetDataSize() );
				break;
			default:
				NeoAssert( false );
		}
	}
}

// Broadcasts input for logical operation (if needed)
static CPtr<CDnnBlob> broadcastOnnxLogicalInput( CDnnBlob& input, const CDnnBlob& output )
{
	if( input.HasEqualDimensions( &output ) ) {
		return &input;
	}

	CPtr<CDnnBlob> preparedInput = CDnnBlob::CreateBlob( input.GetMathEngine(),
		input.GetDataType(), output.GetDesc() );
	if( input.GetDataType() == CT_Float ) {
		input.GetMathEngine().BroadcastCopy( preparedInput->GetData(), input.GetData(),
			preparedInput->GetDesc(), input.GetDesc(), 1 );
	} else {
		input.GetMathEngine().BroadcastCopy( preparedInput->GetData<int>(), input.GetData<int>(),
			preparedInput->GetDesc(), input.GetDesc(), 1 );
	}
	return preparedInput;
}

// Performs logical operation over the blobs of data type T
template<class T>
static void onnxLogicalOperationImpl( COnnxEltwiseLayer::TOperation operation,
	CObjectArray<CDnnBlob>& inputs, CDnnBlob& output )
{
	NeoPresume( inputs.Size() == 2 || inputs.Size() == 3 );
	CPtr<CDnnBlob> firstArg = broadcastOnnxLogicalInput( *inputs[0], output );
	CPtr<CDnnBlob> secondArg = broadcastOnnxLogicalInput( *inputs[1], output );
	CPtr<CDnnBlob> thirdArg = inputs.Size() == 2 ? nullptr : broadcastOnnxLogicalInput( *inputs[2], output );

	IMathEngine& mathEngine = output.GetMathEngine();
	const int dataSize = output.GetDataSize();
	static_assert( static_cast<int>( COnnxEltwiseLayer::TOperation::Count ) == 10, "TOperation::Count != 10" );
	switch( operation ) {
		case COnnxEltwiseLayer::TOperation::Less:
			mathEngine.VectorEltwiseLess( firstArg->GetData<T>(), secondArg->GetData<T>(),
				output.GetData<int>(), dataSize );
			break;
		case COnnxEltwiseLayer::TOperation::Greater:
			mathEngine.VectorEltwiseLess( secondArg->GetData<T>(), firstArg->GetData<T>(),
				output.GetData<int>(), dataSize );
			break;
		case COnnxEltwiseLayer::TOperation::Equal:
			mathEngine.VectorEltwiseEqual( secondArg->GetData<T>(), firstArg->GetData<T>(),
				output.GetData<int>(), dataSize );
			break;
		case COnnxEltwiseLayer::TOperation::LessOrEqual:
			// LessOrEqual == Not(Greater)
			mathEngine.VectorEltwiseLess( secondArg->GetData<T>(), firstArg->GetData<T>(),
				output.GetData<int>(), dataSize );
			mathEngine.VectorEltwiseNot( output.GetData<int>(), output.GetData<int>(), output.GetDataSize() );
			break;
		case COnnxEltwiseLayer::TOperation::GreaterOrEqual:
			// GreaterOrEqual == Not(Less)
			mathEngine.VectorEltwiseLess( firstArg->GetData<T>(), secondArg->GetData<T>(),
				output.GetData<int>(), dataSize );
			mathEngine.VectorEltwiseNot( output.GetData<int>(), output.GetData<int>(), output.GetDataSize() );
			break;
		case COnnxEltwiseLayer::TOperation::Where:
			mathEngine.VectorEltwiseWhere( firstArg->GetData<int>(), secondArg->GetData<T>(), thirdArg->GetData<T>(),
				output.GetData<T>(), dataSize );
			break;
		default:
			NeoAssert( false );
	}
}

// Performs eltwise operation over the blobs of data type T
template<class T>
static void onnxEltwiseOperationImpl( COnnxEltwiseLayer::TOperation operation,
	CObjectArray<CDnnBlob>& inputs, CDnnBlob& output )
{
	if( isOnnxEltwiseLogicalOperation( operation ) ) {
		onnxLogicalOperationImpl<T>( operation, inputs, output );
		return;
	}

	onnxArithmeticOperationImpl<T>( operation, inputs, output );
}

//---------------------------------------------------------------------------------------------------------------------

static const int OnnxEltwiseLayerVersion = 0;

void COnnxEltwiseLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxEltwiseLayerVersion );
	COnnxLayerBase::Serialize( archive );
	archive.SerializeEnum( operation );
}

void COnnxEltwiseLayer::CalculateShapes()
{
	CheckInputs();
	CheckOutputs();
	CheckLayerArchitecture( GetOutputCount() == 1, "arithmetic operator with multiple outputs" );

	if( inputShapeBlobs[0] != nullptr ) {
		// The output is a shape-blob, that's why it's needed to be calculated here
		CArray<CBlobDesc> inputShapeDescs;
		for( int i = 0; i < inputShapeBlobs.Size(); ++i ) {
			CheckLayerArchitecture( inputShapeBlobs[i] != nullptr, "Missing shape blob" );
			inputShapeDescs.Add( inputShapeBlobs[i]->GetDesc() );
		}
		CBlobDesc outputDesc = getOnnxEltwiseOutputDesc( operation, inputShapeDescs );
		outputShapeBlobs[0] = CDnnBlob::CreateBlob( inputShapeBlobs[0]->GetMathEngine(),
			outputDesc.GetDataType(), outputDesc );

		if( getOnnxEltwiseOperationType( inputShapeBlobs ) == CT_Float ) {
			onnxEltwiseOperationImpl<float>( operation, inputShapeBlobs, *outputShapeBlobs[0] );
		} else {
			onnxEltwiseOperationImpl<int>( operation, inputShapeBlobs, *outputShapeBlobs[0] );
		}
	} else {
		// The output will be calculated during RunOnce
		outputDescs[0] = getOnnxEltwiseOutputDesc( operation, inputDescs );
		EnableInPlace( inputDescs[0].HasEqualDimensions( outputDescs[0] )
			&& inputDescs[0].GetDataType() == outputDescs[0].GetDataType() && InputsMayBeOverwritten() );
	}
}

void COnnxEltwiseLayer::RunOnce()
{
	if( inputShapeBlobs[0] == nullptr ) {
		if( getOnnxEltwiseOperationType( inputBlobs ) == CT_Float ) {
			onnxEltwiseOperationImpl<float>( operation, inputBlobs, *outputBlobs[0] );
		} else {
			onnxEltwiseOperationImpl<int>( operation, inputBlobs, *outputBlobs[0] );
		}
	}
}

} // namespace NeoML
