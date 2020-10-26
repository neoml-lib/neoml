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

#include "EltwiseNode.h"
#include "GraphCache.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

// Checks if tensor shapes are equal
bool isEqual( const CTensorShape& first, const CTensorShape& second )
{
	if( first.Size() != second.Size() ) {
		return false;
	}

	for( int i = 0; i < first.Size(); ++i ) {
		if( first[i] != second[i] ) {
			return false;
		}
	}

	return true;
}

// Calculates shape of the result of the numpy-style broadcast operation
// If shapes can be broadcasted writes broadcasted shape to the result and returns true
// Returns false if shapes can't be broadcasted (in this case result will be empty)
bool broadcastShape( const CTensorShape& first, const CTensorShape& second, CTensorShape& result )
{
	int dimCount = max( first.Size(), second.Size() );
	result.SetSize( dimCount );

	for( int i = 0; i < dimCount; ++i ) {
		const int dimFirst = first.Size() - 1 - i >= 0 ? first[first.Size() - 1 - i] : 1;
		const int dimSecond = second.Size() - 1 - i >= 0 ? second[second.Size() - 1 - i] : 1;
		if( dimFirst != dimSecond && dimFirst != 1 && dimSecond != 1 ) {
			result.Empty();
			return false;
		}
		result[dimCount - 1 - i] = max( dimFirst, dimSecond );
	}

	return true;
}

//---------------------------------------------------------------------------------------------------------------------

CEltwiseNodeBase::CEltwiseNodeBase( int nodeIndex, const onnx::NodeProto& eltwise, int opsetVersion,
		TOperation _operation, int _argsNum ) :
	COpNode( nodeIndex, eltwise, opsetVersion ),
	operation( _operation ),
	argsNum( _argsNum ),
	userInputCached( -1 )
{
	if( argsNum > 0 ) {
		CheckOnnxProtocol( InputCount() == argsNum, "expected " + Str( argsNum ) + " arguments", eltwise );
	}
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", eltwise );
}

void CEltwiseNodeBase::CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine )
{
	// Calculate output shape
	CTensorShape outputShape;
	tensors[Input[0]].Shape.CopyTo( outputShape );
	bool hasUserProvidedData = tensors[Input[0]].Data == nullptr;
	for( int i = 1; i < InputCount(); ++i ) {
		CTensorShape currShape;
		outputShape.CopyTo( currShape );
		CheckOnnxProtocol( broadcastShape( currShape, tensors[Input[i]].Shape, outputShape ),
			"input tensors can't be broadcasted", OnnxNode );
		hasUserProvidedData |= ( tensors[Input[i]].Data == nullptr );
	}
	outputShape.CopyTo( tensors[Output[0]].Shape );

	if( !hasUserProvidedData ) {
		// All of the inputs contain pre-calculated data
		// We can simply pre-calculate this layer's output
		tensors[Output[0]].Data = precalcOutput( tensors, outputShape, mathEngine );
		return;
	}

	// NeoML doesn't support numpy-style broadcast
	// That's why we can manually broadcast pre-calculated data and add it via CSourceLayer
	// But user-provided data has to have the same shape as output (there is no way to broadcast it in NeoML)
	for( int i = 0; i < InputCount(); ++i ) {
		CheckNeoOnnxSupport( tensors[Input[i]].Data != nullptr || isEqual( tensors[Input[i]].Shape, outputShape ),
			"broadcastable user data", OnnxNode );
	}

	// NeoML doesn't have EltwiseDiv or EltwiseSub layers
	// NeoOnnx emulates them by EltwiseMul(X, 1 / Y), EltwiseSum(X, -Y)
	// But that's possible only if second tensor is pre-calculated
	if( operation == O_Sub || operation == O_Div ) {
		CheckNeoOnnxInternal( argsNum == 2, "wrong number of arguments", OnnxNode );
		CheckNeoOnnxInternal( tensors[Input[1]].Data != nullptr, "user data as second argument", OnnxNode );
	}
}

void CEltwiseNodeBase::LabelTensorDims( const CTensorCache &tensors, CDimCache &dims )
{
	if( tensors[Output[0]].Data != nullptr ) {
		// If the data was pre-calculated there is no need in labeling
		return;
	}

	const int inputIndex = userInput( tensors );
	if( !dims[Input[inputIndex]].IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( tensors[Output[0]].Shape, dims[Input[inputIndex]], dims[Output[0]] ),
			"labeling output dimensions failed", OnnxNode );
	}

	if( !dims[Output[0]].IsEmpty() ) {
		for( int i = 0; i < InputCount(); ++i ) {
			if( tensors[Input[i]].Data == nullptr ) {
				CheckNeoOnnxInternal( SetTensorDim( tensors[Input[i]].Shape, dims[Output[0]], dims[Input[i]] ),
					"labeling input dimensions failed", OnnxNode );
			}
		}
	}
}

void CEltwiseNodeBase::AddLayers( const CGraph& graph, const CTensorCache& tensors, const CDimCache& dims,
	CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	if( tensors[Output[0]].Data != nullptr ) {
		// If the data was pre-calculated there is no need in adding layers
		return;
	}

	IMathEngine& mathEngine = dnn.GetMathEngine();
	CPtr<CBaseLayer> eltwiseLayer;
	static_assert( O_Count == 4, "O_Count != 4" );
	switch( operation ) {
		case O_Add:
		case O_Sub:
			eltwiseLayer = new CEltwiseSumLayer( mathEngine );
			break;
		case O_Mul:
		case O_Div:
			eltwiseLayer = new CEltwiseMulLayer( mathEngine );
			break;
		default:
			CheckNeoOnnxInternal( false, "wrong operation type", OnnxNode );
	}
	eltwiseLayer->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	for( int i = 0; i < InputCount(); ++i ) {
		if( tensors[Input[i]].Data == nullptr ) {
			eltwiseLayer->Connect( i, *neoMLLinks[Input[i]].Layer, neoMLLinks[Input[i]].OutputIndex );
		} else {
			CPtr<CSourceLayer> source = new CSourceLayer( mathEngine );
			source->SetName( eltwiseLayer->GetName() + CString( "_input" ) + Str( i ) );
			eltwiseLayer->Connect( i, *source );
			dnn.AddLayer( *source );
			CPtr<CDnnBlob> data = broadcast( tensors[Input[i]], tensors[Output[0]].Shape, dims[Output[0]],
				operation == O_Sub && i == 1, operation == O_Div && i == 1 );
			source->SetBlob( data );
		}
	}

	dnn.AddLayer( *eltwiseLayer );
	neoMLLinks[Output[0]] = CNeoMLLink( eltwiseLayer, 0 );
}

// Returns index of input with data provided by user
// assert if there is no such inputs
int CEltwiseNodeBase::userInput( const CTensorCache& tensors ) const
{
	if( userInputCached < 0 ) {
		// If this method hasn't been called before
		for( int i = 0; i < InputCount(); ++i ) {
			if( tensors[Input[i]].Data == nullptr ) {
				userInputCached = i;
				break;
			}
		}
		CheckNeoOnnxInternal( userInputCached >= 0, "userInput() was called when there's no user data", OnnxNode );
	}

	return userInputCached;
}

// Broadcasts blob with unlabeled dimensions to the blob with unlabeled dimensions with shape equal to outputShape
CPtr<CDnnBlob> CEltwiseNodeBase::broadcast( const CTensor& input, const CTensorShape& outputShape, bool negative,
	bool inverted ) const
{
	CTensorDim outputDim;
	for( int i = 0; i < outputShape.Size(); ++i ) {
		outputDim.Add( static_cast<TBlobDim>( i ) );
	}
	return broadcast( input, outputShape, outputDim, negative, inverted );
}

// Broadcasts blob with unlabeled dimensions to the blob with labeled dimensions with shape equal to outputShape
// If negative is true then the result will be multiplied by -1
// If inverted is true then the 1 will be divided by result
CPtr<CDnnBlob> CEltwiseNodeBase::broadcast( const CTensor &input, const CTensorShape &outputShape, const CTensorDim& outputDim, 
	bool negative, bool inverted ) const
{
	CheckNeoOnnxInternal( input.Data != nullptr, "cannot broadcast tensor with user data", OnnxNode );
	CBlobDesc outputDesc( CT_Float );
	for( int i = 0; i < outputShape.Size(); ++i ) {
		outputDesc.SetDimSize( outputDim[i], outputShape[i] );
	}
	CPtr<CDnnBlob> output = CDnnBlob::CreateBlob( input.Data->GetMathEngine(), outputDesc );

	// inputShape with 1 added to the beginning (if needed)
	CFastArray<int, 8> inputShape;
	input.Shape.CopyTo( inputShape );
	inputShape.InsertAt( 1, 0, outputShape.Size() - inputShape.Size() );

	// offset by each of coordinates of inputShape
	CFastArray<int, 8> inputOffset;
	inputOffset.SetSize( inputShape.Size() );
	inputOffset[inputOffset.Size() - 1] = 1;
	for( int i = inputOffset.Size() - 2; i >= 0; --i ) {
		inputOffset[i] = inputOffset[i - 1] * inputShape[i - 1];
	}

	float* inputBuff = input.Data->GetBuffer<float>( 0, input.Data->GetDataSize() );
	float* outputBuff = output->GetBuffer<float>( 0, output->GetDataSize() );

	for( int outputIndex = 0; outputIndex < outputDesc.BlobSize(); ++outputIndex ) {
		// We can't just broadcast copy data because of difference in ordering
		// If unlabeled tensor 32 x 3 x 3 is broadcasted to 1 x 32 x 3 x 3 (B x Ch x H x W)
		// we've got to repack data because NeoML uses channel-last ordering while Onnx uses channel-first
		// That's why we have to calculate inputOffset, manually convert outputIndex to inputIndex and copy element one-by-one

		// Calculating input index
		int inputIndex = 0;
		int curr = outputIndex;
		for( int dim = BD_Count - 1; dim >= 0; --dim ) {
			const int dimSize = outputDesc.DimSize( dim );
			if( dimSize > 1 ) {
				const int coord = curr % dimSize;
				curr /= dimSize;
				const int dimIndex = outputDim.Find( static_cast<TBlobDim>( dim ) );
				inputIndex += ( coord % inputShape[dimIndex] ) * inputOffset[dimIndex];
			}
		}

		// Copying data
		outputBuff[outputIndex] = inputBuff[inputIndex];

		// Applying additional operations (if needed)
		if( negative ) {
			outputBuff[outputIndex] *= -1;
		}
		if( inverted ) {
			outputBuff[outputIndex] = 1.f / outputBuff[outputIndex];
		}
	}

	output->ReleaseBuffer( outputBuff, true );
	input.Data->ReleaseBuffer( inputBuff, false );

	return output;
}

// Pre-calculates output
// This case is 
CPtr<CDnnBlob> CEltwiseNodeBase::precalcOutput( const CTensorCache& tensors, const CTensorShape& outputShape,
	IMathEngine& mathEngine ) const
{
	CBlobDesc desc( CT_Float );
	for( int i = 0; i < outputShape.Size(); ++i ) {
		desc.SetDimSize( i, outputShape[i] );
	}
	CPtr<CDnnBlob> output = CDnnBlob::CreateBlob( mathEngine, desc );
	
	static_assert( O_Count == 4, "O_Count != 4" );
	output->Fill( operation == O_Add || operation == O_Sub ? 0.f : 1.f );
	
	for( int i = 0; i < InputCount(); ++i ) {
		CheckNeoOnnxInternal( tensors[Input[i]].Data != nullptr, "precalcOutput was called with user data", OnnxNode );
		CPtr<CDnnBlob> operand = broadcast( tensors[Input[i]], outputShape, operation == O_Sub && i == 1,
			operation == O_Div && i == 1 );
		switch( operation ) {
			case O_Add:
			case O_Sub:
				output->Add( operand );
				break;
			case O_Mul:
			case O_Div:
				mathEngine.VectorEltwiseMultiply( output->GetData(), operand->GetData(), output->GetData(),
					output->GetDataSize() );
				break;
			default:
				CheckNeoOnnxInternal( false, "unknown operation", OnnxNode );
		}
	}

	return output;
}

} // namespace NeoOnnx
