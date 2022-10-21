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

#include "MatMulOperator.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

// Gets MatMul's batch size of the given tensor
static inline int getBatchSize( const CTensorBase& tensor )
{
	// All the dimensions which are prior to height and width are interpreted as a batch
	int batchSize = 1;
	for( int i = 0; i < tensor.DimCount() - 2; ++i ) {
		batchSize *= tensor.Shape()[i];
	}
	return batchSize;
}

// Gets MatMul's matrix height of the given tensor
static inline int getMatrixHeight( const CTensorBase& tensor )
{
	NeoAssert( tensor.DimCount() >= 2 );
	return tensor.Shape()[tensor.DimCount() - 2];
}

// Gets MatMul's matrix width of the given tensor
static inline int getMatrixWidth( const CTensorBase& tensor )
{
	return tensor.Shape().Last();
}

//---------------------------------------------------------------------------------------------------------------------

CMatMulOperator::CMatMulOperator( const onnx::NodeProto& matMul, int opsetVersion ) :
	CLayerOperator( matMul, opsetVersion )
{
	// v1 - original
	// v9 - integer data is supported
	// v13 - bfloat16 data is supported
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 2, "operator must have 2 inputs", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CMatMulOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckOnnxProtocol( inputs[0] != nullptr && inputs[1] != nullptr, "input can't be optional", *this );

	// NeoML doesn't support batch broadcast of the first argument in matrix multiplication
	const int firstBatch = getBatchSize( *inputs[0] );
	const int secondBatch = getBatchSize( *inputs[1] );
	CheckNeoOnnxSupport( firstBatch == secondBatch || secondBatch == 1, "Second argument batch broadcast" );

	CPtr<const CUserTensor> first = prepareTensor( *inputs[0], true, firstBatch != secondBatch, dnn );
	CPtr<const CUserTensor> second = prepareTensor( *inputs[1], false, false, dnn );
	const int outputHeight = getMatrixHeight( *first );
	const int outputWidth = getMatrixWidth( *second );

	CPtr<CMatrixMultiplicationLayer> matmul = new CMatrixMultiplicationLayer( dnn.GetMathEngine() );
	matmul->SetName( Name() );
	matmul->Connect( 0, *first->Layer(), first->OutputIndex() );
	matmul->Connect( 1, *second->Layer(), second->OutputIndex() );
	dnn.AddLayer( *matmul );

	const CTensorShape& biggerShape = inputs[1]->DimCount() > inputs[0]->DimCount() ? inputs[1]->Shape()
		: inputs[0]->Shape();
	if( inputs[0]->DimCount() > 5 || inputs[1]->DimCount() > inputs[0]->DimCount() || firstBatch != secondBatch ) {
		// We have to transform matrix multiplication output because it doesn't support more than 3 batch dims
		// or because CMatrixMultiplicationLayer takes batch and matrix height dimensions from the first input
		// or because batch is merged with the height
		CTensorLayout outputLayout( biggerShape.Size() );
		CPtr<CTransformLayer> transform = new CTransformLayer( dnn.GetMathEngine() );
		transform->SetName( Name() + "_TransformOutput" );
		CTensorShape outputShape;
		biggerShape.CopyTo( outputShape );
		if( inputs[1]->DimCount() > inputs[0]->DimCount() && firstBatch != secondBatch ) {
			// Corner case: the number of batch dimensions is used from the second input
			// But the sizes of these dimensions must be taken from the first input
			// E.g. firstArg is (N,H,K) and secondArg is (1,1,1,K,W) and output is (1,1,N,H,W)
			const int dimDiff = inputs[1]->DimCount() - inputs[0]->DimCount();
			for( int firstDimIndex = 0; firstDimIndex < inputs[0]->DimCount() - 2; ++firstDimIndex ) {
				outputShape[firstDimIndex + dimDiff] = inputs[0]->Shape()[firstDimIndex];
			}
		}
		outputShape[outputShape.Size() - 2] = outputHeight;
		outputShape.Last() = outputWidth;
		for( TBlobDim dim = BD_BatchLength; dim < BD_Count; ++dim ) {
			const int dimIndex = outputLayout.Find( dim );
			transform->SetDimensionRule( dim, CTransformLayer::O_SetSize,
				dimIndex == NotFound ? 1 : outputShape[dimIndex] );
		}
		if( outputShape.Size() > 2 ) {
			// Heuristic for dynamic batch size
			transform->SetDimensionRule( outputLayout[0], CTransformLayer::O_Remainder, 1 );
		}
		transform->Connect( *matmul );
		dnn.AddLayer( *transform );
		outputs.Add( new CUserTensor( outputShape, outputLayout, CLayerOutput( transform, 0 ) ) );
		return;
	}

	CTensorShape outputShape;
	outputShape.SetBufferSize( max( 2, biggerShape.Size() ) );
	CTensorLayout outputLayout;
	outputLayout.SetBufferSize( outputShape.BufferSize() );
	for( int i = 0; i < biggerShape.Size() - 2; ++i ) {
		outputShape.Add( biggerShape[i] );
		// CMatrixMultiplicationLayer forms output mostly on the first input
		outputLayout.Add( first->Layout()[i] );
	}
	// It's guaranteed by prepareTensor that matrix height is located in the BD_Height
	outputShape.Add( outputHeight );
	outputLayout.Add( BD_Height );
	// It's required by CMatrixMultiplication layer that matrix width is located in the BD_Channels
	outputShape.Add( outputWidth );
	outputLayout.Add( BD_Channels );

	outputs.Add( new CUserTensor( outputShape, outputLayout, CLayerOutput( matmul, 0 ) ) );
}

// Prepares tensor for CMatrixMultiplicationLayer
// It puts matrix height into BD_Height
// It puts matrix width into BD_Channels
CPtr<const CUserTensor> CMatMulOperator::prepareTensor( const CTensorBase& tensor, bool isFirstArg,
	bool mergeBatchWithHeight, CDnn& dnn ) const
{
	CPtr<const CUserTensor> currTensor = AsUserTensor( tensor, Name() + ( isFirstArg ? "_Input0" : "_Input1" ), dnn );

	if( currTensor->DimCount() == 1 ) {
		CTensorLayout newLayout = currTensor->Layout();
		CTensorShape newShape;
		currTensor->Shape().CopyTo( newShape );

		// If first argument is 1-dimensional, then 1 must be prepended to the dimensions
		// If second aragument is 1-dimensional, then 1 must be appended to the dimensions
		if( isFirstArg ) {
			newShape.InsertAt( 1, 0 );
			newLayout.InsertAt( newLayout[0] == BD_BatchLength ? BD_BatchWidth : BD_BatchLength, 0 );
		} else {
			newShape.Add( 1 );
			newLayout.Add( newLayout[0] == BD_Channels ? BD_Depth : BD_Channels );
		}

		currTensor = new CUserTensor( newShape, newLayout, dynamic_cast<const CUserTensor&>( *currTensor ).LayerOutput() );
	}

	if( currTensor->DimCount() > 5 || mergeBatchWithHeight ) {
		// This is a tricky case because NeoML doesn't have enough batch dimensions
		// or batch and height must be merged

		// Step 1: we must guarantee that tensor is not transposed in memory (before transforming it)
		if( IsTransposedLayout( currTensor->Layout() ) ) {
			currTensor = dynamic_cast<const CUserTensor*>(
				ConvertTensor( *currTensor, CTensorLayout( currTensor->DimCount() ) ).Ptr() );
		}
		// Step 2: transform it into Batch x Height x Channels for CMatrixMultiplicationLayer
		CPtr<CTransformLayer> transform = new CTransformLayer( dnn.GetMathEngine() );

		const int batchSize = getBatchSize( *currTensor );
		const int height = getMatrixHeight( *currTensor );
		const int width = getMatrixWidth( *currTensor );

		transform->SetName( Name() + ( isFirstArg ? "_Transform0" : "_Transform1" ) );

		for( TBlobDim dim = BD_BatchLength; dim < BD_Channels; ++dim ) {
			transform->SetDimensionRule( dim, CTransformLayer::O_SetSize, 1 );
		}

		if( mergeBatchWithHeight ) {
			transform->SetDimensionRule( BD_Height, CTransformLayer::O_Remainder, 1 );
		} else {
			transform->SetDimensionRule( BD_BatchLength, CTransformLayer::O_Remainder, 1 );
			transform->SetDimensionRule( BD_Height, CTransformLayer::O_InputDim,
				currTensor->Layout()[currTensor->DimCount() - 2] );
		}
		transform->SetDimensionRule( BD_Channels, CTransformLayer::O_InputDim, currTensor->Layout().Last() );

		transform->Connect( 0, *currTensor->Layer(), currTensor->OutputIndex() );
		dnn.AddLayer( *transform );
		return new CUserTensor( CTensorShape( { batchSize, height, width } ),
			CTensorLayout( { BD_BatchLength, BD_Height, BD_Channels } ),
			CLayerOutput( transform, 0 ) );
	}

	CTensorLayout outputLayout( { BD_BatchLength, BD_BatchWidth, BD_ListSize, BD_Height, BD_Channels } );
	outputLayout.DeleteAt( 0, outputLayout.Size() - currTensor->DimCount() );
	return ConvertTensor( *currTensor, outputLayout );
}

} // namespace NeoOnnx
