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

CMatMulOperator::CMatMulOperator( const onnx::NodeProto& matMul, int opsetVersion ) :
	CLayerOperator( matMul, opsetVersion )
{
	// The differences between versions are in legacy optimization flags
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 2, "operator must have 2 inputs", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CMatMulOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	// The only scenario we support is this:
	//     first input - user-provided data, single matrix
	//     second input - pre-calculated data, single matrix
	// In this case we can emulate this operator by using CFullyConnectedLayer
	CheckOnnxProtocol( inputs[0] != nullptr && inputs[1] != nullptr, "input can't be optional", *this );
	NeoAssert( !inputs[0]->IsCalculated() );
	CheckNeoOnnxSupport( inputs[1]->IsCalculated(), "User-provided second input", *this );

	CPtr<const CDataTensor> weight = dynamic_cast<const CDataTensor*>( prepareTensor( *inputs[1], false ).Ptr() );
	CPtr<const CUserTensor> input = dynamic_cast<const CUserTensor*>( prepareTensor( *inputs[0], true ).Ptr() );

	const int batchSize = input->Shape()[0];
	const int inputElems = input->Shape()[1];
	const int outputElems = weight->Shape()[1];

	CPtr<CDnnBlob> weightBlob = CDnnBlob::CreateDataBlob( dnn.GetMathEngine(), CT_Float, 1, outputElems, inputElems );
	weightBlob->TransposeFrom( weight->Data(), weight->Layout()[0], weight->Layout()[1] );

	CPtr<CFullyConnectedLayer> fc = new CFullyConnectedLayer( dnn.GetMathEngine() );
	fc->SetName( Name() );
	fc->SetNumberOfElements( outputElems );
	fc->SetWeightsData( weightBlob );
	fc->SetZeroFreeTerm( true );
	fc->Connect( 0, *input->Layer(), input->OutputIndex() );
	dnn.AddLayer( *fc );

	// Prepend 1's to the shape if inputs had more than 2 dimensions
	CTensorShape outputShape;
	outputShape.Add( 1, max( 2, max( inputs[0]->DimCount(), inputs[1]->DimCount() ) ) - 2 );
	outputShape.Add( { batchSize, outputElems } );

	// Prepend unused blob dimensions if inputs had more than 2 dimensions
	CTensorLayout outputLayout = input->Layout();
	for( TBlobDim dim = BD_BatchLength; dim < BD_Count && outputLayout.Size() < outputShape.Size(); ++dim ) {
		if( outputLayout.Find( dim ) == NotFound ) {
			outputLayout.InsertAt( dim, outputLayout.Size() - 2 );
		}
	}

	outputs.Add( new CUserTensor( outputShape, outputLayout, CLayerOutput( fc, 0 ) ) );
}

// Prepares tensor for CFullyConnectedLayer
CPtr<const CTensorBase> CMatMulOperator::prepareTensor( const CTensorBase& tensor, bool isFirstArg ) const
{
	CPtr<const CTensorBase> currTensor = &tensor;

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

		if( currTensor->IsCalculated() ) {
			currTensor = new CDataTensor( newShape, newLayout, *dynamic_cast<const CDataTensor&>( *currTensor ).Data() );
		} else {
			currTensor = new CUserTensor( newShape, newLayout, dynamic_cast<const CUserTensor&>( *currTensor ).LayerOutput() );
		}
	}

	if( currTensor->DimCount() > 2 ) {
		// N-dimensional tensor is treated like a stack of Dim_0 * ... * Dim_N-3 matrices of size Dim_N-2 x Dim_N-1
		// But NeoOnnx supports only multiplicaiton of single matrices
		const int dimCount = currTensor->DimCount();

		// Check that every stack dimension is equal to 1
		for( int i = 0; i < dimCount - 2; ++i ) {
			CheckNeoOnnxSupport( currTensor->Shape()[i] == 1, "Non-trivial 2+-dimensional tensor", *this );
		}

		// Reinterpreting tensor as 2-dimensional
		CTensorShape newShape( { currTensor->Shape()[dimCount - 2], currTensor->Shape()[dimCount - 1] } );
		CTensorLayout newLayout( { currTensor->Layout()[dimCount - 2], currTensor->Layout()[dimCount - 1] } );
		if( currTensor->IsCalculated() ) {
			currTensor = new CDataTensor( newShape, newLayout, *dynamic_cast<const CDataTensor&>( *currTensor ).Data() );
		} else {
			currTensor = new CUserTensor( newShape, newLayout, dynamic_cast<const CUserTensor&>( *currTensor ).LayerOutput() );
		}
	}

	NeoAssert( currTensor->DimCount() == 2 );
	const CTensorLayout& tensorLayout = currTensor->Layout();
	if( tensorLayout[0] < BD_Height && tensorLayout[1] >= BD_Height ) {
		return currTensor;
	}

	return ConvertTensor( *currTensor, CTensorLayout( { BD_BatchWidth, BD_Channels } ) );
}

} // namespace NeoOnnx
