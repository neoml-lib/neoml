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

#include "GemmNode.h"
#include "FlattenNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CGemmNode::CGemmNode( const onnx::NodeProto& gemm, int opsetVersion, IMathEngine& /*mathEngine*/ ) :
	COpNode( gemm, opsetVersion ),
	alpha( attributes.GetOptionalFloat( "alpha", 1.f ) ),
	beta( attributes.GetOptionalFloat( "beta", 1.f ) ),
	transA( attributes.GetOptionalInt( "transA", 0 ) ),
	transB( attributes.GetOptionalInt( "transB", 0 ) )
{
	// Older versions have broadcast support
	CheckNeoOnnxSupport( opsetVersion >= 7 && opsetVersion <= MaxOpsetVersion, "opset version", gemm );

	CheckOnnxProtocol( input.Size() == 2 || input.Size() == 3, "node must have 2 or 3 inputs", gemm );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", gemm );

	CheckNeoOnnxSupport( alpha == 1.0f, "alpha != 1", gemm ); // TODO: add "alpha != 1.0" support.
	CheckNeoOnnxSupport( beta == 1.0f, "beta != 1", gemm ); // TODO: add "beta != 1.0" support.
	CheckNeoOnnxSupport( transA == 0, "transA != 0", gemm ); // TODO: add "TransA != 0" support.
	CheckNeoOnnxSupport( transB != 0, "transB == 0", gemm ); // TODO: add "TransB != 0" support.
}

void CGemmNode::CalcOutputShape()
{
	CheckNeoOnnxSupport( InputTensor( 0 ).Data == nullptr, "constant input", onnxNode );
	const CTensorShape& inputShape = InputTensor( 0 ).Shape;
	CheckOnnxProtocol( inputShape.Size() == 2, "input must be 2-dimensional", onnxNode );
	const int batchSize = inputShape[transA == 0 ? 0 : 1];
	const int inputObjectSize = inputShape[transA == 0 ? 1 : 0];

	CheckNeoOnnxSupport( InputTensor( 1 ).Data != nullptr, "non-constant weights", onnxNode );
	const CTensorShape& matrixShape = InputTensor( 1 ).Shape;
	CheckOnnxProtocol( matrixShape.Size() == 2, "weights must be 2-dimensional", onnxNode );
	CheckOnnxProtocol( matrixShape[transB == 0 ? 0 : 1] == inputObjectSize, "wrong weight size", onnxNode );
	const int numberOfElements = matrixShape[transB == 0 ? 1 : 0];

	if( input.Size() == 3 ) {
		CheckNeoOnnxSupport( InputTensor( 2 ).Data != nullptr, "non-constant bias", onnxNode );
		const CTensorShape& biasShape = InputTensor( 2 ).Shape;
		CheckOnnxProtocol( biasShape.Size() == 1, "bias must be 1-dimensional", onnxNode );
		CheckOnnxProtocol( biasShape[0] == numberOfElements, "wrong bias size", onnxNode );
	}

	output[0].Shape = { batchSize, numberOfElements };
}

void CGemmNode::CalcOutputData()
{
	CheckNeoOnnxSupport( InputTensor( 0 ).Data == nullptr, "output pre-calculation", onnxNode );
	// The output[0].Data was already set to nullptr in default constructor.
}

void CGemmNode::MarkTensorDims()
{
	// Gemm operator in onnx always works with 2-dimensional tensors.
	CheckNeoOnnxInternal( output[0].SetTensorDim( { BD_BatchWidth, BD_Channels } ),
		"marking output dimensions failed", onnxNode );
	CheckNeoOnnxInternal( InputTensor( 0 ).SetTensorDim( { BD_BatchWidth, BD_Channels } ),
		"marking input dimensions failed", onnxNode );
}

void CGemmNode::AddLayers( CDnn& net )
{
	CPtr<CFullyConnectedLayer> fc = new CFullyConnectedLayer( net.GetMathEngine() );
	fc->SetName( "NeoMLLayer" + Str( net.GetLayerCount() ) );

	const CTensorShape& matrixShape = InputTensor( 1 ).Shape;
	const int numberOfElements = matrixShape[transB == 0 ? 1 : 0];

	fc->SetNumberOfElements( numberOfElements );

	CPtr<CDnnBlob> weight = InputTensor( 1 ).Data->GetCopy();
	CBlobDesc weightDesc( CT_Float );
	weightDesc.SetDimSize( BD_BatchWidth, weight->GetDesc().DimSize( 0 ) );
	weightDesc.SetDimSize( BD_Channels, weight->GetDesc().DimSize( 1 ) );
	weight->ReinterpretDimensions( weightDesc );

	// If there is a 'Flatten' node before this, we need to reorder weights.
	weight = reorderWeightAfterFlatten( weight );

	fc->SetWeightsData( weight );

	if( input.Size() > 2 ) {
		fc->SetFreeTermData( InputTensor( 2 ).Data );
	} else {
		fc->SetZeroFreeTerm( true );
	}

	fc->Connect( 0, InputLayer( 0 ), InputLayerIndex( 0 ) );
	net.AddLayer( *fc );

	neoMLInputInfo.Add( CNeoMLInputInfo( fc, 0 ) );
}

// Reorders weight matrix if this 'Gemm' is located after 'Flatten'.
CPtr<CDnnBlob> CGemmNode::reorderWeightAfterFlatten( CDnnBlob* weight ) const
{
	const CFlattenNode* flatten = dynamic_cast<CFlattenNode*>( input[0].InputNode );
	
	if( flatten == nullptr ) {
		return weight;
	}

	const CTensorShape& flattenInputShape = flatten->InputTensor( 0 ).Shape;
	const CTensorDim& flattenInputDim = flatten->InputTensor( 0 ).Dim;

	CBlobDesc newWeightDesc( CT_Float );
	for( int dimIndex = 0; dimIndex < flattenInputShape.Size(); ++dimIndex ) {
		newWeightDesc.SetDimSize( flattenInputDim[dimIndex], flattenInputShape[dimIndex] );
	}

	if( ( newWeightDesc.Height() == 1 && newWeightDesc.Width() == 1 )
		|| ( newWeightDesc.Channels() == 1 && newWeightDesc.Depth() == 1 ) )
	{
		return weight;
	}

	// Weights needs conversion from CHW to HWC
	IMathEngine& mathEngine = weight->GetMathEngine();
	CPtr<CDnnBlob> newWeight = weight->GetClone();
	mathEngine.TransposeMatrix( weight->GetObjectCount(), weight->GetData(), newWeightDesc.Channels(), newWeightDesc.Depth(),
		newWeightDesc.Height() * newWeightDesc.Width(), 1, newWeight->GetData(), newWeight->GetDataSize() );
	return newWeight;
}

} // namespace NeoOnnx
