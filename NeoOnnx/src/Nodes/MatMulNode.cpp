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

#include "MatMulNode.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CMatMulNode::CMatMulNode( const onnx::NodeProto& matMul, int opsetVersion ) :
	CLayerOpNode( matMul, opsetVersion )
{
	// The differences between versions are in legacy optimization flags
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 2, "node must have 2 inputs", *this );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", *this );
}

void CMatMulNode::AddLayers( const CObjectArray<const CTensorBase>& inputs,
	CObjectArray<const CTensorBase>& outputs, CDnn& dnn )
{
	// The only scenario we support is this:
	//     first input - user-provided data, single matrix
	//     second input - pre-calculated data, single matrix
	// In this case we can emulate this node by CFullyConnectedLayer
	CheckNeoOnnxInternal( inputs[0] != nullptr && !inputs[0]->IsCalculated(), "First input must be provided by user", *this );
	CheckNeoOnnxInternal( inputs[1] != nullptr, "Undefined second input", *this );
	CheckNeoOnnxSupport( inputs[1]->IsCalculated(), "User-provided second input", *this );

	CheckNeoOnnxSupport( inputs[0]->DimCount() == 2, "3+ dimensional first input", *this );
	CheckNeoOnnxSupport( inputs[1]->DimCount() == 2, "3+ dimensional second input", *this );

	const int batchSize = inputs[0]->Shape()[0];
	const int inputElems = inputs[0]->Shape()[1];
	const int outputElems = inputs[1]->Shape()[1];

	CTensorShape outputShape( { batchSize, outputElems } );
	CTensorLayout layout( { BD_BatchWidth, BD_Channels } );

	// We need to convert weight to correct pack
	CPtr<const CDataTensor> weight = dynamic_cast<const CDataTensor*>( prepareTensor( *inputs[1] ).Ptr() );
	CPtr<const CUserTensor> input = dynamic_cast<const CUserTensor*>( prepareTensor( *inputs[0] ).Ptr() );

	CPtr<CDnnBlob> weightBlob = CDnnBlob::CreateDataBlob( dnn.GetMathEngine(), CT_Float, 1, outputElems, inputElems );
	weightBlob->TransposeFrom( weight->Data(), weight->Layout()[0], weight->Layout()[1] );

	CPtr<CFullyConnectedLayer> fc = new CFullyConnectedLayer( dnn.GetMathEngine() );
	fc->SetName( Name() );
	fc->SetNumberOfElements( outputElems );
	fc->SetWeightsData( weightBlob );
	fc->SetZeroFreeTerm( true );
	fc->Connect( 0, *input->Layer(), input->OutputIndex() );
	dnn.AddLayer( *fc );

	outputs[0] = new CUserTensor( outputShape, layout, CLayerOutput( fc, 0 ) );
}

// Prepares tensor for CFullyConnectedLayer
CPtr<const CTensorBase> CMatMulNode::prepareTensor( const CTensorBase& tensor ) const
{
	const CTensorLayout& inputLayout = tensor.Layout();
	CheckNeoOnnxInternal( inputLayout.Size() == 2, "input must be 2 dimensional", *this );
	if( inputLayout[0] < BD_Height && inputLayout[1] >= BD_Height ) {
		return &tensor;
	}

	return ConvertTensor( tensor, CTensorLayout( { BD_BatchWidth, BD_Channels } ) );
}

} // namespace NeoOnnx
