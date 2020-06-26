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

#include "AddNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CAddNode::CAddNode( const onnx::NodeProto& add, CMap<CString, CInputInfo>& nodeOutputs ) :
	CNode( add, nodeOutputs )
{
	CheckOnnxProtocol( input.Size() == 2, "node must have 2 inputs", add );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", add );
}

void CAddNode::OnnxReshape()
{
	CTensorShape outputShape;
	TTensorType outputDataType = TT_ConstantTensor;

	for( int inputIndex = 0; inputIndex < 2; ++inputIndex ) {
		const CTensor& inputTensor = InputTensor( inputIndex );

		if( outputShape.IsEmpty() ) {
			inputTensor.GetShape().CopyTo( outputShape );
		} else {
			// NeoML doesn't support numpy-style tensor broadcasting...
			CheckNeoOnnxSupport( outputShape.Size() == inputTensor.GetShape().Size(),
				"tensor broadcasting", onnxNode );
			for( int i = 0; i < inputTensor.GetShape().Size(); ++i ) {
				CheckNeoOnnxSupport( outputShape[i] == inputTensor.GetShape()[i],
					"tensor broadcasting", onnxNode );
			}
		}

		if( inputTensor.GetType() == TT_DataTensor ) {
			outputDataType = TT_DataTensor;
		}
	}

	CPtr<CDnnBlob> outputBlob( nullptr );
	if( outputDataType == TT_ConstantTensor ) {
		// Precalculating the result.
		outputBlob = InputTensor( 0 ).GetData()->GetCopy();
		outputBlob->Add( InputTensor( 1 ).GetData() );
	}
	
	outputData.Add( CTensor( outputDataType, outputShape, outputBlob ) );
}

void CAddNode::MarkTensorDims()
{
	if( !InputTensor( 0 ).GetTensorDim().IsEmpty() ) {
		CheckNeoOnnxInternal( outputData[0].SetTensorDim( InputTensor( 0 ).GetTensorDim() ),
			"marking output dimensions failed", onnxNode );
	}

	if( !outputData[0].GetTensorDim().IsEmpty() ) {
		CheckNeoOnnxInternal( InputTensor( 0 ).SetTensorDim( outputData[0].GetTensorDim() ),
			"marking input dimensions failed", onnxNode );
	}
}

void CAddNode::AddLayers( CDnn& dnn )
{
	IMathEngine& mathEngine = dnn.GetMathEngine();

	CPtr<CEltwiseSumLayer> addLayer = new CEltwiseSumLayer( mathEngine );
	addLayer->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	addLayer->Connect( 0, InputLayer( 0 ), InputLayerIndex( 0 ) );
	addLayer->Connect( 1, InputLayer( 1 ), InputLayerIndex( 1 ) );

	dnn.AddLayer( *addLayer );

	outputInfo.Add( COutputInfo( addLayer, 0 ) );
}

} // namespace NeoOnnx
