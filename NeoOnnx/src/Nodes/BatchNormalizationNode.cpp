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

#include "BatchNormalizationNode.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CBatchNormalizationNode::CBatchNormalizationNode( const onnx::NodeProto& batchNormalization, int opsetVersion ) :
	COpNode( batchNormalization, opsetVersion ),
	eps( Attributes.GetOptionalFloat( "epsilon", 1e-5f ) )
{
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", batchNormalization );

	CheckNeoOnnxSupport( OpsetVersion > 6 || Attributes.GetOptionalInt( "is_test", 0 ) != 0,
		"training batch normalization is not supported", batchNormalization );

	CheckOnnxProtocol( InputCount() == 5 || InputCount() == 6, "node must have 5 or 6 inputs", OnnxNode );
	CheckNeoOnnxSupport( OutputCount() == 1, "node must have 1 output", OnnxNode );
	if( OpsetVersion < 9 ) {
		CheckNeoOnnxSupport( Attributes.GetOptionalInt( "spatial", 1 ) != 0,
			"non-spatial batch norm" );
	}
}

void CBatchNormalizationNode::AddLayers( const CObjectArray<const CTensorBase>& inputs,
	CObjectArray<const CTensorBase>& outputs, CDnn& dnn )
{
	CPtr<const CUserTensor> input = convertInput( dynamic_cast<const CUserTensor&>( *inputs[0] ) );
	CPtr<CBatchNormalizationLayer> bnLayer = new CBatchNormalizationLayer( dnn.GetMathEngine() );
	bnLayer->SetName( Name() );
	bnLayer->SetChannelBased( true );
	bnLayer->SetFinalParams( calculateFinalParams( input->Shape()[1], inputs ) );

	bnLayer->Connect( 0, *input->Layer(), input->OutputIndex() );
	dnn.AddLayer( *bnLayer );

	outputs[0] = new CUserTensor( input->Shape(), input->Layout(), CLayerOutput( bnLayer, 0 ) );
}

// Converts layer input if needed
CPtr<const CUserTensor> CBatchNormalizationNode::convertInput( const CUserTensor& input ) const
{
	if( input.Layout().DimType == DT_NeoML ) {
		const CDimOrder& inputDimOrder = input.Layout().OnnxOrder;
		bool needConversion = false;
		for( int dimIndex = 0; dimIndex < inputDimOrder.Size(); ++dimIndex ) {
			// First dimension must be batch (BD_BatchLength, BD_BatchWidth or BD_ListSize)
			// Second dimension must be channels
			// Other dimensions must be spatial (BD_Height, BD_Width or BD_Depth)
			if( ( dimIndex < 1 && inputDimOrder[dimIndex] >= BD_Height )
				|| ( dimIndex == 1 && inputDimOrder[dimIndex] != BD_Channels )
				|| ( dimIndex > 1 && ( inputDimOrder[dimIndex] < BD_Height || inputDimOrder[dimIndex] == BD_Channels ) ) )
			{
				needConversion = true;
				break;
			}
		}

		if( !needConversion ) {
			return &input;
		}
	}

	CDimOrder outputOrder( { BD_BatchWidth, BD_Channels, BD_Height, BD_Width, BD_Depth } );
	outputOrder.SetSize( input.Shape().Size() );

	return dynamic_cast<const CUserTensor*>( ConvertTensor( input, CTensorLayout( outputOrder ) ).Ptr() );
}

// Calculates final params blob based on onnx node's inputs
// This format can is compatible with NeoML::CBatchNormalizationLayer
CPtr<CDnnBlob> CBatchNormalizationNode::calculateFinalParams( int channels, const CObjectArray<const CTensorBase>& inputs )
{
	for( int inputIndex = 1; inputIndex < 5; ++inputIndex ) {
		CheckNeoOnnxSupport( inputs[inputIndex]->IsCalculated(), "non-constant weights", OnnxNode );
		CheckOnnxProtocol( inputs[inputIndex]->Shape().Size() == 1, "weights must be 1-dimensional", OnnxNode );
		CheckOnnxProtocol( inputs[inputIndex]->Shape()[0] == channels, "weights must have 'channels' length", OnnxNode );
	}

	const CDnnBlob* scale = dynamic_cast<const CDataTensor&>( *inputs[1] ).Data();
	const CDnnBlob* bias = dynamic_cast<const CDataTensor&>( *inputs[2] ).Data();
	const CDnnBlob* mean = dynamic_cast<const CDataTensor&>( *inputs[3] ).Data();
	const CDnnBlob* var = dynamic_cast<const CDataTensor&>( *inputs[4] ).Data();

	IMathEngine& mathEngine = scale->GetMathEngine();

	// Calculating final params
	CPtr<CDnnBlob> finalParams = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, 1, 2, channels );

	CFloatHandleStackVar epsVar( mathEngine );
	epsVar.SetValue( eps );
	CFloatHandle gamma = finalParams->GetObjectData( 0 );
	mathEngine.VectorAddValue( var->GetData(), gamma, channels, epsVar );
	mathEngine.VectorSqrt( gamma, gamma, channels );
	mathEngine.VectorEltwiseDivide( scale->GetData(), gamma, gamma, channels );

	CFloatHandle beta = finalParams->GetObjectData( 1 );
	mathEngine.VectorEltwiseMultiply( mean->GetData(), gamma, beta, channels );
	mathEngine.VectorSub( bias->GetData(), beta, beta, channels );

	return finalParams;
}

} // namespace NeoOnnx
