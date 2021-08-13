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

#include "onnx.pb.h"

#include "BatchNormalizationOperator.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

namespace NeoOnnx {

// Converts layer input to the layout, supported by batch normalization layer
static CPtr<const CUserTensor> convertInput( const CUserTensor& input )
{
	const CTensorLayout& inputLayout = input.Layout();
	bool needConversion = false;
	for( int dimIndex = 0; dimIndex < inputLayout.Size(); ++dimIndex ) {
		// First dimension must be batch (BD_BatchLength, BD_BatchWidth or BD_ListSize)
		// Second dimension must be channels
		// Other dimensions must be spatial (BD_Height, BD_Width or BD_Depth)
		if( ( dimIndex < 1 && inputLayout[dimIndex] >= BD_Height )
			|| ( dimIndex == 1 && inputLayout[dimIndex] != BD_Channels )
			|| ( dimIndex > 1 && ( inputLayout[dimIndex] < BD_Height || inputLayout[dimIndex] == BD_Channels ) ) )
		{
			needConversion = true;
			break;
		}
	}

	if( !needConversion ) {
		// input's layout is compatible with batch normalzation
		// No conversion needed
		return &input;
	}

	CTensorLayout outputLayout( { BD_BatchWidth, BD_Channels, BD_Height, BD_Width, BD_Depth } );
	outputLayout.SetSize( input.DimCount() );

	return dynamic_cast<const CUserTensor*>( ConvertTensor( input, outputLayout ).Ptr() );
}

// Calculates NeoML::CBatchNormalizationLayer's final params blob from onnx operator's inputs
static CPtr<CDnnBlob> calculateFinalParams( float eps, const CTensorArray& inputs )
{
	const CDnnBlob* scale = dynamic_cast<const CDataTensor&>( *inputs[1] ).Data();
	NeoAssert( scale != nullptr );
	const CDnnBlob* bias = dynamic_cast<const CDataTensor&>( *inputs[2] ).Data();
	NeoAssert( bias != nullptr );
	const CDnnBlob* mean = dynamic_cast<const CDataTensor&>( *inputs[3] ).Data();
	NeoAssert( mean != nullptr );
	const CDnnBlob* var = dynamic_cast<const CDataTensor&>( *inputs[4] ).Data();
	NeoAssert( var != nullptr );

	const int channels = scale->GetDataSize();

	IMathEngine& mathEngine = scale->GetMathEngine();

	// Calculate final params
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

// --------------------------------------------------------------------------------------------------------------------

CBatchNormalizationOperator::CBatchNormalizationOperator( const onnx::NodeProto& batchNormalization, int opsetVersion ) :
	CLayerOperator( batchNormalization, opsetVersion ),
	eps( 1e-5f )
{
	// v1 - original
	// v6 - legacy optimization attributes are removed
	// v7 - 'is_test' attribute is removed
	// v9 - 'spatial' attribute is removed
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 5 || InputCount() == 6, "operator must have 5 or 6 inputs", *this );
	CheckNeoOnnxSupport( OutputCount() == 1, "operator must have 1 output", *this );

	GetAttribute( "epsilon", eps );

	if( OpsetVersion < 7 ) {
		int isTest = 0;
		GetAttribute( "is_test", isTest );
		CheckNeoOnnxSupport( isTest != 0, "training batch normalization is not supported", *this );
	}

	if( OpsetVersion < 9 ) {
		int spatial = 1;
		GetAttribute( "spatial", spatial );
		CheckNeoOnnxSupport( spatial != 0, "non-spatial batch norm", *this );
	}
}

void CBatchNormalizationOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckOnnxProtocol( inputs[0] != nullptr, "input can't be optional", *this );

	const int channels = inputs[0]->Shape()[1];
	// The number of required inputs of BatchNormalization operator
	const int batchNormReqInputCount = 5;
	for( int inputIndex = 1; inputIndex < batchNormReqInputCount; ++inputIndex ) {
		CheckOnnxProtocol( inputs[inputIndex] != nullptr, "input can't be optional", *this );
		CheckNeoOnnxSupport( inputs[inputIndex]->IsCalculated(), "non-constant weights", *this );
		CheckOnnxProtocol( inputs[inputIndex]->DimCount() == 1, "weights must be 1-dimensional", *this );
		CheckOnnxProtocol( inputs[inputIndex]->Shape()[0] == channels, "weights must have 'channels' length", *this );
	}

	CPtr<CBatchNormalizationLayer> bnLayer = new CBatchNormalizationLayer( dnn.GetMathEngine() );
	bnLayer->SetName( Name() );
	bnLayer->SetChannelBased( true );
	bnLayer->SetFinalParams( calculateFinalParams( eps, inputs ) );

	CPtr<const CUserTensor> userData = convertInput( *AsUserTensor( *inputs[0], Name() + "_Source", dnn ) );
	bnLayer->Connect( 0, *userData->Layer(), userData->OutputIndex() );
	dnn.AddLayer( *bnLayer );

	outputs.Add( new CUserTensor( userData->Shape(), userData->Layout(), CLayerOutput( bnLayer, 0 ) ) );
}

} // namespace NeoOnnx

