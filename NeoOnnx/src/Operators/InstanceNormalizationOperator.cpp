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

#include "../common.h"
#pragma hdrstop

#include "onnx.pb.h"

#include "InstanceNormalizationOperator.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

namespace NeoOnnx {

// Validator for InstanceNorm operator
class CInstanceNormLayoutValidator : public ITensorLayoutValidator {
	bool operator()( const CTensorLayout& layout ) const override;
};

bool CInstanceNormLayoutValidator::operator()( const CTensorLayout& layout ) const
{
	// In compatible layout first 2 dims must be batch dims and the rest must be object dims
	const int batchDims = 2;
	for( int dimIndex = 0; dimIndex < layout.Size(); ++dimIndex ) {
		if( ( dimIndex < batchDims && layout[dimIndex] >= BD_Height )
			|| ( dimIndex >= batchDims && layout[dimIndex] < BD_Height ) )
		{
			return false;
		}
	}
	return true;
}

// Applies normalization to the InstanceNormalization input
static CPtr<const CUserTensor> applyNormalization( const CUserTensor& input, float eps, const CString& layerName, CDnn& dnn )
{
	CPtr<const CUserTensor> currInput = ConvertTensor( input, CInstanceNormLayoutValidator() );
	CPtr<CObjectNormalizationLayer> objNormLayer = new CObjectNormalizationLayer( dnn.GetMathEngine() );
	objNormLayer->SetName( layerName );
	objNormLayer->SetEpsilon( eps );
	objNormLayer->Connect( 0, *currInput->Layer(), currInput->OutputIndex() );
	dnn.AddLayer( *objNormLayer );
	return new CUserTensor( currInput->Layout(), CLayerOutput( objNormLayer.Ptr(), 0 ) );
}

// Applies scale and bias to the normalized InstanceNormalization input
static CPtr<const CUserTensor> applyScaleAndBias( const CUserTensor& input, const CDataTensor& scale,
	const CDataTensor& bias, const CString& layerName, CDnn& dnn )
{
	IMathEngine& mathEngine = dnn.GetMathEngine();
	CPtr<const CUserTensor> currInput = ConvertTensor( input, CBatchNormLayoutValidator() );

	CPtr<CBatchNormalizationLayer> batchNormLayer = new CBatchNormalizationLayer( mathEngine );
	batchNormLayer->SetName( layerName );
	batchNormLayer->SetChannelBased( true );

	const int channels = scale.Data()->GetDataSize();
	CPtr<CDnnBlob> finalParams = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, 1, 2, channels );
	mathEngine.VectorCopy( finalParams->GetObjectData( 0 ), scale.Data()->GetData(), channels );
	mathEngine.VectorCopy( finalParams->GetObjectData( 1 ), bias.Data()->GetData(), channels );
	batchNormLayer->SetFinalParams( finalParams );

	batchNormLayer->Connect( 0, *currInput->Layer(), currInput->OutputIndex() );
	dnn.AddLayer( *batchNormLayer );
	return new CUserTensor( currInput->Layout(), CLayerOutput( batchNormLayer.Ptr(), 0 ) );
}

// --------------------------------------------------------------------------------------------------------------------

CInstanceNormalizationOperator::CInstanceNormalizationOperator( const onnx::NodeProto& instanceNormalization, int opsetVersion ) :
	CLayerOperator( instanceNormalization, opsetVersion ),
	eps( 1e-5f )
{
	// v1 - original
	// v6 - legacy optimization attribute removed
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 3, "operator must have 3 inputs", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );

	GetAttribute( "epsilon", eps );
}

void CInstanceNormalizationOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNoNullInputs( inputs );
	CheckNoShapeInputs( inputs );

	CheckNeoOnnxSupport( inputs[1]->Type() == TTensorType::Data, "User-provided scale", *this );
	CheckNeoOnnxSupport( inputs[2]->Type() == TTensorType::Data, "User-provided B", *this );
	CheckNeoOnnxSupport( inputs[0]->DimCount() < 6, "6+ dimensional input", *this );

	CPtr<const CUserTensor> currInput = AsUserTensor( *inputs[0], Name() + "_Source", dnn );

	// Step 1: applying normalization (x - mean) / sqrt(var + eps)
	// It's emulated via ObjectNormalizationLayer
	// Because of the fact that CObjectNormalizationLayer normalizes whole object (including channels)
	// we have to:
	//    1. Move ONNX channels to the batch
	//    2. Apply scale and bias somewhere else
	currInput = applyNormalization( *currInput, eps, Name() + "_ObjectNormalization", dnn );

	// Step 2: apply scale and bias
	// It's applied by using CBatchNormalizationLayer with final params filled with data from scale and B
	const CDataTensor& scale = dynamic_cast<const CDataTensor&>( *inputs[1] );
	const CDataTensor& bias = dynamic_cast<const CDataTensor&>( *inputs[2] );
	currInput = applyScaleAndBias( *currInput, scale, bias, Name() + "_ScaleAndBias", dnn );

	outputs.Add( currInput.Ptr() );
}

} // namespce NeoOnnx

