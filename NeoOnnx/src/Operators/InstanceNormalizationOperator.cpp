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

// Checks whether the tensor layout is compatible with CObjectNormalizationLayer or not
static bool isObjectNormalizationCompatible( const CTensorLayout& layout )
{
	// In compatible layout first 2 dims must be batch dims
	// and the rest must be object dims
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
	CPtr<const CUserTensor> currInput = &input;
	if( !isObjectNormalizationCompatible( currInput->Layout() ) ) {
		CTensorLayout objNormLayout( { BD_BatchWidth, BD_ListSize, BD_Height, BD_Width, BD_Depth } );
		objNormLayout.SetSize( currInput->DimCount() );
		currInput = ConvertTensor( *currInput, objNormLayout );
	}
	CPtr<CObjectNormalizationLayer> objNormLayer = new CObjectNormalizationLayer( dnn.GetMathEngine() );
	objNormLayer->SetName( layerName );
	objNormLayer->SetEpsilon( eps );

	int spatialSize = 1;
	for( int dimIndex = 2; dimIndex < input.DimCount(); ++dimIndex ) {
		spatialSize *= input.Shape()[dimIndex];
	}
	CPtr<CDnnBlob> objNormParam = CDnnBlob::CreateVector( dnn.GetMathEngine(), CT_Float, spatialSize );
	objNormParam->Fill( 1.f );
	objNormLayer->SetScale( objNormParam );
	objNormParam->Clear();
	objNormLayer->SetBias( objNormParam );

	objNormLayer->Connect( 0, *currInput->Layer(), currInput->OutputIndex() );
	dnn.AddLayer( *objNormLayer );
	return new CUserTensor( currInput->Shape(), currInput->Layout(), CLayerOutput( objNormLayer.Ptr(), 0 ) );
}

// Applies scale and bias to the normalized InstanceNormalization input
static CPtr<const CUserTensor> applyScaleAndBias( const CUserTensor& input, const CDataTensor& scale,
	const CDataTensor& bias, const CString& layerName, CDnn& dnn )
{
	IMathEngine& mathEngine = dnn.GetMathEngine();
	CPtr<const CUserTensor> currInput = &input;
	// CObjectNormalization layout is 100% incompatible with the CBatchNormalization layout
	CTensorLayout batchNormLayout( { BD_BatchWidth, BD_Channels, BD_Height, BD_Width, BD_Depth } );
	batchNormLayout.SetSize( currInput->DimCount() );
	currInput = ConvertTensor( *currInput, batchNormLayout );

	CPtr<CBatchNormalizationLayer> batchNormLayer = new CBatchNormalizationLayer( mathEngine );
	batchNormLayer->SetName( layerName );
	batchNormLayer->SetChannelBased( true );

	const int channels = currInput->Shape()[1];
	CPtr<CDnnBlob> finalParams = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, 1, 2, channels );
	mathEngine.VectorCopy( finalParams->GetObjectData( 0 ), scale.Data()->GetData(), channels );
	mathEngine.VectorCopy( finalParams->GetObjectData( 1 ), bias.Data()->GetData(), channels );
	batchNormLayer->SetFinalParams( finalParams );

	batchNormLayer->Connect( 0, *currInput->Layer(), currInput->OutputIndex() );
	dnn.AddLayer( *batchNormLayer );
	return new CUserTensor( currInput->Shape(), currInput->Layout(), CLayerOutput( batchNormLayer.Ptr(), 0 ) );
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
	CheckOnnxProtocol( inputs[0] != nullptr, "input can't be optional", *this );
	CheckOnnxProtocol( inputs[1] != nullptr, "scale can't be optional", *this );
	CheckOnnxProtocol( inputs[2] != nullptr, "B can't be optional", *this );

	CheckNeoOnnxSupport( inputs[1]->IsCalculated(), "User-provided scale", *this );
	CheckNeoOnnxSupport( inputs[2]->IsCalculated(), "User-provided B", *this );
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

