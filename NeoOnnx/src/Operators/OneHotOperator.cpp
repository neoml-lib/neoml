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

#include "OneHotOperator.h"
#include "NeoOnnxCheck.h"

#include <NeoML/Dnn/Layers/Onnx/OnnxOneHotLayer.h>

namespace NeoOnnx {

COneHotOperator::COneHotOperator( const onnx::NodeProto& oneHot, int opsetVersion ) :
	CLayerOperator( oneHot, opsetVersion )
{
	// v9 - original
	// v11 - negative axes supported
	CheckOnnxProtocol( InputCount() == 3, "operator must have 3 inputs", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void COneHotOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNoNullInputs( inputs );
	CheckNeoOnnxSupport( inputs[I_Depth]->Type() != TTensorType::User, "user-provided depth", *this );

	CPtr<COnnxOneHotLayer> oneHotLayer = new COnnxOneHotLayer( dnn.GetMathEngine() );
	oneHotLayer->SetName( Name() );
	CPtr<const CShapeTensor> depthTensor = AsShapeTensor( *inputs[I_Depth], Name() + "_depth", dnn );
	oneHotLayer->Connect( 1, *depthTensor->Layer(), depthTensor->OutputIndex() );
	CPtr<const CShapeTensor> valuesTensor = AsShapeTensor( *inputs[I_Values], Name() + "_values", dnn );
	oneHotLayer->Connect( 2, *valuesTensor->Layer(), valuesTensor->OutputIndex() );
	dnn.AddLayer( *oneHotLayer );

	CPtr<const CTensorBase> baseIndices = prepareIndices( *inputs[I_Indices] );
	const int axis = getAxis( baseIndices->DimCount() );
	CTensorLayout outputLayout = baseIndices->Layout();
	outputLayout.InsertAt( BD_Channels, axis );

	if( baseIndices->Type() != TTensorType::User && inputs[1]->Type() == TTensorType::Data ) {
		CPtr<const CShapeTensor> indices = AsShapeTensor( *baseIndices, Name() + "_data", dnn );
		oneHotLayer->Connect( 0, *indices->Layer(), indices->OutputIndex() );

		// Extract depth value
		int depth = 0;
		const CDnnBlob* depthBlob = dynamic_cast<const CDataTensor*>( inputs[1].Ptr() )->Data();
		CheckOnnxProtocol( depthBlob->GetDataSize() == 1, "size of depth isn't equal to 1", *this );
		if( depthBlob->GetDataType() == CT_Float ) {
			depth = static_cast<int>( depthBlob->GetData().GetValue() );
		} else {
			depth = depthBlob->GetData<int>().GetValue();
		}
		CheckOnnxProtocol( depth > 0, "non-positive depth", *this );

		CTensorShape outputShape;
		indices->Shape().CopyTo( outputShape );
		outputShape.InsertAt( depth, axis );

		outputs.Add( new CShapeTensor( outputLayout, outputShape, CLayerOutput( oneHotLayer, 0 ) ) );
	} else {
		CPtr<const CUserTensor> indices = AsUserTensor( *baseIndices, Name() + "_data", dnn );
		oneHotLayer->Connect( 0, *indices->Layer(), indices->OutputIndex() );
		outputs.Add( new CUserTensor( outputLayout, CLayerOutput( oneHotLayer, 0 ) ) );
	}
}

// Converts indices into layout, compatible with NeoOnnx
CPtr<const CTensorBase> COneHotOperator::prepareIndices( const CTensorBase& indicesInput ) const
{
	// COnnxOneHotLayer always puts one-hot vector dimension into BD_Channels
	// That's why indices must not use BD_Channels prior to this layer

	// Try to find the latest of dimensions not used in layout
	// Because even if BD_Channel was used, it's more effective to replace it with closest dim available
	// (it reduces chances of additional Transpose operations during conversion)
	TBlobDim unusedDim = BD_Count;
	for( int dim = static_cast<int>( BD_Channels ); dim >= static_cast<int>( BD_BatchLength ); --dim ) {
		if( indicesInput.Layout().Find( static_cast<TBlobDim>( dim ) ) == NotFound ) {
			unusedDim = static_cast<TBlobDim>( dim );
			break;
		}
	}

	if( unusedDim == BD_Channels ) {
		// BD_Channels isn't used by indices already
		// No conversion needed
		return &indicesInput;
	}

	// Convert tensor to layout, which doesn't use BD_Channels
	CTensorLayout newLayout = indicesInput.Layout();
	newLayout.ReplaceAt( unusedDim, newLayout.Find( BD_Channels ) );
	return ConvertTensor( indicesInput, newLayout );
}

// Returns non-negative axis index, which is used for one-hot vector
int COneHotOperator::getAxis( const int indicesDimCount ) const
{
	int axis = -1;
	GetAttribute( "axis", axis );
	if( axis < 0 ) {
		axis += indicesDimCount + 1;
	}
	return axis;
}

} // namespace NeoOnnx
