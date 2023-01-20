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
	CheckNeoOnnxSupport( inputs[1]->Type() != TTensorType::User, "user-provided depth", *this );
	CheckNeoOnnxSupport( inputs[2]->Type() == TTensorType::Data, "non-fixed values", *this );

	const CDnnBlob& valuesBlob = *dynamic_cast<const CDataTensor&>( *inputs[2] ).Data();
	CheckNeoOnnxSupport( valuesBlob.GetDataSize() == 2, "values must contain 2 elements", *this );
	if( valuesBlob.GetDataType() == CT_Float ) {
		CheckNeoOnnxSupport( valuesBlob.GetData().GetValueAt( 0 ) == 0.f, "off value must be 0", *this );
		CheckNeoOnnxSupport( valuesBlob.GetData().GetValueAt( 1 ) == 1.f, "on value must be 1", *this );
	} else {
		CheckNeoOnnxSupport( valuesBlob.GetData<int>().GetValueAt( 0 ) == 0, "off value must be 0", *this );
		CheckNeoOnnxSupport( valuesBlob.GetData<int>().GetValueAt( 1 ) == 1, "on value must be 1", *this );
	}

	CPtr<const CTensorBase> baseIndices = inputs[0];
	if( baseIndices->Layout().Find( BD_Channels ) != NotFound ) {
		baseIndices = ConvertTensor( *baseIndices, CTensorLayout::IOLayout( baseIndices->DimCount() ) );
	}
	CheckNeoOnnxSupport( baseIndices->DimCount() < 7, "OneHot with 7-dimensional input", *this );

	CPtr<COnnxOneHotLayer> oneHotLayer = new COnnxOneHotLayer( dnn.GetMathEngine() );
	oneHotLayer->SetName( Name() );
	CPtr<const CShapeTensor> depthTensor = AsShapeTensor( *inputs[1], Name() + "_depth", dnn );
	oneHotLayer->Connect( 1, *depthTensor->Layer(), depthTensor->OutputIndex() );
	dnn.AddLayer( *oneHotLayer );

	CTensorLayout outputLayout = baseIndices->Layout();
	int axis = -1;
	GetAttribute( "axis", axis );
	if( axis < 0 ) {
		axis += baseIndices->DimCount() + 1;
	}
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

} // namespace NeoOnnx
