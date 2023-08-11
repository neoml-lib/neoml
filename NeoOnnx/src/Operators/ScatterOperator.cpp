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

#include "ScatterOperator.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

class CScatterNDLayoutValidator : public ITensorLayoutValidator {
public:
	bool operator()( const CTensorLayout& layout ) const override
		{ return !IsTransposedLayout( layout ) && layout.Last() == BD_Channels; }
};

CScatterNDOperator::CScatterNDOperator( const onnx::NodeProto& scatterND, int opsetVersion ) :
	CLayerOperator( scatterND, opsetVersion )
{
	// v11 - original
	// v13 - support bfloat16
	// v16 - new reduction attribute
	CheckNeoOnnxSupport( OpsetVersion >= 11 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 3, "operator must have 3 inputs", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );

	if( OpsetVersion >= 16 ) {
		CString reduction = "none";
		GetAttribute( "reduction", reduction );
		CheckNeoOnnxSupport( reduction == "none", "non-default reduction", *this );
	}
}

void CScatterNDOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNoNullInputs( inputs );
	CheckNoShapeInputs( inputs );

	CPtr<const CUserTensor> dataTensor = AsUserTensor(
		*ConvertTensor( *inputs[0], CTensorLayout::IOLayout( inputs[0]->DimCount() ) ), Name() + "_Data", dnn );
	CPtr<const CUserTensor> indicesTensor = AsUserTensor( *ConvertTensor( *inputs[1], CScatterNDLayoutValidator() ),
		Name() + "_Indices", dnn );
	CPtr<const CUserTensor> updatesTensor = AsUserTensor( *ConvertTensor( *inputs[2], COnnxTensorLayoutValidator() ),
		Name() + "_Indices", dnn );

	CPtr<CScatterNDLayer> scatterND = new CScatterNDLayer( dnn.GetMathEngine() );
	scatterND->SetName( Name() );
	scatterND->Connect( CScatterNDLayer::I_Data, *dataTensor->Layer(), dataTensor->OutputIndex() );
	scatterND->Connect( CScatterNDLayer::I_Indices, *indicesTensor->Layer(), indicesTensor->OutputIndex() );
	scatterND->Connect( CScatterNDLayer::I_Updates, *updatesTensor->Layer(), updatesTensor->OutputIndex() );
	dnn.AddLayer( *scatterND );
	outputs.Add( new CUserTensor( dataTensor->Layout(), CLayerOutput( scatterND.Ptr(), 0 ) ) );
}

} // namespace NeoOnnx
