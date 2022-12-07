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

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/Layers/Onnx/OnnxReshapeLayer.h>
#include <NeoML/Dnn/Layers/Onnx/BaseReshaper.h>

namespace NeoML {

static const int OnnxReshapeLayerVersion = 0;

void COnnxReshapeLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxReshapeLayerVersion );
	CBaseLayer::Serialize( archive );
	tensorLayout.Serialize( archive );
}

void COnnxReshapeLayer::Reshape()
{
	CheckArchitecture( GetInputCount() == 2, GetPath(), "Layer must have 2 inputs" );
	CheckArchitecture( GetOutputCount() == 1, GetPath(), "Layer must have 1 output" );
	const CBaseReshaper* shapeProvider = dynamic_cast<const CBaseReshaper*>( GetInputLayer( 1 ) );
	CheckArchitecture( shapeProvider != nullptr, GetPath(), "Second input must contain shape" );
	CheckArchitecture( shapeProvider->GetoutputShapeTensors().IsValidIndex( GetInputOutputNumber( 1 ) ), GetPath(),
		"Wrong input number" );
	const CShapeTensor& newShape = shapeProvider->GetoutputShapeTensors()[GetInputOutputNumber( 1 )];
	CheckArchitecture( newShape.ElementCount() == tensorLayout.Size(), GetPath(), "Dimension number mismatch" );

	outputDescs[0] = CBlobDesc( inputDescs[0].GetDataType() );
	for( int dimIndex = 0; dimIndex < tensorLayout.Size(); ++dimIndex ) {
		outputDescs[0].SetDimSize( tensorLayout[dimIndex], newShape[dimIndex] );
	}
}

void COnnxReshapeLayer::RunOnce()
{
	if( inputBlobs[0]->GetDataType() == CT_Float ) {
		MathEngine().VectorCopy( outputBlobs[0]->GetData(), inputBlobs[0]->GetData(), inputBlobs[0]->GetDataSize() );
	} else {
		MathEngine().VectorCopy( outputBlobs[0]->GetData<int>(), inputBlobs[0]->GetData<int>(),
			inputBlobs[0]->GetDataSize() );
	}
}

} // namespace NeoML
