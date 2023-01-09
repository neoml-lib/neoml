/* Copyright © 2017-2023 ABBYY Production LLC

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

#include <NeoML/Dnn/Layers/Onnx/OnnxExpandLayer.h>
#include <NeoML/Dnn/Layers/Onnx/BaseReshaper.h>

namespace NeoML {

static const int OnnxExpandLayerVersion = 0;

void COnnxExpandLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxExpandLayerVersion );
	CBaseLayer::Serialize( archive );
	tensorLayout.Serialize( archive );
}

void COnnxExpandLayer::Reshape()
{
	CheckArchitecture( GetInputCount() == 2, GetPath(), "Layer must have 2 inputs" );
	CheckArchitecture( GetOutputCount() == 1, GetPath(), "Layer must have 1 output" );
	const CBaseReshaper* shapeProvider = dynamic_cast<const CBaseReshaper*>( GetInputLayer( 1 ) );
	CheckArchitecture( shapeProvider != nullptr, GetPath(), "Second input must contain shape" );
	CheckArchitecture( shapeProvider->GetoutputShapeTensors().IsValidIndex( GetInputOutputNumber( 1 ) ), GetPath(),
		"Wrong input number" );
	const CShapeTensor& newShape = shapeProvider->GetoutputShapeTensors()[GetInputOutputNumber( 1 )];
	CheckArchitecture( newShape.ElementCount() <= tensorLayout.Size(), GetPath(), "Dimension number mismatch" );

	const int preservedDims = tensorLayout.Size() - newShape.ElementCount();

	outputDescs[0] = inputDescs[0];
	for( int dimIndex = 0; dimIndex < newShape.ElementCount(); ++dimIndex ) {
		CheckArchitecture( newShape[dimIndex] > 0, GetPath(), "Negative axis size" );
		outputDescs[0].SetDimSize( tensorLayout[preservedDims + dimIndex], newShape[dimIndex] );
	}
}

void COnnxExpandLayer::RunOnce()
{
	if( inputBlobs[0]->GetDataType() == CT_Float ) {
		MathEngine().BroadcastCopy( outputBlobs[0]->GetData(), inputBlobs[0]->GetData(), outputBlobs[0]->GetDesc(),
			inputBlobs[0]->GetDesc(), 1 );
	} else {
		MathEngine().BroadcastCopy( outputBlobs[0]->GetData<int>(), inputBlobs[0]->GetData<int>(), outputBlobs[0]->GetDesc(),
			inputBlobs[0]->GetDesc(), 1 );
	}
}

} // namespace NeoML
