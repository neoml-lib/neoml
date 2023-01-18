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
#include <NeoML/Dnn/Layers/Onnx/OnnxLayerBase.h>

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
	const COnnxLayerBase* shapeProvider = dynamic_cast<const COnnxLayerBase*>( GetInputLayer( 1 ) );
	CheckArchitecture( shapeProvider != nullptr, GetPath(), "Second input must contain shape" );
	CheckArchitecture( shapeProvider->GetOutputShapeBlobs().IsValidIndex( GetInputOutputNumber( 1 ) ), GetPath(),
		"Wrong input number" );
	CPtr<CDnnBlob> newShapeBlob = shapeProvider->GetOutputShapeBlobs()[GetInputOutputNumber( 1 )];
	CheckArchitecture( newShapeBlob->GetDataSize() <= tensorLayout.Size(), GetPath(), "Dimension number mismatch" );
	CheckArchitecture( newShapeBlob->GetDataType() == CT_Int, GetPath(), "Non-integer shape" );

	const int preservedDims = tensorLayout.Size() - newShapeBlob->GetDataSize();

	outputDescs[0] = inputDescs[0];

	CDnnBlobBuffer<int> newShape( *newShapeBlob, TDnnBlobBufferAccess::Read );
	for( int dimIndex = 0; dimIndex < newShape.Size(); ++dimIndex ) {
		CheckArchitecture( newShape[dimIndex] > 0, GetPath(), "Negative axis size" );
		const TBlobDim& dim = tensorLayout[preservedDims + dimIndex];
		outputDescs[0].SetDimSize( dim, newShape[dimIndex] == 1 ? inputDescs[0].DimSize( dim ) : newShape[dimIndex] );
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
