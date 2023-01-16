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
#include <NeoML/Dnn/Layers/Onnx/OnnxLayerBase.h>

namespace NeoML {

static const int OnnxReshapeLayerVersion = 0;

void COnnxReshapeLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxReshapeLayerVersion );
	CBaseLayer::Serialize( archive );
	inputLayout.Serialize( archive );
	outputLayout.Serialize( archive );
}

void COnnxReshapeLayer::Reshape()
{
	CheckArchitecture( GetInputCount() == 2, GetPath(), "Layer must have 2 inputs" );
	CheckArchitecture( GetOutputCount() == 1, GetPath(), "Layer must have 1 output" );
	const COnnxLayerBase* shapeProvider = dynamic_cast<const COnnxLayerBase*>( GetInputLayer( 1 ) );
	CheckArchitecture( shapeProvider != nullptr, GetPath(), "Second input must contain shape" );
	CheckArchitecture( shapeProvider->GetOutputShapeBlobs().IsValidIndex( GetInputOutputNumber( 1 ) ), GetPath(),
		"Wrong input number" );
	CPtr<CDnnBlob> newShapeBlob = shapeProvider->GetOutputShapeBlobs()[GetInputOutputNumber( 1 )];
	CheckArchitecture( newShapeBlob->GetDataSize() == outputLayout.Size(), GetPath(), "Dimension number mismatch" );
	CheckArchitecture( newShapeBlob->GetDataType() == CT_Int, GetPath(), "Non-integer shape" );

	int remIndex = NotFound;
	int remSize = inputDescs[0].BlobSize();
	outputDescs[0] = CBlobDesc( inputDescs[0].GetDataType() );
	CDnnBlobBuffer<int> newShape( *newShapeBlob, TDnnBlobBufferAccess::Read );
	for( int dimIndex = 0; dimIndex < outputLayout.Size(); ++dimIndex ) {
		if( newShape[dimIndex] == -1 ) {
			CheckArchitecture( remIndex == NotFound, GetPath(), "Two remainders" );
			remIndex = dimIndex;
		} else if( newShape[dimIndex] == 0 ) {
			CheckArchitecture( dimIndex < inputLayout.Size(), GetPath(),
				"Attempt to save the dimension of missing input axis" );
			outputDescs[0].SetDimSize( outputLayout[dimIndex],
				inputDescs[0].DimSize( inputLayout[dimIndex] ) );
		} else {
			CheckArchitecture( newShape[dimIndex] > 0, GetPath(), "Negative axis size");
			outputDescs[0].SetDimSize( outputLayout[dimIndex], newShape[dimIndex] );
		}
		remSize /= outputDescs[0].DimSize( outputLayout[dimIndex] );
	}

	if( remIndex != NotFound ) {
		CheckArchitecture( remSize > 0, GetPath(), "Output remainder isn't positive" );
		outputDescs[0].SetDimSize( outputLayout[remIndex], remSize );
		remSize = 1;
	}

	CheckArchitecture( remSize == 1, GetPath(), "Reshape didn't cover all of the data" );
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
