/* Copyright © 2017-2023 ABBYY

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

#include <NeoML/Dnn/Layers/Onnx/OnnxLayerBase.h>

namespace NeoML {

static const int OnnxLayerBaseVersion = 0;

void COnnxLayerBase::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxLayerBaseVersion );
	CBaseLayer::Serialize( archive );
}

void COnnxLayerBase::Reshape()
{
	// Fill the outputs with the blobs consisting of 1 integer
	// It's done in order to avoid nullptr dereferencing problems
	// If layer wants to return an usual blob it will override these outputDescs during CalculateShapes
	for( CBlobDesc& outputDesc : outputDescs ) {
		outputDesc.SetDataType( CT_Int );
		// Change the BD_Channels dimension in order to enforce Reshape call of any layers connected to this one
		// When not needed this method won't be called
		outputDesc.SetDimSize( BD_Channels, outputDesc.Channels() == 1 ? 2 : 1 );
	}

	// Transfer shape-blobs between layers
	inputShapeBlobs.SetSize( GetInputCount() );
	for( int inputIndex = 0; inputIndex < GetInputCount(); ++inputIndex ) {
		COnnxLayerBase* onnxLayer = dynamic_cast<COnnxLayerBase*>( GetInputLayer( inputIndex ) );
		if( onnxLayer == nullptr ) {
			inputShapeBlobs[inputIndex] = nullptr;
		} else {
			inputShapeBlobs[inputIndex] = onnxLayer->outputShapeBlobs[GetInputOutputNumber( inputIndex )];
		}
	}

	// Allocate outputShapeBlobs
	outputShapeBlobs.Empty();
	outputShapeBlobs.Add( nullptr, GetOutputCount() );

	CalculateShapes();
}

} // namespace NeoML
