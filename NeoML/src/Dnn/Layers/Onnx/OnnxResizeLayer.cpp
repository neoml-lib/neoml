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

#include <NeoML/Dnn/Layers/Onnx/OnnxResizeLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxLayerBase.h>

namespace NeoML {

static const int OnnxResizeLayerVersion = 0;

void COnnxResizeLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxResizeLayerVersion );
	CInterpolationLayer::Serialize( archive );
	tensorLayout.Serialize( archive );
}

void COnnxResizeLayer::Reshape()
{
	CheckArchitecture( GetInputCount() == 2, GetPath(), "Layer must have 2 inputs" );
	CheckArchitecture( GetOutputCount() == 1, GetPath(), "Layer must have 1 output" );

	const COnnxLayerBase* secondInputLayer = dynamic_cast<const COnnxLayerBase*>( GetInputLayer( 1 ) );
	CheckArchitecture( secondInputLayer != nullptr, GetPath(), "Second input must be an Onnx layer" );
	CheckArchitecture( secondInputLayer->GetOutputShapeBlobs().IsValidIndex( GetInputOutputNumber( 1 ) ),
		GetPath(), "Wrong input number" );
	CPtr<CDnnBlob> newShapeBlob = secondInputLayer->GetOutputShapeBlobs()[GetInputOutputNumber( 1 )];
	CheckArchitecture( newShapeBlob != nullptr, GetPath(), "Second input blob missing" );
	CheckArchitecture( newShapeBlob->GetDataSize() == tensorLayout.Size(), GetPath(), "Dimension number mismatch" );

	if( newShapeBlob->GetDataType() == CT_Int ) {
		// Resize mode
		CDnnBlobBuffer<int> buff( *newShapeBlob, TDnnBlobBufferAccess::Read );
		for( int i = 0; i < buff.Size(); ++i ) {
			SetRule( tensorLayout[i], CRule::Resize( buff[i] ) );
		}
	} else {
		// Scales mode
		CDnnBlobBuffer<float> buff( *newShapeBlob, TDnnBlobBufferAccess::Read );
		for( int i = 0; i < buff.Size(); ++i ) {
			SetRule( tensorLayout[i], CRule::Scale( buff[i] ) );
		}
	}

	CInterpolationLayer::Reshape();
}

} // namespace NeoML
