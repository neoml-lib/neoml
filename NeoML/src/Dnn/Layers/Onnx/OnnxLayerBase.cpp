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

#include <NeoML/Dnn/Layers/Onnx/OnnxLayerBase.h>

namespace NeoML {

static const int OnnxLayerBaseVersion = 0;

void COnnxLayerBase::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxLayerBaseVersion );
	CBaseLayer::Serialize( archive );
}

bool COnnxLayerBase::HasShapeInputs() const
{
	for( int i = 0; i < inputShapeBlobs.Size(); ++i ) {
		if( inputShapeBlobs[i] != nullptr ) {
			return true;
		}
	}

	return false;
}

void COnnxLayerBase::Reshape()
{
	// Fill the outputs with the blobs consisting of 1 integer
	for( int outputIndex = 0; outputIndex < GetOutputCount(); ++outputIndex ) {
		outputDescs[outputIndex] = CBlobDesc( CT_Int );
	}

	inputShapeBlobs.SetSize( GetInputCount() );
	for( int inputIndex = 0; inputIndex < GetInputCount(); ++inputIndex ) {
		COnnxLayerBase* reshaper = dynamic_cast<COnnxLayerBase*>( GetInputLayer( inputIndex ) );
		if( reshaper == nullptr ) {
			inputShapeBlobs[inputIndex] = nullptr;
		} else {
			inputShapeBlobs[inputIndex] = reshaper->outputShapeBlobs[GetInputOutputNumber( inputIndex )];
		}
	}

	outputShapeBlobs.Empty();
	outputShapeBlobs.SetSize( GetOutputCount() );

	CalculateShapes();
}

} // namespace NeoML
