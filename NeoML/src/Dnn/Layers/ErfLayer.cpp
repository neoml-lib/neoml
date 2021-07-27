/* Copyright © 2017-2021 ABBYY Production LLC

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

#include <NeoML/Dnn/Layers/ErfLayer.h>

#include <cmath>

namespace NeoML {

CErfLayer::CErfLayer( IMathEngine& mathEngine ) :
	CBaseInPlaceLayer( mathEngine, "CErfLayer", false )
{
}

static const int ErfLayerVersion = 0;

void CErfLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( ErfLayerVersion );
	CBaseInPlaceLayer::Serialize( archive );
}

void CErfLayer::RunOnce()
{
	for( int inputIndex = 0; inputIndex < inputBlobs.Size(); ++inputIndex ) {
		const int dataSize = inputBlobs[inputIndex]->GetDataSize();
		const float* inBuffer = inputBlobs[inputIndex]->GetBuffer<float>( 0, dataSize, true );
		float* outBuffer = outputBlobs[0]->GetBuffer<float>( 0, dataSize, false );
		for( int i = 0; i < dataSize; ++i ) {
			outBuffer[i] = std::erff( inBuffer[i] );
		}
		outputBlobs[inputIndex]->ReleaseBuffer( outBuffer, true );
		inputBlobs[inputIndex]->ReleaseBuffer( const_cast< float* >( inBuffer ), false );
	}
}

void CErfLayer::BackwardOnce()
{
	// TODO: add MathEngine impl
	NeoAssert( false );
}

} // namespace NeoML
