/* Copyright © 2017-2020 ABBYY Production LLC

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

#include <NeoML/Dnn/Layers/QualityControlLayer.h>

namespace NeoML {

CQualityControlLayer::CQualityControlLayer( IMathEngine& mathEngine, const char* name ) :
	CBaseLayer( mathEngine, name, false ),
	needReset( true )
{
}

void CQualityControlLayer::Reshape()
{
	CheckInputs();
	CheckArchitecture( inputDescs.Size() == 2, GetName(), "layer expects 2 inputs" );
	CheckArchitecture( inputDescs[0].ObjectCount() == inputDescs[1].ObjectCount(), GetName(),
		"Object count mismatch between inputs" );
	CheckArchitecture( inputDescs[0].ObjectSize() == inputDescs[1].ObjectSize(), GetName(),
		"Object size mismatch between inputs" );
	CheckArchitecture( !outputDescs.IsEmpty(), GetName(), "There is nothing connected to this layer's output" );
}

void CQualityControlLayer::BackwardOnce()
{
	// This layer is usually placed after some learnable layers.
	// In order to learn those layers are requiring for gradient from all succeeding layers (including this one).
	inputDiffBlobs[0]->Clear();
}

void CQualityControlLayer::RunOnce()
{
	if( IsResetNeeded() ) {
		OnReset();
	}
	RunOnceAfterReset();
}

static const int QualityControlLayerVersion = 2000;

void CQualityControlLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( QualityControlLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );
}

} // namespace NeoML
