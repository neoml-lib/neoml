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

#include <NeoML/Dnn/Layers/BaseInPlaceLayer.h>

namespace NeoML {

void CBaseInPlaceLayer::Reshape()
{
	isInPlace = IsInPlaceProcessAvailable();
	inputDescs.CopyTo( outputDescs );

	OnReshaped();
}

void CBaseInPlaceLayer::AllocateOutputBlobs()
{
	if( !isInPlace ) {
		CBaseLayer::AllocateOutputBlobs();
		return;
	}

	if( !outputBlobs.IsEmpty() && outputBlobs[0] == 0 ) {
		inputBlobs.CopyTo( outputBlobs );
	}
}

static const int BaseInPlaceLayerVersion = 2000;

void CBaseInPlaceLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( BaseInPlaceLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );
}

} // namespace NeoML
