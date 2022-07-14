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

#include "common.h"
#pragma hdrstop

#include "LayerUtils.h"

namespace NeoOnnx {

CPtr<CBaseSplitLayer> CreateSplitLayer( IMathEngine& mathEngine, TBlobDim dim )
{
	switch( dim ) {
		case BD_BatchLength:
			return new CSplitBatchLengthLayer( mathEngine );
		case BD_BatchWidth:
			return new CSplitBatchWidthLayer( mathEngine );
		case BD_ListSize:
			return new CSplitListSizeLayer( mathEngine );
		case BD_Height:
			return new CSplitHeightLayer( mathEngine );
		case BD_Width:
			return new CSplitWidthLayer( mathEngine );
		case BD_Depth:
			return new CSplitDepthLayer( mathEngine );
		case BD_Channels:
			return new CSplitChannelsLayer( mathEngine );
		default:
			NeoAssert( false );
	}
	return nullptr;
}

} // namespace NeoOnnx
