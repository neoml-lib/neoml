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

#pragma once

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

// This layer transforms each pixel of multi-channel images into blockSize x blockSize blocks of less channels.
// As a result the height and width of the images is multiplied by blockSize.
class NEOML_API CSpaceToDepthLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CSpaceToDepthLayer )
public:
	explicit CSpaceToDepthLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	int GetBlockSize() const;
	void SetBlockSize( int blockSize );

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	// Specifies the factor for reducing image size
	int blockSize;
};

NEOML_API CLayerWrapper<CSpaceToDepthLayer> SpaceToDepth( int blockSize );

} // namespace NeoML
