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

// This layer transforms multi-channel images into smaller images with more channels 
// (in effect, a k*n x k*m image is transformed into k*k images of n x m size).
// See also https://pjreddie.com/darknet/yolo/ where this transformation is used.
class NEOML_API CReorgLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CReorgLayer )
public:
	explicit CReorgLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	int GetStride() const;
	void SetStride( int stride );

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	// Specifies the factor for reducing image size
	int stride;
};

NEOML_API CLayerWrapper<CReorgLayer> Reorg();

} // namespace NeoML
