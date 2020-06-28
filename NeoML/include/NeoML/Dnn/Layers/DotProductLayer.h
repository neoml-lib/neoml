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

// The layer that calculates dot product of the objects
// It has two input blobs with the float data, of the same dimensions:
//     BatchLength x BatchWidth x ListSize x Channel x Depth x Height x Width
// and one output of the dimensions
//     BatchLength x BatchWidth x ListSize x 1 x 1 x 1 x 1
// that contains the dot product
class NEOML_API CDotProductLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CDotProductLayer )
public:
	explicit CDotProductLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

protected:
	// CBaseLayer methods
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
};

} // namespace NeoML
