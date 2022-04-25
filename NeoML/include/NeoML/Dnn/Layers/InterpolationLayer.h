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

#pragma once

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

// Layer that multiplies some of the blob dimensions
// and fills the new element with approximated values based on its neighbors
// At the moment supports only linear interpolation
//
// For each of the dims outputShape[dim] = inputShape[dim] * GetScale(dim)
//
// By default each scale is equal to 1
class NEOML_API CInterpolationLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CInterpolationLayer )
public:
	explicit CInterpolationLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// Sets the multiplier for a given blob dimension
	// Must be >= 1
	void SetScale( TBlobDim dim, int scale );
	int GetScale( TBlobDim dim ) const;

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	CArray<int> scales;
};

} // namespace NeoML
