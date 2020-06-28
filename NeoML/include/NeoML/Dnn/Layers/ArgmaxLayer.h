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

// Finds the maximum element along the given dimension
class NEOML_API CArgmaxLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CArgmaxLayer )
public:
	explicit CArgmaxLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// The dimension along which the maximum is to be found
	void SetDimension(TBlobDim d);
	TBlobDim GetDimension() const { return dimension; }

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	TBlobDim dimension;
};

} // namespace NeoML
