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

#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

// Adds "addition" to every object of the data
// Expects 2 inputs:
//	- "data": blob of any size (BatchLength x BatchWidth x ListSize x Height x Width x Depth x Channels)
//	- "addition": blob with a single object (1 x 1 x 1 x Height x Width x Depth x Channels)
// The only output contains the result blob of size (BatchLength x BatchWidth x ListSize x Height x Width x Depth x Channels) where
//	result[i] = data[i] + addition
// for each i in [0, BatchLength x BatchWidth x ListSize) and '+' means eltwise sum
class NEOML_API CAddToObjectLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CAddToObjectLayer )
public:
	explicit CAddToObjectLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;
protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
};

} // namespace NeoML
