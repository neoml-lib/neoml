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

// Matrix multiplication
// Has 2 inputs
// Each input contains BatchLength * BatchWidth * ListSize matrices of size (Height * Width * Depth) x Chanells
// Other dimensions must be equal to 1
// input[0].Channels must be equal to input[1].Height * input[1].Width * input[1].Depth
// The result matrix size is (input[0].Height * input[0].Width * input[0].Depth) x input[1].Channels
class NEOML_API CMatrixMultiplicationLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CMatrixMultiplicationLayer )
public:
	explicit CMatrixMultiplicationLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

protected:
	// CBaseLayer implementation
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
};

} // namespace NeoML
