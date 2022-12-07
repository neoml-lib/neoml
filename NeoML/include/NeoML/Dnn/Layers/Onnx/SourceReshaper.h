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
#include <NeoML/Dnn/Layers/Onnx/BaseReshaper.h>

namespace NeoML {

// Class which provides shape tensor to CDnn
class NEOML_API CSourceReshaper : public CBaseReshaper {
public:
	explicit CSourceReshaper( IMathEngine& mathEngine ) : CBaseReshaper( mathEngine, "SourceReshaper" ) {}

	void Serialize( CArchive& archive ) override;

	// Shape tensor to be feeded to CDnn
	CShapeTensor& Tensor() { return tensor; }
	const CShapeTensor& Tensor() const { return tensor; }

protected:
	void CalculateShapes() override;

private:
	CShapeTensor tensor;
};

} // namespace NeoML
