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

// CTransposeLayer implements a layer that transposes the specified dimensions of a blob
class NEOML_API CTransposeLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CTransposeLayer )
public:
	explicit CTransposeLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// Retrieves or sets the names of the dimensions to be transposed
	void SetTransposedDimensions(TBlobDim d1, TBlobDim d2);
	void GetTransposedDimensions(TBlobDim& d1, TBlobDim &d2) const;

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	// The dimensions to be transposed
	TBlobDim d1;
	TBlobDim d2;
};

inline void CTransposeLayer::SetTransposedDimensions(TBlobDim _d1, TBlobDim _d2)
{
	d1 = _d1;
	d2 = _d2;
}

inline void CTransposeLayer::GetTransposedDimensions(TBlobDim& _d1, TBlobDim& _d2) const
{
	_d1 = d1;
	_d2 = d2;
}

} // namespace NeoML
