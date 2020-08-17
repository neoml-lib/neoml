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

// The layer that is used to pass a data blob out of the network
class NEOML_API CSinkLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CSinkLayer )
public:
	explicit CSinkLayer( IMathEngine& mathEngine ) : CBaseLayer( mathEngine, "CCnnSinkLayer", false ) {}

	void Serialize( CArchive& archive ) override;

	// Gets the reference to the output blob
	// It is valid only after the RunOnce method called
	// After each call to RunOnce this blob contains the results
	const CPtr<CDnnBlob>& GetBlob() const;

protected:
	CPtr<CDnnBlob> blob;

	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
};

} // namespace NeoML
