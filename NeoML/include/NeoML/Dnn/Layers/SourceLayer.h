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

// The source layer that passes a data blob into the network. Can have exactly one output
class NEOML_API CSourceLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CSourceLayer )
public:
	explicit CSourceLayer( IMathEngine& mathEngine ) : CBaseLayer( mathEngine, "CCnnSourceLayer", false ), storeBlob( true ) {}

	// Sets the input data blob
	void SetBlob( CDnnBlob* blob );
	// Gets the reference to the input blob
	const CPtr<CDnnBlob>& GetBlob() const { return blob; }

	// Sets whether blob must be serialized with layer or not
	void StoreBlob( bool value ) { storeBlob = value; }
	// Returns true if blob will be serialized with layer
	bool IsStoringBlob() const { return storeBlob; }

	void Serialize( CArchive& archive ) override;

protected:
	CPtr<CDnnBlob> blob;
	bool storeBlob;
	// CBaseLayer class methods
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	void AllocateOutputBlobs() override;
};

// Creates CSourceLayer with name
NEOML_API CSourceLayer* Source( CDnn& network, const char* name );

} // namespace NeoML
