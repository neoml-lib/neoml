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

// The layer that pases data blob to the network
// Unlike CSourceLayer this layer isn't interpreted as a user-provided data
// The source layer that passes a data blob into the network. Can have exactly one output
class NEOML_API CDataLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CDataLayer )
public:
	explicit CDataLayer( IMathEngine& mathEngine ) : CBaseLayer( mathEngine, "CDataLayer", false ) {}

	// Sets the input data blob
	void SetBlob( CDnnBlob* blob );
	// Gets the reference to the input blob
	const CPtr<CDnnBlob>& GetBlob() const { return blob; }

	void Serialize( CArchive& archive ) override;

protected:
	CPtr<CDnnBlob> blob;
	// CBaseLayer class methods
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	void AllocateOutputBlobs() override;
	int BlobsForBackward() const override { return 0; }
};

// Creates CDataLayer with the given name
NEOML_API CDataLayer* Data( CDnn& network, const char* name );

} // namespace NeoML
