/* Copyright © 2017-2021 ABBYY Production LLC

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

// Layer that converts data type
class NEOML_API CCastLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CCastLayer )
public:
	explicit CCastLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// Sets output blob type
	// CT_Float by default
	void SetOutputType( TBlobType type );
	TBlobType GetOutputType() const { return outputType; }

protected:
	// CBaseLayer methods
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	int BlobsForBackward() const override { return 0; }

private:
	// output blob type
	TBlobType outputType;
};

NEOML_API CLayerWrapper<CCastLayer> Cast( TBlobType outputType );

} // namespace NeoML
