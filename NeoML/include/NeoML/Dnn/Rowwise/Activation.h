/* Copyright © 2017-2023 ABBYY

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
#include <NeoML/Dnn/Layers/ActivationLayers.h>
#include <NeoML/Dnn/Rowwise/RowwiseOperation.h>

namespace NeoML {

class NEOML_API CRowwiseActivation : public IRowwiseOperation {
public:
	// Creates an equivalent of a block layer
	CRowwiseActivation( IMathEngine& mathEngine, const CActivationDesc& desc )
		: mathEngine( mathEngine ), desc( desc ) {}

	// Constructor for serialization
	explicit CRowwiseActivation( IMathEngine& mathEngine ) : mathEngine( mathEngine ), desc( AF_HSwish ) {}

	// IRowwiseOperation implementation
	CRowwiseOperationDesc* GetDesc( const CBlobDesc& inputDesc ) override;
	void Serialize( CArchive& archive ) override;

private:
	IMathEngine& mathEngine; // math engine used for calculations
	CActivationDesc desc;
};

} // namespace NeoML
