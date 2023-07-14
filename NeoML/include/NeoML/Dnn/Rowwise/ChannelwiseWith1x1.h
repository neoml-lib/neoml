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
#include <NeoML/Dnn/DnnBlob.h>
#include <NeoML/Dnn/Layers/ActivationLayers.h>
#include <NeoML/Dnn/Rowwise/RowwiseOperation.h>

namespace NeoML {

class CChannelwiseWith1x1Layer;

class NEOML_API CRowwiseChWith1x1 : public IRowwiseOperation {
public:
	// Creates an equivalent of a block layer
	explicit CRowwiseChWith1x1( const CChannelwiseWith1x1Layer& blockLayer );
	// Constructor for serialization
	explicit CRowwiseChWith1x1( IMathEngine& mathEngine );

	// IRowwiseOperation implementation
	CRowwiseOperationDesc* GetDesc() override;
	void Serialize( CArchive& archive ) override;

private:
	IMathEngine& mathEngine; // math engine used for calculations
	int stride; // stride of channnelwise convolution
	CPtr<CDnnBlob> channelwiseFilter; // filter of channelwise convolution
	CPtr<CDnnBlob> channelwiseFreeTerm; // free term of channelwise convolution (if present)
	CActivationDesc activation; // activation after channelwise convolution
	CPtr<CDnnBlob> convFilter; // filter of 1x1 convolution
	CPtr<CDnnBlob> convFreeTerm; // free term of 1x1 convolution (if present)
	bool residual; // Does block have residual connection?
};

} // namespace NeoML
