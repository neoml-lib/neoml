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

class NEOML_API CChannelwiseWith1x1Rowwise : public IRowwiseOperation {
public:
	// Creates an equivalent of a block layer
	explicit CChannelwiseWith1x1Rowwise( const CChannelwiseWith1x1Layer& blockLayer );
	// Constructor for serialization
	explicit CChannelwiseWith1x1Rowwise( IMathEngine& mathEngine );

	~CChannelwiseWith1x1Rowwise() override;

	// IRowwiseOperation implementation
	CRowwiseOperationDesc* GetDesc( const CBlobDesc& inputDesc ) override;
	void Serialize( CArchive& archive ) override
	{
	} // TODO: realization

private:
	IMathEngine& mathEngine; // math engine used for calculations
	int stride; // stride of channnelwise convolution
	CPtr<CDnnBlob> channelwiseFilter; // filter of channelwise convolution
	CPtr<CDnnBlob> channelwiseFreeTerm; // free term of channelwise convolution (if present)
	CActivationDesc activation; // activation after channelwise convolution
	CPtr<CDnnBlob> convFilter; // filter of 1x1 convolution
	CPtr<CDnnBlob> convFreeTerm; // free term of 1x1 convolution (if present)
	bool residual; // Does block have residual connection?
	CChannelwiseConvolutionDesc* convDesc{}; // descriptor of channelwise convolution
};

} // namespace NeoML
