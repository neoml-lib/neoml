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
#include <NeoML/Dnn/Layers/ConvLayer.h>

namespace NeoML {

// The layer that performs channel-wise convolution
// Each channel of the input blob is convolved with the corresponding channel of the single filter
class NEOML_API CChannelwiseConvLayer : public CBaseConvLayer {
	NEOML_DNN_LAYER( CChannelwiseConvLayer )
public:
	explicit CChannelwiseConvLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	CPtr<CDnnBlob> GetFilterData() const override;

	// Sets the blob with the filter coefficients.
	// The BD_Height of the newFilter must be equal to the filter height.
	// The BD_Width of the newFilter must be equal to the filter width.
	// The BD_Channels of the newFilter blob must be equal to the filter count (and to the BD_Channels of inputs).
	// The other newFilter's dimensions must be equal to 1.
	// If newFilter is null, the filter data will be reset.
	void SetFilterData( const CPtr<CDnnBlob>& newFilter ) override;

protected:
	virtual ~CChannelwiseConvLayer() { destroyConvDesc(); }

	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	void LearnOnce() override;
	bool IsFilterTransposed() const override { return true; }

private:
	// Convolution descriptor
	CChannelwiseConvolutionDesc* convDesc;

	void initConvDesc();
	void destroyConvDesc();
};

} // namespace NeoML
