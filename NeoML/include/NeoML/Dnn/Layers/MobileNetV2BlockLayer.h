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
#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

// This layer computes a block in a MobileNetV2 architecture
//
// The block may be without residual connection
//     conv1x1 (expand) -> relu (expandReLU) -> channelwiseConv3x3 -> relu (channelwiseReLU) -> conv1x1 (down)
// or it may be with residual connection
//     -+--> block without residual ----> sum ->
//      |                                  |
//      +----------------------------------+
//
// This layer is faster and consumes less memory than the composite of layers but it has some restrictions:
//     - this layer is untrainable
//     - all 1x1 convolutions must have no paddings and stride == 1
//     - channelwise convolution must have stride 1 or 2, padding == 1 and dilation == 1
//     - only ReLU activation is supported (upper thresholds in ReLU are supported)
//     - free terms are supported (but may be nullptr) for all the convolutions i nblock
class NEOML_API CMobileNetV2BlockLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CMobileNetV2BlockLayer )
public:
	explicit CMobileNetV2BlockLayer( IMathEngine& mathEngine );
	~CMobileNetV2BlockLayer();

	// Expand convolution and RelU parameters
	CPtr<CDnnBlob> GetExpandFilter() const { return getParamBlob( P_ExpandFilter ); }
	void SetExpandFilter( const CPtr<CDnnBlob>& blob ) { setParamBlob( P_ExpandFilter, blob ); }
	CPtr<CDnnBlob> GetExpandFreeTerm() const { return getParamBlob( P_ExpandFreeTerm ); }
	void SetExpandFreeTerm( const CPtr<CDnnBlob>& blob ) { setParamBlob( P_ExpandFreeTerm, blob ); }
	float GetExpandReLUThreshold() const { return expandReLUThreshold.GetValue(); }
	void SetExpandReLUThreshold( float newValue ) { expandReLUThreshold.SetValue( newValue ); }

	// Channelwise convolution and ReLU parameters
	int GetStride() const { return stride; }
	void SetStride( int newValue ) { NeoAssert( newValue == 1 || newValue == 2 ); stride = newValue; }
	CPtr<CDnnBlob> GetChannelwiseFilter() const { return getParamBlob( P_ChannelwiseFilter ); }
	void SetChannelwiseFilter( const CPtr<CDnnBlob>& blob ) { setParamBlob( P_ChannelwiseFilter, blob ); }
	CPtr<CDnnBlob> GetChannelwiseFreeTerm() const { return getParamBlob( P_ChannelwiseFreeTerm ); }
	void SetChannelwiseFreeTerm( const CPtr<CDnnBlob>& blob ) { setParamBlob( P_ChannelwiseFreeTerm, blob ); }
	float GetChannelwiseReLUThreshold() const { return channelwiseReLUThreshold.GetValue(); }
	void SetChannelwiseReLUThreshold( float newValue ) { channelwiseReLUThreshold.SetValue( newValue ); }

	// Down convolution parameters
	CPtr<CDnnBlob> GetDownFilter() const { return getParamBlob( P_DownFilter ); }
	void SetDownFilter( const CPtr<CDnnBlob>& blob ) { setParamBlob( P_DownFilter, blob ); }
	CPtr<CDnnBlob> GetDownFreeTerm() const { return getParamBlob( P_DownFreeTerm ); }
	void SetDownFreeTerm( const CPtr<CDnnBlob>& blob ) { setParamBlob( P_DownFreeTerm, blob ); }

	// Residual connection
	bool HasResidual() const { return residual; }
	void SetResidual( bool newValue ) { residual = newValue; }

	// Serialization
	void Serialize( CArchive& archive ) override;

protected:
	// CBaseLayer methods
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override { NeoAssert( false ); }

private:
	// paramBlobs indices
	enum TParam {
		P_ExpandFilter,
		P_ExpandFreeTerm,
		P_ChannelwiseFilter,
		P_ChannelwiseFreeTerm,
		P_DownFilter,
		P_DownFreeTerm,

		P_Count
	};

	bool residual; // Does block have residual connection?
	int stride; // stride of channnelwise convolution
	CFloatHandleVar expandReLUThreshold; // threshold of expand convolution ReLU
	CFloatHandleVar channelwiseReLUThreshold; // threshold of channelwise convolution ReLU
	CChannelwiseConvolutionDesc* convDesc; // descriptor of channelwise convolution

	CPtr<CDnnBlob> getParamBlob( TParam param ) const;
	void setParamBlob( TParam param, const CPtr<CDnnBlob>& blob );
};

} // namespace NeoML
