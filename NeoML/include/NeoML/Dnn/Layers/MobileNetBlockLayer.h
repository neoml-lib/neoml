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

class NEOML_API CMobileNetBlockLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CMobileNetBlockLayer )
public:
	explicit CMobileNetBlockLayer( IMathEngine& mathEngine );
	~CMobileNetBlockLayer();

	CPtr<CDnnBlob> GetExpandFilter() const { return getParamBlob( P_ExpandFilter ); }
	void SetExpandFilter( const CPtr<CDnnBlob>& blob ) { setParamBlob( P_ExpandFilter, blob ); }
	CPtr<CDnnBlob> GetExpandFreeTerm() const { return getParamBlob( P_ExpandFreeTerm ); }
	void SetExpandFreeTerm( const CPtr<CDnnBlob>& blob ) { setParamBlob( P_ExpandFreeTerm, blob ); }
	float GetExpandReLUThreshold() const { return expandReLUThreshold.GetValue(); }
	void SetExpandReLUThreshold( float newValue ) { expandReLUThreshold.SetValue( newValue ); }

	int GetStride() const { return stride; }
	void SetStride( int newValue ) { NeoAssert( newValue == 1 || newValue == 2 ); stride = newValue; }
	CPtr<CDnnBlob> GetChannelwiseFilter() const { return getParamBlob( P_ChannelwiseFilter ); }
	void SetChannelwiseFilter( const CPtr<CDnnBlob>& blob ) { setParamBlob( P_ChannelwiseFilter, blob ); }
	CPtr<CDnnBlob> GetChannelwiseFreeTerm() const { return getParamBlob( P_ChannelwiseFreeTerm ); }
	void SetChannelwiseFreeTerm( const CPtr<CDnnBlob>& blob ) { setParamBlob( P_ChannelwiseFreeTerm, blob ); }
	float GetChannelwiseReLUThreshold() const { return channelwiseReLUThreshold.GetValue(); }
	void SetChannelwiseReLUThreshold( float newValue ) { channelwiseReLUThreshold.SetValue( newValue ); }

	CPtr<CDnnBlob> GetDownFilter() const { return getParamBlob( P_DownFilter ); }
	void SetDownFilter( const CPtr<CDnnBlob>& blob ) { setParamBlob( P_DownFilter, blob ); }
	CPtr<CDnnBlob> GetDownFreeTerm() const { return getParamBlob( P_DownFreeTerm ); }
	void SetDownFreeTerm( const CPtr<CDnnBlob>& blob ) { setParamBlob( P_DownFreeTerm, blob ); }

	bool HasResidual() const { return residual; }
	void SetResidual( bool newValue ) { residual = newValue; }

	void Serialize( CArchive& archive ) override;

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override { NeoAssert( false ); }

private:
	enum TParam {
		P_ExpandFilter,
		P_ExpandFreeTerm,
		P_ChannelwiseFilter,
		P_ChannelwiseFreeTerm,
		P_DownFilter,
		P_DownFreeTerm,

		P_Count
	};

	bool residual;
	int stride;
	CChannelwiseConvolutionDesc* convDesc;
	CBlobDesc channelwiseInputDesc;
	CBlobDesc channelwiseOutputDesc;
	CFloatHandleVar expandReLUThreshold;
	CFloatHandleVar channelwiseReLUThreshold;

	CPtr<CDnnBlob> getParamBlob( TParam param ) const;
	void setParamBlob( TParam param, const CPtr<CDnnBlob>& blob );
};

int NEOML_API ReplaceMobileNetBlocks( CDnn& dnn );

} // namespace NeoML
