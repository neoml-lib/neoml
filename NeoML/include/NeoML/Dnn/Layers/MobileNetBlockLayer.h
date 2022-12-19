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

	CPtr<CDnnBlob>& ExpandFilter() { return paramBlobs[P_ExpandFilter]; }
	CPtr<CDnnBlob>& ExpandFreeTerm() { return paramBlobs[P_ExpandFreeTerm]; }
	CPtr<CDnnBlob>& ChannelwiseFilter() { return paramBlobs[P_ChannelwiseFilter]; }
	CPtr<CDnnBlob>& ChannelwiseFreeTerm() { return paramBlobs[P_ChannelwiseFreeTerm]; }
	CPtr<CDnnBlob>& DownFilter() { return paramBlobs[P_DownFilter]; }
	CPtr<CDnnBlob>& DownFreeTerm() { return paramBlobs[P_DownFreeTerm]; }
	CFloatHandleVar& ExpandReLUThreshold() { return expandReLUThreshold; }
	CFloatHandleVar& ChannelwiseReLUThreshold() { return channelwiseReLUThreshold; }
	bool& Residual() { return residual; }
	int& Stride() { return stride; }

	void Serialize( CArchive& archive ) override;

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override { NeoAssert( false ); }

private:
	enum TParams {
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
};

int NEOML_API ReplaceMobileNetBlocks( CDnn& dnn );

} // namespace NeoML
