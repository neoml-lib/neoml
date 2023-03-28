/* Copyright © 2017-2023 ABBYY Production LLC

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

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/Layers/ChannelwiseWith1x1Layer.h>
#include <NeoML/Dnn/Layers/ConvLayer.h>
#include <NeoML/Dnn/Layers/ChannelwiseConvLayer.h>
#include <NeoML/Dnn/Layers/EltwiseLayer.h>
#include "MobileNetBlockUtils.h"

namespace NeoML {

CChannelwiseWith1x1Layer::CChannelwiseWith1x1Layer( IMathEngine& mathEngine, int stride,
		const CPtr<CDnnBlob>& channelwiseFilter, const CPtr<CDnnBlob>& channelwiseFreeTerm,
		const CActivationDesc& activation, const CPtr<CDnnBlob>& convFilter, const CPtr<CDnnBlob>& convFreeTerm,
		bool residual ) :
	CBaseLayer( mathEngine, "ChannelwiseWith1x1", false ),
	stride( stride ),
	activation( activation ),
	residual( residual ),
	convDesc( nullptr )
{
	NeoAssert( activation.GetType() == AF_ReLU || activation.GetType() == AF_HSwish );
	paramBlobs.SetSize( P_Count );
	paramBlobs[P_ChannelwiseFilter] = MobileNetParam( channelwiseFilter );
	paramBlobs[P_ChannelwiseFreeTerm] = MobileNetFreeTerm( channelwiseFreeTerm );
	paramBlobs[P_ConvFilter] = MobileNetParam( convFilter );
	paramBlobs[P_ConvFreeTerm] = MobileNetFreeTerm( convFreeTerm );
}

CChannelwiseWith1x1Layer::CChannelwiseWith1x1Layer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "ChannelwiseWith1x1", false ),
	stride( 1 ),
	activation( AF_HSwish ),
	residual( false ),
	convDesc( nullptr )
{
	paramBlobs.SetSize( P_Count );
}

CChannelwiseWith1x1Layer::~CChannelwiseWith1x1Layer()
{
	if( convDesc != nullptr ) {
		delete convDesc;
		convDesc = nullptr;
	}
}

CPtr<CDnnBlob> CChannelwiseWith1x1Layer::ChannelwiseFilter() const
{
	return MobileNetParam( paramBlobs[P_ChannelwiseFilter] );
}

CPtr<CDnnBlob> CChannelwiseWith1x1Layer::ChannelwiseFreeTerm() const
{
	return MobileNetParam( paramBlobs[P_ChannelwiseFreeTerm] );
}

CPtr<CDnnBlob> CChannelwiseWith1x1Layer::ConvFilter() const
{
	return MobileNetParam( paramBlobs[P_ConvFilter] );
}

CPtr<CDnnBlob> CChannelwiseWith1x1Layer::ConvFreeTerm() const
{
	return MobileNetParam( paramBlobs[P_ConvFreeTerm] );
}

void CChannelwiseWith1x1Layer::SetResidual( bool newValue )
{
	if( newValue == residual ) {
		return;
	}

	residual = newValue;
	ForceReshape();
}

static const int ChannelwiseWith1x1LayerVersion = 0;

void CChannelwiseWith1x1Layer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( ChannelwiseWith1x1LayerVersion );
	CBaseLayer::Serialize( archive );

	archive.Serialize( residual );
	archive.Serialize( stride );

	if( archive.IsLoading() ) {
		activation = LoadActivationDesc( archive );
		NeoAssert( activation.GetType() == AF_ReLU || activation.GetType() == AF_HSwish );
	} else {
		StoreActivationDesc( activation, archive );
	}
}

void CChannelwiseWith1x1Layer::Reshape()
{
	CheckInput1();

	NeoAssert( inputDescs[0].Depth() == 1 );
	const int inputChannels = inputDescs[0].Channels();

	NeoAssert( stride == 1 || stride == 2 );
	NeoAssert( paramBlobs[P_ChannelwiseFilter] != nullptr );
	NeoAssert( paramBlobs[P_ChannelwiseFilter]->GetObjectCount() == 1 );
	NeoAssert( paramBlobs[P_ChannelwiseFilter]->GetHeight() == 3 );
	NeoAssert( paramBlobs[P_ChannelwiseFilter]->GetWidth() == 3 );
	NeoAssert( paramBlobs[P_ChannelwiseFilter]->GetDepth() == 1 );
	NeoAssert( paramBlobs[P_ChannelwiseFilter]->GetChannelsCount() == inputChannels );
	if( paramBlobs[P_ChannelwiseFreeTerm] != nullptr ) {
		NeoAssert( paramBlobs[P_ChannelwiseFreeTerm]->GetDataSize() == inputChannels );
	}

	NeoAssert( paramBlobs[P_ConvFilter] != nullptr );
	const int outputChannels = paramBlobs[P_ConvFilter]->GetObjectCount();
	NeoAssert( paramBlobs[P_ConvFilter]->GetHeight() == 1 );
	NeoAssert( paramBlobs[P_ConvFilter]->GetWidth() == 1 );
	NeoAssert( paramBlobs[P_ConvFilter]->GetDepth() == 1 );
	NeoAssert( paramBlobs[P_ConvFilter]->GetChannelsCount() == inputChannels );
	if( paramBlobs[P_ConvFreeTerm] != nullptr ) {
		NeoAssert( paramBlobs[P_ConvFreeTerm]->GetDataSize() == outputChannels );
	}

	NeoAssert( !residual || ( inputChannels == outputChannels && stride == 1 ) );

	outputDescs[0] = inputDescs[0];
	if( stride == 2 ) {
		outputDescs[0].SetDimSize( BD_Height, ( inputDescs[0].Height() + 1 ) / 2 );
		outputDescs[0].SetDimSize( BD_Width, ( inputDescs[0].Width() + 1 ) / 2 );
	}
	outputDescs[0].SetDimSize( BD_Channels, outputChannels );

	if( convDesc != nullptr ) {
		delete convDesc;
		convDesc = nullptr;
	}
	CBlobDesc channelwiseOutputDesc = outputDescs[0];
	channelwiseOutputDesc.SetDimSize( BD_Channels, inputChannels );
	CBlobDesc freeTermDesc = paramBlobs[P_ChannelwiseFreeTerm] != nullptr
		? paramBlobs[P_ChannelwiseFreeTerm]->GetDesc() : CBlobDesc();
	convDesc = MathEngine().InitBlobChannelwiseConvolution( inputDescs[0], 1, 1, stride, stride,
		paramBlobs[P_ChannelwiseFilter]->GetDesc(),
		paramBlobs[P_ChannelwiseFreeTerm] != nullptr ? &freeTermDesc : nullptr, channelwiseOutputDesc );
}

void CChannelwiseWith1x1Layer::RunOnce()
{
	const CConstFloatHandle channelwiseFt = paramBlobs[P_ChannelwiseFreeTerm] == nullptr ? CConstFloatHandle()
		: paramBlobs[P_ChannelwiseFreeTerm]->GetData<const float>();
	const CConstFloatHandle convFt = paramBlobs[P_ConvFreeTerm] == nullptr ? CConstFloatHandle()
		: paramBlobs[P_ConvFreeTerm]->GetData<const float>();
	MathEngine().ChannelwiseWith1x1( inputBlobs[0]->GetDesc(), outputBlobs[0]->GetDesc(), *convDesc,
		inputBlobs[0]->GetData(), paramBlobs[P_ChannelwiseFilter]->GetData(),
		channelwiseFt.IsNull() ? nullptr : &channelwiseFt,
		activation.GetType(),
		activation.GetType() == AF_HSwish ? 0.f : activation.GetParam<CReLULayer::CParam>().UpperThreshold,
		paramBlobs[P_ConvFilter]->GetData(),
		convFt.IsNull() ? nullptr : &convFt,
		residual, outputBlobs[0]->GetData() );
}

} // namespace NeoML
