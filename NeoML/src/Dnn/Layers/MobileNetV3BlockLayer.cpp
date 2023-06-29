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

#include <NeoML/Dnn/Layers/MobileNetV3BlockLayer.h>
#include <NeoML/Dnn/Layers/ConvLayer.h>
#include <NeoML/Dnn/Layers/ChannelwiseConvLayer.h>
#include <NeoML/Dnn/Layers/EltwiseLayer.h>
#include "MobileNetBlockUtils.h"

namespace NeoML {

CMobileNetV3PreSEBlockLayer::CMobileNetV3PreSEBlockLayer( IMathEngine& mathEngine, const CPtr<CDnnBlob>& expandFilter,
		const CPtr<CDnnBlob>& expandFreeTerm, const CActivationDesc& expandActivation, int stride,
		const CPtr<CDnnBlob>& channelwiseFilter, const CPtr<CDnnBlob>& channelwiseFreeTerm,
		const CActivationDesc& channelwiseActivation ) :
	CBaseLayer( mathEngine, "MobileNetV3PreSEBlock", false ),
	expandActivation( expandActivation ),
	stride( stride ),
	channelwiseActivation( channelwiseActivation ),
	convDesc( nullptr )
{
	NeoAssert( IsValidMobileNetBlockActivation( expandActivation ) );
	NeoAssert( IsValidMobileNetBlockActivation( channelwiseActivation ) );
	paramBlobs.SetSize( P_Count );
	paramBlobs[P_ExpandFilter] = MobileNetParam( expandFilter );
	paramBlobs[P_ExpandFreeTerm] = MobileNetFreeTerm( expandFreeTerm );
	paramBlobs[P_ChannelwiseFilter] = MobileNetParam( channelwiseFilter );
	paramBlobs[P_ChannelwiseFreeTerm] = MobileNetFreeTerm( channelwiseFreeTerm );
}

CMobileNetV3PreSEBlockLayer::CMobileNetV3PreSEBlockLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "MobileNetV3PreSEBlock", false ),
	expandActivation( AF_HSwish ),
	stride( 1 ),
	channelwiseActivation( AF_HSwish ),
	convDesc( nullptr )
{
	paramBlobs.SetSize( P_Count );
}

CMobileNetV3PreSEBlockLayer::~CMobileNetV3PreSEBlockLayer()
{
	if( convDesc != nullptr ) {
		delete convDesc;
	}
}

CPtr<CDnnBlob> CMobileNetV3PreSEBlockLayer::ExpandFilter() const
{
	return MobileNetParam( paramBlobs[P_ExpandFilter] );
}

CPtr<CDnnBlob> CMobileNetV3PreSEBlockLayer::ExpandFreeTerm() const
{
	return MobileNetParam( paramBlobs[P_ExpandFreeTerm] );
}

CPtr<CDnnBlob> CMobileNetV3PreSEBlockLayer::ChannelwiseFilter() const
{
	return MobileNetParam( paramBlobs[P_ChannelwiseFilter] );
}

CPtr<CDnnBlob> CMobileNetV3PreSEBlockLayer::ChannelwiseFreeTerm() const
{
	return MobileNetParam( paramBlobs[P_ChannelwiseFreeTerm] );
}

static const int MobileNetV3PreSEBlockLayerVersion = 0;

void CMobileNetV3PreSEBlockLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( MobileNetV3PreSEBlockLayerVersion );
	CBaseLayer::Serialize( archive );

	archive.Serialize( stride );

	if( archive.IsLoading() ) {
		expandActivation = LoadActivationDesc( archive );
		channelwiseActivation = LoadActivationDesc( archive );
		check( IsValidMobileNetBlockActivation( expandActivation ), ERR_BAD_ARCHIVE, archive.Name() );
		check( IsValidMobileNetBlockActivation( channelwiseActivation ), ERR_BAD_ARCHIVE, archive.Name() );
	} else {
		StoreActivationDesc( expandActivation, archive );
		StoreActivationDesc( channelwiseActivation, archive );
	}
}

void CMobileNetV3PreSEBlockLayer::Reshape()
{
	CheckInput1();

	NeoAssert( inputDescs[0].Depth() == 1 );
	const int inputChannels = inputDescs[0].Channels();

	NeoAssert( paramBlobs[P_ExpandFilter] != nullptr );
	const int expandedChannels = paramBlobs[P_ExpandFilter]->GetObjectCount();
	NeoAssert( paramBlobs[P_ExpandFilter]->GetHeight() == 1 );
	NeoAssert( paramBlobs[P_ExpandFilter]->GetWidth() == 1 );
	NeoAssert( paramBlobs[P_ExpandFilter]->GetDepth() == 1 );
	NeoAssert( paramBlobs[P_ExpandFilter]->GetChannelsCount() == inputChannels );
	if( paramBlobs[P_ExpandFreeTerm] != nullptr ) {
		NeoAssert( paramBlobs[P_ExpandFreeTerm]->GetDataSize() == expandedChannels );
	}

	NeoAssert( stride == 1 || stride == 2 );
	NeoAssert( paramBlobs[P_ChannelwiseFilter] != nullptr );
	NeoAssert( paramBlobs[P_ChannelwiseFilter]->GetObjectCount() == 1 );
	NeoAssert( paramBlobs[P_ChannelwiseFilter]->GetHeight() == paramBlobs[P_ChannelwiseFilter]->GetWidth() );
	NeoAssert( paramBlobs[P_ChannelwiseFilter]->GetWidth() == 3 || paramBlobs[P_ChannelwiseFilter]->GetWidth() == 5 );
	NeoAssert( paramBlobs[P_ChannelwiseFilter]->GetDepth() == 1 );
	NeoAssert( paramBlobs[P_ChannelwiseFilter]->GetChannelsCount() == expandedChannels );
	if( paramBlobs[P_ChannelwiseFreeTerm] != nullptr ) {
		NeoAssert( paramBlobs[P_ChannelwiseFreeTerm]->GetDataSize() == expandedChannels );
	}

	outputDescs[0] = inputDescs[0];
	if( stride == 2 ) {
		outputDescs[0].SetDimSize( BD_Height, ( inputDescs[0].Height() + 1 ) / 2 );
		outputDescs[0].SetDimSize( BD_Width, ( inputDescs[0].Width() + 1 ) / 2 );
	}
	outputDescs[0].SetDimSize( BD_Channels, expandedChannels );

	if( convDesc != nullptr ) {
		delete convDesc;
		convDesc = nullptr;
	}

	CBlobDesc channelwiseInputDesc = inputDescs[0];
	channelwiseInputDesc.SetDimSize( BD_Channels, expandedChannels );
	CBlobDesc channelwiseOutputDesc = outputDescs[0];
	channelwiseOutputDesc.SetDimSize( BD_Channels, expandedChannels );
	CBlobDesc freeTermDesc = paramBlobs[P_ChannelwiseFreeTerm] != nullptr
		? paramBlobs[P_ChannelwiseFreeTerm]->GetDesc() : CBlobDesc();
	const int padding = paramBlobs[P_ChannelwiseFilter]->GetWidth() == 3 ? 1 : 2;
	convDesc = MathEngine().InitBlobChannelwiseConvolution( channelwiseInputDesc, padding, padding, stride, stride,
		paramBlobs[P_ChannelwiseFilter]->GetDesc(),
		paramBlobs[P_ChannelwiseFreeTerm] != nullptr ? &freeTermDesc : nullptr,
		channelwiseOutputDesc );
}

void CMobileNetV3PreSEBlockLayer::RunOnce()
{
	const CConstFloatHandle expandFt = paramBlobs[P_ExpandFreeTerm] == nullptr ? CConstFloatHandle()
		: paramBlobs[P_ExpandFreeTerm]->GetData<const float>();
	const CConstFloatHandle channelwiseFt = paramBlobs[P_ChannelwiseFreeTerm] == nullptr ? CConstFloatHandle()
		: paramBlobs[P_ChannelwiseFreeTerm]->GetData<const float>();
	MathEngine().MobileNetV3PreSEBlock( inputBlobs[0]->GetDesc(), outputBlobs[0]->GetDesc(), *convDesc,
		inputBlobs[0]->GetData(), paramBlobs[P_ExpandFilter]->GetData(),
		expandFt.IsNull() ? nullptr : &expandFt,
		expandActivation.GetType(), MobileNetActivationParam( expandActivation ),
		paramBlobs[P_ChannelwiseFilter]->GetData(),
		channelwiseFt.IsNull() ? nullptr : &channelwiseFt,
		channelwiseActivation.GetType(), MobileNetActivationParam( channelwiseActivation ),
		outputBlobs[0]->GetData() );
}

//---------------------------------------------------------------------------------------------------------------------

CMobileNetV3PostSEBlockLayer::CMobileNetV3PostSEBlockLayer( IMathEngine& mathEngine,
		const CActivationDesc& activation, const CPtr<CDnnBlob>& downFilter,
		const CPtr<CDnnBlob>& downFreeTerm ) :
	CBaseLayer( mathEngine, "MobileNetV3PostSEBlock", false ),
	activation( activation )
{
	NeoAssert( IsValidMobileNetBlockActivation( activation ) );
	paramBlobs.SetSize( P_Count );
	paramBlobs[P_DownFilter] = MobileNetParam( downFilter );
	paramBlobs[P_DownFreeTerm] = MobileNetFreeTerm( downFreeTerm );
}

CMobileNetV3PostSEBlockLayer::CMobileNetV3PostSEBlockLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "MobileNetV3PostSEBlock", false ),
	activation( AF_HSwish )
{
	paramBlobs.SetSize( P_Count );
}

CPtr<CDnnBlob> CMobileNetV3PostSEBlockLayer::DownFilter() const
{
	return MobileNetParam( paramBlobs[P_DownFilter] );
}

CPtr<CDnnBlob> CMobileNetV3PostSEBlockLayer::DownFreeTerm() const
{
	return MobileNetParam( paramBlobs[P_DownFreeTerm] );
}

static const int MobileNetV3PostSEBlockLayerVersion = 0;

void CMobileNetV3PostSEBlockLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( MobileNetV3PostSEBlockLayerVersion );
	CBaseLayer::Serialize( archive );

	if( archive.IsLoading() ) {
		activation = LoadActivationDesc( archive );
		check( IsValidMobileNetBlockActivation( activation ), ERR_BAD_ARCHIVE, archive.Name() );
	} else {
		StoreActivationDesc( activation, archive );
	}
}

void CMobileNetV3PostSEBlockLayer::Reshape()
{
	NeoAssert( GetInputCount() == 2 || GetInputCount() == 3 );

	NeoAssert( inputDescs[I_Channelwise].Depth() == 1 );
	const int inputChannels = inputDescs[0].Channels();
	const int batchSize = inputDescs[0].ObjectCount();

	NeoAssert( inputDescs[I_SqueezeAndExcite].ObjectCount() == batchSize );
	NeoAssert( inputDescs[I_SqueezeAndExcite].GeometricalSize() == 1 );
	NeoAssert( inputDescs[I_SqueezeAndExcite].Channels() == inputChannels );

	NeoAssert( paramBlobs[P_DownFilter] != nullptr );
	const int outputChannels = paramBlobs[P_DownFilter]->GetObjectCount();
	NeoAssert( paramBlobs[P_DownFilter]->GetGeometricalSize() == 1 );
	NeoAssert( paramBlobs[P_DownFilter]->GetChannelsCount() == inputChannels );
	if( paramBlobs[P_DownFreeTerm] != nullptr ) {
		NeoAssert( paramBlobs[P_DownFreeTerm]->GetDataSize() == outputChannels );
	}

	const bool hasResidual = inputDescs.Size() > I_ResidualInput;
	if( hasResidual ) {
		NeoAssert( inputDescs[I_ResidualInput].ObjectCount() == batchSize );
		NeoAssert( inputDescs[I_ResidualInput].Height() == inputDescs[I_Channelwise].Height() );
		NeoAssert( inputDescs[I_ResidualInput].Width() == inputDescs[I_Channelwise].Width() );
		NeoAssert( inputDescs[I_ResidualInput].Depth() == inputDescs[I_Channelwise].Depth() );
		NeoAssert( inputDescs[I_ResidualInput].Channels() == outputChannels );
	}

	outputDescs[0] = inputDescs[I_Channelwise];
	outputDescs[0].SetDimSize( BD_Channels, outputChannels );
}

void CMobileNetV3PostSEBlockLayer::RunOnce()
{
	const CConstFloatHandle residual = inputBlobs.Size() <= I_ResidualInput ? CConstFloatHandle()
		: inputBlobs[I_ResidualInput]->GetData<const float>();
	const CConstFloatHandle downFt = paramBlobs[P_DownFreeTerm] == nullptr ? CConstFloatHandle()
		: paramBlobs[P_DownFreeTerm]->GetData<const float>();
	MathEngine().MobileNetV3PostSEBlock( inputBlobs[I_Channelwise]->GetDesc(), outputBlobs[0]->GetChannelsCount(),
		inputBlobs[I_Channelwise]->GetData(), inputBlobs[I_SqueezeAndExcite]->GetData(),
		residual.IsNull() ? nullptr : &residual,
		activation.GetType(), MobileNetActivationParam( activation ), paramBlobs[P_DownFilter]->GetData(),
		downFt.IsNull() ? nullptr : &downFt,
		outputBlobs[0]->GetData() );
}

} // namespace NeoML
