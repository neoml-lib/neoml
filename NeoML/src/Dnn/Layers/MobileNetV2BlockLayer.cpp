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

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/Layers/MobileNetV2BlockLayer.h>
#include <NeoML/Dnn/Layers/ConvLayer.h>
#include <NeoML/Dnn/Layers/ChannelwiseConvLayer.h>
#include <NeoML/Dnn/Layers/EltwiseLayer.h>

namespace NeoML {

static void storeActivationDesc( const CActivationDesc& desc, CArchive& archive )
{
	TActivationFunction type = desc.GetType();
	NeoAssert( type == AF_ReLU || type == AF_HSwish );

	archive.SerializeEnum( type );
	if( type == AF_ReLU ) {
		float threshold = desc.GetParam<CReLULayer::CParam>().UpperThreshold;
		archive.Serialize( threshold );
	}
}

static CActivationDesc loadActivationDesc( CArchive& archive )
{
	TActivationFunction type = AF_HSwish;
	archive.SerializeEnum( type );
	check( type == AF_ReLU || type == AF_HSwish, ERR_BAD_ARCHIVE, archive.Name() );

	switch( type ) {
		case AF_HSwish:
			return CActivationDesc( type );
		case AF_ReLU:
		{
			float threshold = 0;
			archive.Serialize( threshold );
			return CActivationDesc( AF_ReLU, CReLULayer::CParam{ threshold } );
		}
		default:
			NeoAssert( false );
	}

	// Avoid possible compiler warnings
	return CActivationDesc( type );
}

static CPtr<CDnnBlob> nullifyFreeTerm( CDnnBlob* freeTerm )
{
	if( freeTerm == nullptr ) {
		return freeTerm;
	}

	CDnnBlobBuffer<> buffer( *freeTerm, TDnnBlobBufferAccess::Read );
	for( int i = 0; i < buffer.Size(); ++i ) {
		if( buffer[i] != 0 ) {
			return freeTerm;
		}
	}

	return nullptr;
}

//---------------------------------------------------------------------------------------------------------------------

CMobileNetV2BlockLayer::CMobileNetV2BlockLayer( IMathEngine& mathEngine, const CPtr<CDnnBlob>& expandFilter,
		const CPtr<CDnnBlob>& expandFreeTerm, const CActivationDesc& expandActivation, int stride,
		const CPtr<CDnnBlob>& channelwiseFilter, const CPtr<CDnnBlob>& channelwiseFreeTerm,
		const CActivationDesc& channelwiseActivation, const CPtr<CDnnBlob>& downFilter,
		const CPtr<CDnnBlob>& downFreeTerm, bool residual ) :
	CBaseLayer( mathEngine, "MobileNetV2Block", false ),
	residual( residual ),
	stride( stride ),
	expandActivation( expandActivation ),
	channelwiseActivation( channelwiseActivation ),
	convDesc( nullptr )
{
	paramBlobs.SetSize( P_Count );
	setParamBlob( P_ExpandFilter, expandFilter );
	setParamBlob( P_ExpandFreeTerm, nullifyFreeTerm( expandFreeTerm ) );
	setParamBlob( P_ChannelwiseFilter, channelwiseFilter );
	setParamBlob( P_ChannelwiseFreeTerm, nullifyFreeTerm( channelwiseFreeTerm ) );
	setParamBlob( P_DownFilter, downFilter );
	setParamBlob( P_DownFreeTerm, nullifyFreeTerm( downFreeTerm ) );
}

CMobileNetV2BlockLayer::CMobileNetV2BlockLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "MobileNetV2Block", false ),
	residual( false ),
	stride( 0 ),
	expandActivation( AF_HSwish ),
	channelwiseActivation( AF_HSwish ),
	convDesc( nullptr )
{
	paramBlobs.SetSize( P_Count );
}

CMobileNetV2BlockLayer::~CMobileNetV2BlockLayer()
{
	if( convDesc != nullptr ) {
		delete convDesc;
	}
}

void CMobileNetV2BlockLayer::SetResidual( bool newValue )
{
	if( newValue == residual ) {
		return;
	}

	residual = newValue;
	ForceReshape();
}

static const int MobileNetV2BlockLayerVersion = 1;

void CMobileNetV2BlockLayer::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( MobileNetV2BlockLayerVersion );
	CBaseLayer::Serialize( archive );

	archive.Serialize( residual );
	archive.Serialize( stride );

	if( version == 0 ) {
		// In v0 only ReLU activation was supported
		float expandReLUParam = 0;
		float channelwiseReLUParam = 0;
		archive.Serialize( expandReLUParam );
		archive.Serialize( channelwiseReLUParam );
		expandActivation = CActivationDesc( AF_ReLU, CReLULayer::CParam{ expandReLUParam } );
		channelwiseActivation = CActivationDesc( AF_ReLU, CReLULayer::CParam{ channelwiseReLUParam } );
		return;
	}

	if( archive.IsLoading() ) {
		expandActivation = loadActivationDesc( archive );
		channelwiseActivation = loadActivationDesc( archive );
	} else {
		storeActivationDesc( expandActivation, archive );
		storeActivationDesc( channelwiseActivation, archive );
	}
}

void CMobileNetV2BlockLayer::Reshape()
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
	NeoAssert( paramBlobs[P_ChannelwiseFilter]->GetHeight() == 3 );
	NeoAssert( paramBlobs[P_ChannelwiseFilter]->GetWidth() == 3 );
	NeoAssert( paramBlobs[P_ChannelwiseFilter]->GetDepth() == 1 );
	NeoAssert( paramBlobs[P_ChannelwiseFilter]->GetChannelsCount() == expandedChannels );
	if( paramBlobs[P_ChannelwiseFreeTerm] != nullptr ) {
		NeoAssert( paramBlobs[P_ChannelwiseFreeTerm]->GetDataSize() == expandedChannels );
	}

	NeoAssert( paramBlobs[P_DownFilter] != nullptr );
	const int outputChannels = paramBlobs[P_DownFilter]->GetObjectCount();
	NeoAssert( paramBlobs[P_DownFilter]->GetHeight() == 1 );
	NeoAssert( paramBlobs[P_DownFilter]->GetWidth() == 1 );
	NeoAssert( paramBlobs[P_DownFilter]->GetDepth() == 1 );
	NeoAssert( paramBlobs[P_DownFilter]->GetChannelsCount() == expandedChannels );
	if( paramBlobs[P_DownFreeTerm] != nullptr ) {
		NeoAssert( paramBlobs[P_DownFreeTerm]->GetDataSize() == outputChannels );
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
	CBlobDesc channelwiseInputDesc = inputDescs[0];
	channelwiseInputDesc.SetDimSize( BD_Channels, expandedChannels );
	CBlobDesc channelwiseOutputDesc = outputDescs[0];
	channelwiseOutputDesc.SetDimSize( BD_Channels, expandedChannels );
	CBlobDesc freeTermDesc = paramBlobs[P_ChannelwiseFreeTerm] != nullptr
		? paramBlobs[P_ChannelwiseFreeTerm]->GetDesc() : CBlobDesc();
	convDesc = MathEngine().InitBlobChannelwiseConvolution( channelwiseInputDesc, 1, 1, stride, stride,
		paramBlobs[P_ChannelwiseFilter]->GetDesc(),
		paramBlobs[P_ChannelwiseFreeTerm] != nullptr ? &freeTermDesc : nullptr, channelwiseOutputDesc );

	if( InputsMayBeOverwritten() && inputDescs[0].HasEqualDimensions( outputDescs[0] ) ) {
		NeoAssert( stride == 1 );
		EnableInPlace( true );
	}
}

void CMobileNetV2BlockLayer::RunOnce()
{
	const CConstFloatHandle expandFt = paramBlobs[P_ExpandFreeTerm] == nullptr ? CConstFloatHandle()
		: paramBlobs[P_ExpandFreeTerm]->GetData<const float>();
	const CConstFloatHandle channelwiseFt = paramBlobs[P_ChannelwiseFreeTerm] == nullptr ? CConstFloatHandle()
		: paramBlobs[P_ChannelwiseFreeTerm]->GetData<const float>();
	const CConstFloatHandle downFt = paramBlobs[P_DownFreeTerm] == nullptr ? CConstFloatHandle()
		: paramBlobs[P_DownFreeTerm]->GetData<const float>();
	MathEngine().MobileNetV2Block( inputBlobs[0]->GetDesc(), outputBlobs[0]->GetDesc(), *convDesc,
		inputBlobs[0]->GetData(), paramBlobs[P_ExpandFilter]->GetData(),
		expandFt.IsNull() ? nullptr : &expandFt,
		expandActivation.GetType(),
		expandActivation.GetType() == AF_HSwish ? 0.f : expandActivation.GetParam<CReLULayer::CParam>().UpperThreshold,
		paramBlobs[P_ChannelwiseFilter]->GetData(), channelwiseFt.IsNull() ? nullptr : &channelwiseFt,
		channelwiseActivation.GetType(),
		channelwiseActivation.GetType() == AF_HSwish ? 0.f : channelwiseActivation.GetParam<CReLULayer::CParam>().UpperThreshold,
		paramBlobs[P_DownFilter]->GetData(),
		downFt.IsNull() ? nullptr : &downFt,
		residual, outputBlobs[0]->GetData() );
}

CPtr<CDnnBlob> CMobileNetV2BlockLayer::getParamBlob( TParam param ) const
{
	if( paramBlobs[param] == nullptr ) {
		return nullptr;
	}

	return paramBlobs[param]->GetCopy();
}

void CMobileNetV2BlockLayer::setParamBlob( TParam param, const CPtr<CDnnBlob>& blob )
{
	paramBlobs[param] = blob == nullptr ? nullptr : blob->GetCopy();
}

} // namespace NeoML
