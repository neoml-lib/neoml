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
	TActivationFunction type;
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

//---------------------------------------------------------------------------------------------------------------------

CMobileNetV3PostSEBlockLayer::CMobileNetV3PostSEBlockLayer( IMathEngine& mathEngine,
		const CActivationDesc& activation, const CPtr<CDnnBlob>& downFilter,
		const CPtr<CDnnBlob>& downFreeTerm ) :
	CBaseLayer( mathEngine, "MobileNetV3PostSEBlock", false ),
	activation( activation )
{
	paramBlobs.SetSize( P_Count );
	setParamBlob( P_DownFilter, downFilter );
	setParamBlob( P_DownFreeTerm, downFreeTerm );
}

CMobileNetV3PostSEBlockLayer::CMobileNetV3PostSEBlockLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "MobileNetV3PostSEBlock", false ),
	activation( AF_HSwish )
{
	paramBlobs.SetSize( P_Count );
}

static const int MobileNetV3PostSEBlockLayerVersion = 0;

void CMobileNetV3PostSEBlockLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( MobileNetV3PostSEBlockLayerVersion );
	CBaseLayer::Serialize( archive );

	if( archive.IsLoading() ) {
		activation = loadActivationDesc( archive );
	} else {
		storeActivationDesc( activation, archive );
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
		activation.GetType(),
		activation.GetType() == AF_HSwish ? 0.f : activation.GetParam<CReLULayer::CParam>().UpperThreshold,
		paramBlobs[P_DownFilter]->GetData(),
		downFt.IsNull() ? nullptr : &downFt,
		outputBlobs[0]->GetData() );
}

CPtr<CDnnBlob> CMobileNetV3PostSEBlockLayer::getParamBlob( TParam param ) const
{
	if( paramBlobs[param] == nullptr ) {
		return nullptr;
	}

	return paramBlobs[param]->GetCopy();
}

void CMobileNetV3PostSEBlockLayer::setParamBlob( TParam param, const CPtr<CDnnBlob>& blob )
{
	paramBlobs[param] = blob == nullptr ? nullptr : blob->GetCopy();
}

} // namespace NeoML
