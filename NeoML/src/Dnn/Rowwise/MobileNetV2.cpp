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

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/Rowwise/MobileNetV2.h>
#include <NeoML/Dnn/Layers/MobileNetV2BlockLayer.h>

namespace NeoML {

CMobileNetV2Rowwise::CMobileNetV2Rowwise( const CMobileNetV2BlockLayer& blockLayer ) :
	mathEngine( blockLayer.MathEngine() ),
	expandFilter( blockLayer.ExpandFilter() ),
	expandFreeTerm( blockLayer.ExpandFreeTerm() ),
	expandActivation( blockLayer.ExpandActivation() ),
	stride( blockLayer.Stride() ),
	channelwiseFilter( blockLayer.ChannelwiseFilter() ),
	channelwiseFreeTerm( blockLayer.ChannelwiseFreeTerm() ),
	channelwiseActivation( blockLayer.ChannelwiseActivation() ),
	downFilter( blockLayer.DownFilter() ),
	downFreeTerm( blockLayer.DownFreeTerm() ),
	residual( blockLayer.Residual() )
{
}

CMobileNetV2Rowwise::CMobileNetV2Rowwise( IMathEngine& mathEngine ) :
	mathEngine( mathEngine ),
	stride( 1 ),
	expandActivation( AF_HSwish ),
	channelwiseActivation( AF_HSwish ),
	residual( false )
{
}

CRowwiseOperationDesc* CMobileNetV2Rowwise::GetDesc( const CBlobDesc& inputDesc )
{
	return mathEngine.InitMobileNetV2Rowwise( expandFilter->GetChannelsCount(), expandFilter->GetData(),
		expandFreeTerm == nullptr ? nullptr : &expandFreeTerm->GetData<const float>(),
		expandFilter->GetObjectCount(), expandActivation.GetType(),
		expandActivation.GetType() == AF_ReLU ? expandActivation.GetParam<CReLULayer::CParam>().UpperThreshold : 0,
		channelwiseFilter->GetData(),
		channelwiseFreeTerm == nullptr ? nullptr : &channelwiseFreeTerm->GetData<const float>(),
		stride, channelwiseActivation.GetType(),
		channelwiseActivation.GetType() == AF_ReLU ? channelwiseActivation.GetParam<CReLULayer::CParam>().UpperThreshold : 0,
		downFilter->GetData(),
		downFreeTerm == nullptr ? nullptr : &downFreeTerm->GetData<const float>(),
		downFilter->GetObjectCount(), residual );
}

void CMobileNetV2Rowwise::Serialize( CArchive& archive )
{
	(void) archive.SerializeVersion( 0 ); // version
	SerializeBlob( mathEngine, archive, expandFilter );
	SerializeBlob( mathEngine, archive, expandFreeTerm );
	if( archive.IsStoring() ) {
		StoreActivationDesc( expandActivation, archive );
	} else {
		expandActivation = LoadActivationDesc( archive );
	}
	archive.Serialize( stride );
	SerializeBlob( mathEngine, archive, channelwiseFilter );
	SerializeBlob( mathEngine, archive, channelwiseFreeTerm );
	if( archive.IsStoring() ) {
		StoreActivationDesc( channelwiseActivation, archive );
	} else {
		channelwiseActivation = LoadActivationDesc( archive );
	}
	SerializeBlob( mathEngine, archive, downFilter );
	SerializeBlob( mathEngine, archive, downFreeTerm );
	archive.Serialize( residual );
}

REGISTER_NEOML_ROWWISE_OPERATION( CMobileNetV2Rowwise, "MobileNetV2RowwiseOperation" )

} // namespace NeoML