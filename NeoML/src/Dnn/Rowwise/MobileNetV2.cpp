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
#include "../Layers/MobileNetBlockUtils.h"

namespace NeoML {

CRowwiseMobileNetV2::CRowwiseMobileNetV2( const CMobileNetV2BlockLayer& blockLayer ) :
	mathEngine( blockLayer.MathEngine() ),
	expandFilter( blockLayer.ExpandFilter() ),
	expandFreeTerm( blockLayer.ExpandFreeTerm() ),
	expandActivation( blockLayer.ExpandActivation() ),
	channelwiseFilter( blockLayer.ChannelwiseFilter() ),
	channelwiseFreeTerm( blockLayer.ChannelwiseFreeTerm() ),
	stride( blockLayer.Stride() ),
	channelwiseActivation( blockLayer.ChannelwiseActivation() ),
	downFilter( blockLayer.DownFilter() ),
	downFreeTerm( blockLayer.DownFreeTerm() ),
	residual( blockLayer.Residual() )
{
}

CRowwiseMobileNetV2::CRowwiseMobileNetV2( IMathEngine& mathEngine ) :
	mathEngine( mathEngine ),
	expandActivation( AF_HSwish ),
	stride( 1 ),
	channelwiseActivation( AF_HSwish ),
	residual( false )
{
}

CRowwiseOperationDesc* CRowwiseMobileNetV2::GetDesc()
{
	CConstFloatHandle expandFreeTermData = expandFreeTerm == nullptr ? CConstFloatHandle()
		: expandFreeTerm->GetData<const float>();
	CConstFloatHandle channelwiseFreeTermData = channelwiseFreeTerm == nullptr ? CConstFloatHandle()
		: channelwiseFreeTerm->GetData<const float>();
	CConstFloatHandle downFreeTermData = downFreeTerm == nullptr ? CConstFloatHandle()
		: downFreeTerm->GetData<const float>();

	CRowwiseOperationDesc* rowwiseDesc = mathEngine.InitRowwiseMobileNetV2( expandFilter->GetChannelsCount(),
		expandFilter->GetData(), ( expandFreeTerm == nullptr ) ? nullptr : &expandFreeTermData, expandFilter->GetObjectCount(),
		expandActivation.GetType(), MobileNetReluParam( expandActivation ),
		channelwiseFilter->GetData(), ( channelwiseFreeTerm == nullptr ) ? nullptr : &channelwiseFreeTermData, stride,
		channelwiseActivation.GetType(), MobileNetReluParam( channelwiseActivation ),
		downFilter->GetData(), ( downFreeTerm == nullptr ) ? nullptr : &downFreeTermData,
		downFilter->GetObjectCount(), residual );

	NeoAssert( rowwiseDesc != nullptr );
	return rowwiseDesc;
}

void CRowwiseMobileNetV2::Serialize( CArchive& archive )
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

REGISTER_NEOML_ROWWISE_OPERATION( CRowwiseMobileNetV2, "RowwiseMobileNetV2Operation" )

} // namespace NeoML
