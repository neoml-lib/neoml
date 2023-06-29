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

#include <NeoML/Dnn/Rowwise/ChannelwiseWith1x1.h>
#include <NeoML/Dnn/Layers/ChannelwiseWith1x1Layer.h>
#include "../Layers/MobileNetBlockUtils.h"

namespace NeoML {

CRowwiseChWith1x1::CRowwiseChWith1x1( const CChannelwiseWith1x1Layer& blockLayer ) :
	mathEngine( blockLayer.MathEngine() ),
	stride( blockLayer.Stride() ),
	channelwiseFilter( blockLayer.ChannelwiseFilter() ),
	channelwiseFreeTerm( blockLayer.ChannelwiseFreeTerm() ),
	activation( blockLayer.Activation() ),
	convFilter( blockLayer.ConvFilter() ),
	convFreeTerm( blockLayer.ConvFreeTerm() ),
	residual( blockLayer.Residual() )
{
}

CRowwiseChWith1x1::CRowwiseChWith1x1( IMathEngine& mathEngine ) :
	mathEngine( mathEngine ),
	stride( 1 ),
	activation( AF_HSwish ),
	residual( false )
{
}

CRowwiseOperationDesc* CRowwiseChWith1x1::GetDesc( const CBlobDesc& inputDesc )
{
	return mathEngine.InitRowwiseChWith1x1( stride, channelwiseFilter->GetData(),
		channelwiseFreeTerm == nullptr ? nullptr : &channelwiseFreeTerm->GetData<const float>(),
		activation.GetType(), MobileNetActivationParam( activation ), convFilter->GetData(),
		convFreeTerm == nullptr ? nullptr : &convFreeTerm->GetData<const float>(),
		convFilter->GetObjectCount(), residual );
}

void CRowwiseChWith1x1::Serialize( CArchive& archive )
{
	(void) archive.SerializeVersion( 0 ); // version
	archive.Serialize( stride );
	SerializeBlob( mathEngine, archive, channelwiseFilter );
	SerializeBlob( mathEngine, archive, channelwiseFreeTerm );
	if( archive.IsStoring() ) {
		StoreActivationDesc( activation, archive );
	} else {
		activation = LoadActivationDesc( archive );
	}
	SerializeBlob( mathEngine, archive, convFilter );
	SerializeBlob( mathEngine, archive, convFreeTerm );
	archive.Serialize( residual );
}

REGISTER_NEOML_ROWWISE_OPERATION( CRowwiseChWith1x1, "RowwiseChWith1x1Operation" )

} // namespace NeoML
