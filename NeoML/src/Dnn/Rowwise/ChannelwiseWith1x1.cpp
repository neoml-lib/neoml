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

namespace NeoML {

CChannelwiseWith1x1Rowwise::CChannelwiseWith1x1Rowwise( const CChannelwiseWith1x1Layer& blockLayer ) :
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

CChannelwiseWith1x1Rowwise::CChannelwiseWith1x1Rowwise( IMathEngine& mathEngine ) :
	mathEngine( mathEngine ),
	stride( 1 ),
	activation( AF_HSwish ),
	residual( false )
{
}

CChannelwiseWith1x1Rowwise::~CChannelwiseWith1x1Rowwise()
{
	delete convDesc;
}

CRowwiseOperationDesc* CChannelwiseWith1x1Rowwise::GetDesc( const CBlobDesc& inputDesc )
{
	return nullptr;
}

} // namespace NeoML
