/* Copyright © 2017-2020 ABBYY Production LLC

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

#include <NeoML/Dnn/Layers/FastLstmLayer.h>

namespace NeoML {

CFastLstmLayer::CFastLstmLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnFastLstmLayer", true ),
	recurrentActivation( AF_Sigmoid ),
	isInCompatibilityMode( false ),
	hiddenSize( 0 )
{
}

void CFastLstmLayer::Serialize( CArchive& archive )
{
	// FIXME:
}

void CFastLstmLayer::SetHiddenSize( int size )
{
	hiddenSize = size;
	ForceReshape();
}

void CFastLstmLayer::SetDropoutRate( float newDropoutRate )
{
	dropoutRate = newDropoutRate;
	// FIXME: What should we do? Set reshape flage or reinitialize dropout right now.
}

void CFastLstmLayer::SetReverseSequence( bool _isReverseSequense )
{
	// FIXME:
}

void CFastLstmLayer::Reshape()
{
	// FIXME:
}

void CFastLstmLayer::RunOnce()
{
	// FIXME:
}

void CFastLstmLayer::BackwardOnce()
{
	// FIXME:
}

void CFastLstmLayer::setWeightsData( const CPtr<CDnnBlob>& src, CPtr<CDnnBlob>& dst )
{
	// FIXME:
}

void CFastLstmLayer::setFreeTermData( const CPtr<CDnnBlob>& src, CPtr<CDnnBlob>& dst )
{
	// FIXME:
}

} // namespace NeoML
