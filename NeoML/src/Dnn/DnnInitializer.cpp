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

#include <NeoML/Dnn/Dnn.h>
#include <NeoML/Dnn/DnnInitializer.h>

namespace NeoML {

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void CDnnXavierInitializer::InitializeLayerParams(CDnnBlob& blob, int inputCount)
{
	double deviation = sqrt(1. / max(inputCount, 1));

	CArray<float> tempData;
	tempData.SetSize(blob.GetDataSize());

	float* data = tempData.GetPtr();
	for(int i = 0; i < tempData.Size(); ++i) {
		*data++ = (float)Random().Normal(0, deviation);
	}

	blob.CopyFrom(tempData.GetPtr());
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void CDnnXavierUniformInitializer::InitializeLayerParams( CDnnBlob& blob, int inputCount )
{
	const double deviation = sqrt( 1. / max( inputCount, 1 ) );
	float* buffer = blob.GetBuffer<float>( 0, blob.GetDataSize(), false );

	float* data = buffer;
	for( int i = 0; i < blob.GetDataSize(); ++i ) {
		*data++ = static_cast<float>( Random().Uniform( -deviation, deviation ) );
	}

	blob.ReleaseBuffer( buffer, true );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
CDnnUniformInitializer::CDnnUniformInitializer(CRandom& _random) :
	CDnnInitializer(_random), lowerBound(-1.f), upperBound(1.f)
{
}

CDnnUniformInitializer::CDnnUniformInitializer(CRandom& _random, float _lowerBound, float _upperBound) :
	CDnnInitializer(_random), lowerBound(_lowerBound), upperBound(_upperBound)
{
}

void CDnnUniformInitializer::InitializeLayerParams(CDnnBlob& blob, int)
{
	CArray<float> tempData;
	tempData.SetSize(blob.GetDataSize());

	float* data = tempData.GetPtr();
	for(int i = 0; i < tempData.Size(); ++i) {
		*data++ = (float)Random().Uniform(lowerBound, upperBound);
	}

	blob.CopyFrom(tempData.GetPtr());
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void CDnnDistributedInitializer::InitializeLayerParams( CDnnBlob& blob, int inputCount )
{
	if( mathEngine->GetDistributedInfo().Thread == 0 ){
		baseInitializer->InitializeLayerParams( blob, inputCount );
	}

	mathEngine->Broadcast( blob.GetData(), blob.GetDataSize(), 0 );
}

}
