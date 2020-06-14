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

#include <NeoML/Dnn/Layers/PrecisionRecallLayer.h>

namespace NeoML {

CPrecisionRecallLayer::CPrecisionRecallLayer( IMathEngine& mathEngine ) :
	CQualityControlLayer( mathEngine, "CCnnPrecisionRecallLayer" ),
	positivesTotal( 0 ),
	negativesTotal( 0 ),
	positivesCorrect( 0 ),
	negativesCorrect( 0 )
{
}

void CPrecisionRecallLayer::Reshape()
{
	CQualityControlLayer::Reshape();
	// Intended for binary classification
	// For multi-class classificiation use AccuracyLayer
	NeoAssert( inputDescs[0].Channels() == 1 && inputDescs[0].Height() == 1
		&& inputDescs[0].Width() == 1 );
	NeoAssert( inputDescs[0].ObjectCount() == inputDescs[1].ObjectCount() );
	NeoAssert( inputDescs[0].ObjectSize() >= 1 );
	NeoAssert( inputDescs[1].Channels() == 1 && inputDescs[1].Height() == 1
		&& inputDescs[1].Width() == 1 );

	outputDescs[0] = CBlobDesc( CT_Float );
	outputDescs[0].SetDimSize( BD_Channels, 4 );
}

void CPrecisionRecallLayer::GetLastResult( CArray<int>& results )
{
	results.FreeBuffer();
	results.Add( positivesCorrect );
	results.Add( positivesTotal );
	results.Add( negativesCorrect );
	results.Add( negativesTotal );
}

void CPrecisionRecallLayer::OnReset()
{
	positivesCorrect = 0;
	positivesTotal = 0;
	negativesCorrect = 0;
	negativesTotal = 0;
}

void CPrecisionRecallLayer::RunOnceAfterReset()
{
	CPtr<CDnnBlob> inputBlob = inputBlobs[0];
	CPtr<CDnnBlob> expectedLabelsBlob = inputBlobs[1];

	CArray<float> labels;
	labels.SetSize( expectedLabelsBlob->GetObjectCount() );
	expectedLabelsBlob->CopyTo( labels.GetPtr(), labels.Size() );

	CArray<float> networkOutputs;
	networkOutputs.SetSize( inputBlob->GetObjectCount() );
	inputBlob->CopyTo( networkOutputs.GetPtr(), networkOutputs.Size() );

	for( int i = 0; i < inputBlob->GetObjectCount(); i++ ) {

		if( labels[i] > 0 ) {
			if( networkOutputs[i] >= 0 ) {
				positivesCorrect++;
			}
			positivesTotal++;
		} else {
			if( networkOutputs[i] < 0 ) {
				negativesCorrect++;
			}
			negativesTotal++;
		}
	}

	CFastArray<float, 1> buffer;
	buffer.Add( static_cast<float>( positivesCorrect ) );
	buffer.Add( static_cast<float>( positivesTotal ) );
	buffer.Add( static_cast<float>( negativesCorrect ) );
	buffer.Add( static_cast<float>( negativesTotal ) );

	outputBlobs[0]->CopyFrom( buffer.GetPtr() );
}

static const int PrecisionRecallLayerVersion = 2000;

void CPrecisionRecallLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( PrecisionRecallLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CQualityControlLayer::Serialize( archive );
}

} // namespace NeoML
