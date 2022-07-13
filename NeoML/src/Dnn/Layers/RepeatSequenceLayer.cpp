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

#include <NeoML/Dnn/Layers/RepeatSequenceLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

CRepeatSequenceLayer::CRepeatSequenceLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnRepeatSequenceLayer", false ),
	repeatCount(1)
{
}

void CRepeatSequenceLayer::SetRepeatCount(int count)
{
	if(repeatCount == count) {
		return;
	}
	repeatCount = count; 
	ForceReshape();
}

static const int RepeatSequenceLayerVersion = 2000;

void CRepeatSequenceLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( RepeatSequenceLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );
	
	archive.Serialize( repeatCount );
}

void CRepeatSequenceLayer::Reshape()
{
	CheckInput1();
	outputDescs[0] = inputDescs[0];
	outputDescs[0].SetDimSize(BD_BatchLength, outputDescs[0].BatchLength() * repeatCount);
}

void CRepeatSequenceLayer::RunOnce()
{
	NeoPresume( outputBlobs[0]->GetDataSize() % inputBlobs[0]->GetDataSize() == 0 );
	const int copyCount = outputBlobs[0]->GetDataSize() / inputBlobs[0]->GetDataSize();

	// If the layer is inside a recurrent network, copyCount should be 1
	// Otherwise it should be repeatCount
	NeoPresume( copyCount == 1 || copyCount == repeatCount );
	MathEngine().SetVectorToMatrixRows(outputBlobs[0]->GetData(), copyCount, inputBlobs[0]->GetDataSize(),
		inputBlobs[0]->GetData());
}

void CRepeatSequenceLayer::BackwardOnce()
{
	NeoPresume( outputDiffBlobs[0]->GetDataSize() % inputDiffBlobs[0]->GetDataSize() == 0 );
	const int copyCount = outputDiffBlobs[0]->GetDataSize() / inputDiffBlobs[0]->GetDataSize();

	// If the layer is inside a recurrent network, copyCount should be 1
	// Otherwise it should be repeatCount
	NeoPresume( copyCount == 1 || copyCount == repeatCount );
	MathEngine().SumMatrixRows(1, inputDiffBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
		copyCount, inputDiffBlobs[0]->GetDataSize());
}

CLayerWrapper<CRepeatSequenceLayer> RepeatSequence( int repeatCount )
{
	return CLayerWrapper<CRepeatSequenceLayer>( "RepeatSequence", [=]( CRepeatSequenceLayer* result ) {
		result->SetRepeatCount( repeatCount );
	} );
}

} // namespace NeoML
