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
#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoML/Dnn/Layers/SubSequenceLayer.h>

namespace NeoML {

CSubSequenceLayer::CSubSequenceLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnSubSequenceLayer", false ),
	startPos( 0 ),
	length( INT_MAX )
{
}

void CSubSequenceLayer::SetStartPos(int _startPos)
{
	if(startPos == _startPos) {
		return;
	}
	startPos = _startPos;
	ForceReshape();
}

void CSubSequenceLayer::SetLength(int _length)
{
	if(length == _length) {
		return;
	}
	length = _length;
	ForceReshape();
}

static const int SubSequenceLayerVersion = 2000;

void CSubSequenceLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( SubSequenceLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	archive.Serialize(startPos);
	archive.Serialize(length);
}

void CSubSequenceLayer::getSequenceInfo(int& sequenceStart, int& subSequenceLength) const
{
	int maxSequenceLen = inputDescs[0].BatchLength();
	sequenceStart = startPos >= 0 ? min(startPos, maxSequenceLen) : max(0, maxSequenceLen + startPos);
	subSequenceLength = length >= 0 ? min(length, maxSequenceLen - sequenceStart) :
		min(-max(length, -maxSequenceLen), sequenceStart + 1);
}

void CSubSequenceLayer::Reshape()
{
	CheckInputs();
	CheckOutputs();

	int sequenceStart; 
	int subSequenceLength;
	getSequenceInfo(sequenceStart, subSequenceLength);

	outputDescs[0] = inputDescs[0];
	outputDescs[0].SetDimSize(BD_BatchLength, subSequenceLength);

	indices = 0;
	if( IsBackwardPerformed() ) {
		indices = CDnnBlob::CreateDataBlob( MathEngine(), CT_Int, outputDescs[0].BatchLength(), outputDescs[0].BatchWidth(), 1 );
		RegisterRuntimeBlob(indices);
	}
}

void CSubSequenceLayer::RunOnce()
{
	int sequenceStart; 
	int subSequenceLength;
	getSequenceInfo(sequenceStart, subSequenceLength);
	NeoAssert(subSequenceLength == outputBlobs[0]->GetBatchLength());

	CIntHandle indexHandle = (indices == 0) ? CIntHandle() : indices->GetData<int>();
	MathEngine().BlobGetSubSequence( inputBlobs[0]->GetDesc(), inputBlobs[0]->GetData(), indexHandle,
		outputBlobs[0]->GetDesc(), outputBlobs[0]->GetData(), sequenceStart, length < 0 );
}

void CSubSequenceLayer::BackwardOnce()
{
	MathEngine().MatrixSpreadRows( outputDiffBlobs[0]->GetData(),
		outputDiffBlobs[0]->GetBatchLength() * outputDiffBlobs[0]->GetBatchWidth(), 
		outputDiffBlobs[0]->GetObjectSize() * outputDiffBlobs[0]->GetListSize(),
		inputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetBatchLength() * inputDiffBlobs[0]->GetBatchWidth(), 
		indices->GetData<int>(), CConstFloatHandle() );
}

} // namespace NeoML
