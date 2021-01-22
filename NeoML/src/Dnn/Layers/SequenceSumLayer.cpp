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

#include <NeoML/Dnn/Layers/SequenceSumLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

void CSequenceSumLayer::Reshape()
{
	CheckInputs();
	CheckArchitecture( GetInputCount() == 1,
		GetName(), "Sequence sum layer must have one input" );

	outputDescs[0] = inputDescs[0];
	outputDescs[0].SetDimSize(BD_BatchLength, 1);;
}

void CSequenceSumLayer::RunOnce()
{
	MathEngine().SumMatrixRows(1, outputBlobs[0]->GetData(), inputBlobs[0]->GetData(),
		inputBlobs[0]->GetBatchLength(), outputBlobs[0]->GetDataSize());
}

void CSequenceSumLayer::BackwardOnce()
{
	inputDiffBlobs[0]->Clear();
	MathEngine().AddVectorToMatrixRows(1, inputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetBatchLength(), outputDiffBlobs[0]->GetDataSize(), outputDiffBlobs[0]->GetData());
}

static const int SequenceSumLayerVersion = 2000;

void CSequenceSumLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( SequenceSumLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );
}

CLayerWrapper<CSequenceSumLayer> SequenceSum()
{
	return CLayerWrapper<CSequenceSumLayer>( "SequenceSum" );
}

} // namespace NeoML
