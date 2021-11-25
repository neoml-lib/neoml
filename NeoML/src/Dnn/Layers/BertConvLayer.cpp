/* Copyright © 2017-2021 ABBYY Production LLC

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

#include <NeoML/Dnn/Layers/BertConvLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

CBertConvLayer::CBertConvLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CBertConvLayer", false )
{
}

static const int BertConvLayerVersion = 0;

void CBertConvLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( BertConvLayerVersion );
	CBaseLayer::Serialize( archive );
}

void CBertConvLayer::Reshape()
{
	CheckArchitecture( inputDescs.Size() == 2, GetName(), "Layer must have 2 inputs" );
	CheckArchitecture( outputDescs.Size() == 1, GetName(), "Layer must have 1 output" );

	CheckArchitecture( inputDescs[0].ListSize() == 1, GetName(), "Data input's list size must be 1" );
	CheckArchitecture( inputDescs[0].Height() == 1, GetName(), "Data input's height must be 1" );
	CheckArchitecture( inputDescs[0].Width() == 1, GetName(), "Data input's width must be 1" );
	CheckArchitecture( inputDescs[0].Depth() == 1, GetName(), "Data input's depth must be 1" );

	const int seqLen = inputDescs[0].BatchLength();
	const int batchSize = inputDescs[0].BatchWidth();

	CheckArchitecture( inputDescs[1].ListSize() == 1, GetName(), "Kernel input's list size must be 1" );
	CheckArchitecture( inputDescs[1].Width() == 1, GetName(), "Kernel input's width must be 1" );
	CheckArchitecture( inputDescs[1].Depth() == 1, GetName(), "Kernel input's depth must be 1" );
	CheckArchitecture( inputDescs[1].Channels() == 1, GetName(), "Kernel input's channels must be 1" );

	CheckArchitecture( inputDescs[1].BatchLength() == seqLen, GetName(), "Inputs' batch length mismatch" );
	CheckArchitecture( inputDescs[1].BatchWidth() % batchSize == 0, GetName(),
		"Kernel input's batch width must be a multiple of Data input's batch width" );

	const int numHeads = inputDescs[1].BatchWidth() / batchSize;

	CheckArchitecture( inputDescs[0].Channels() % numHeads == 0, GetName(),
		"Data input's channels must be a multiple of number of heads" );

	const int headSize = inputDescs[0].Channels() / numHeads;

	outputDescs[0] = inputDescs[1];
	outputDescs[0].SetDimSize( BD_Height, headSize );
}

void CBertConvLayer::RunOnce()
{
	const int seqLen = inputBlobs[0]->GetBatchLength();
	const int batchSize = inputBlobs[0]->GetBatchWidth();
	const int numHeads = inputBlobs[1]->GetBatchWidth() / batchSize;
	const int headSize = inputBlobs[0]->GetChannelsCount() / numHeads;
	const int kernelSize = inputBlobs[1]->GetHeight();

	MathEngine().BertConv( inputBlobs[0]->GetData(), inputBlobs[1]->GetData(), seqLen, batchSize, numHeads,
		headSize, kernelSize, outputBlobs[0]->GetData() );
}

void CBertConvLayer::BackwardOnce()
{
	const int seqLen = inputBlobs[0]->GetBatchLength();
	const int batchSize = inputBlobs[0]->GetBatchWidth();
	const int numHeads = inputBlobs[1]->GetBatchWidth() / batchSize;
	const int headSize = inputBlobs[0]->GetChannelsCount() / numHeads;
	const int kernelSize = inputBlobs[1]->GetHeight();

	MathEngine().BertConvBackward( inputBlobs[0]->GetData(), inputBlobs[1]->GetData(), outputDiffBlobs[0]->GetData(),
		seqLen, batchSize, numHeads, headSize, kernelSize, inputDiffBlobs[0]->GetData(), inputDiffBlobs[1]->GetData() );
}

} // namespace NeoML
