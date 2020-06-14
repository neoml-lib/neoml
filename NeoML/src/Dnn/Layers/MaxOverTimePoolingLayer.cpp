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

#include <NeoML/Dnn/Layers/MaxOverTimePoolingLayer.h>

namespace NeoML {

CMaxOverTimePoolingLayer::CMaxOverTimePoolingLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnMaxOverTimePoolingLayer", false ),
	desc( 0 ),
	globalDesc( 0 ),
	filterLength( 0 ),
	strideLength( 0 )
{
}

static const int MaxOverTimePoolingLayerVersion = 2000;

void CMaxOverTimePoolingLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( MaxOverTimePoolingLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	archive.Serialize(filterLength);
	archive.Serialize(strideLength);

	if( archive.IsLoading() ) {
		ForceReshape();
	}
}

void CMaxOverTimePoolingLayer::SetFilterLength(int length)
{
	if(filterLength == length) {
		return;
	}

	filterLength = length;
	ForceReshape();
}

void CMaxOverTimePoolingLayer::SetStrideLength(int length)
{
	if(strideLength == length) {
		return;
	}
	strideLength = length;
	ForceReshape();
}

void CMaxOverTimePoolingLayer::Reshape()
{
	CheckInputs();
	CheckOutputs();
	CheckArchitecture( GetInputCount() == 1, GetName(), "max-over-time pooling with multiple inputs" );
	CheckArchitecture( GetOutputCount() == 1, GetName(), "max-over-time pooling with multiple outputs" );

	int batchLength = 1;
	if(filterLength > 0 && strideLength > 0) {
		CheckArchitecture( filterLength <= inputDescs[0].BatchLength(),
			GetName(), "max-over-time pooling filter length is greater than input length" );
		batchLength = (inputDescs[0].BatchLength() - filterLength) / strideLength + 1;
	}
	outputDescs[0] = inputDescs[0];
	outputDescs[0].SetDimSize(BD_BatchLength, batchLength);

	maxIndices = 0;
	if(IsBackwardPerformed()) {
		maxIndices = CDnnBlob::CreateBlob( MathEngine(), CT_Int, outputDescs[0] );
		RegisterRuntimeBlob(maxIndices);
	}
	destroyDescs();
}

void CMaxOverTimePoolingLayer::RunOnce()
{
	initDescs();

	CIntHandle maxIndicesData;
	if( maxIndices != 0 ) {
		maxIndicesData = maxIndices->GetData<int>();
	}

	if( filterLength > 0 && strideLength > 0 ) {
		MathEngine().BlobMaxOverTimePooling( *desc, inputBlobs[0]->GetData(),
			 maxIndices != 0 ? &maxIndicesData : 0, outputBlobs[0]->GetData() );
	} else {
		MathEngine().BlobGlobalMaxOverTimePooling( *globalDesc, inputBlobs[0]->GetData(),
			 maxIndices != 0 ? &maxIndicesData : 0, outputBlobs[0]->GetData() );
	}
}

void CMaxOverTimePoolingLayer::BackwardOnce()
{
	initDescs();

	NeoPresume( maxIndices->GetDataSize() == outputDiffBlobs[0]->GetDataSize());
	
	inputDiffBlobs[0]->Clear();
	
	if(filterLength > 0 && strideLength > 0) {
		MathEngine().BlobMaxOverTimePoolingBackward( *desc, outputDiffBlobs[0]->GetData(),
			maxIndices->GetData<int>(), inputDiffBlobs[0]->GetData() );
	} else {
		MathEngine().BlobGlobalMaxOverTimePoolingBackward( *globalDesc, outputDiffBlobs[0]->GetData(),
			maxIndices->GetData<int>(), inputDiffBlobs[0]->GetData() );
	}
}

void CMaxOverTimePoolingLayer::initDescs()
{
	if( desc == 0 && filterLength > 0 && strideLength > 0 ) {
		desc = MathEngine().InitMaxOverTimePooling( inputBlobs[0]->GetDesc(), filterLength, strideLength, outputBlobs[0]->GetDesc() );
	}
	if( globalDesc == 0 && filterLength == 0 && strideLength == 0 ) {
		globalDesc = MathEngine().InitGlobalMaxOverTimePooling( inputBlobs[0]->GetDesc(), outputBlobs[0]->GetDesc() );
	}
}

void CMaxOverTimePoolingLayer::destroyDescs()
{
	if( desc != 0 ) {
		delete desc;
		desc = 0;
	}
	if( globalDesc != 0 ) {
		delete globalDesc;
		globalDesc = 0;
	}
}

} // namespace NeoML
