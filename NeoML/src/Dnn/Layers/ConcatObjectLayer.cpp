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
#include <NeoML/Dnn/Layers/ConcatLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

void CConcatObjectLayer::Reshape()
{
	CheckInputs();
	int batchLength = inputDescs[0].BatchLength();
	int batchWidth = inputDescs[0].BatchWidth();
	int objectSize = inputDescs[0].ObjectSize();

	for(int i = 1; i < inputDescs.Size(); ++i) {
		CheckArchitecture( inputDescs[i].BatchLength() == batchLength, GetName(), "input batch length mismatch" );
		CheckArchitecture( inputDescs[i].BatchWidth() == batchWidth, GetName(), "input batch width mismatch" );
		objectSize += inputDescs[i].ObjectSize();
	}

	outputDescs[0] = CBlobDesc( inputDescs[0].GetDataType() );
	outputDescs[0].SetDimSize( BD_BatchLength, batchLength );
	outputDescs[0].SetDimSize( BD_BatchWidth, batchWidth );
	outputDescs[0].SetDimSize( BD_Channels, objectSize );
}

void CConcatObjectLayer::RunOnce()
{
	CDnnBlob::MergeByObject( MathEngine(), inputBlobs, outputBlobs[0] );
}

void CConcatObjectLayer::BackwardOnce()
{
	CDnnBlob::SplitByObject( MathEngine(), outputDiffBlobs[0], inputDiffBlobs );
}

} // namespace NeoML
