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

#include <NeoML/Dnn/Layers/AddToObjectLayer.h>

namespace NeoML {

CAddToObjectLayer::CAddToObjectLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CAddToObjectLayer", false )
{
}

static const int AddToObjectLayerVersion = 0;

void CAddToObjectLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( AddToObjectLayerVersion );
	CBaseLayer::Serialize( archive );
}

void CAddToObjectLayer::Reshape()
{
	CheckInputs();
	NeoAssert( inputDescs.Size() == 2 );

	CheckArchitecture( inputDescs[0].Channels() == inputDescs[1].Channels(),
		GetName(), "input Channels dimensions mismatch" );
	CheckArchitecture( inputDescs[0].Depth() == inputDescs[1].Depth(),
		GetName(), "input Depth dimensions mismatch" );
	CheckArchitecture( inputDescs[0].Width() == inputDescs[1].Width(),
		GetName(), "input Width dimensions mismatch" );
	CheckArchitecture( inputDescs[0].Height() == inputDescs[1].Height(),
		GetName(), "input Height dimensions mismatch" );

	CheckArchitecture( inputDescs[1].ObjectCount() == 1, GetName(),
		"CAddToObjectLayer wrong input BatchLength dimension" );

	outputDescs.SetSize( 1 );
	const CBlobDesc& inputDesc = inputDescs[0];
	outputDescs[0] = inputDesc;
}

void CAddToObjectLayer::RunOnce()
{
	MathEngine().AddVectorToMatrixRows( 1, inputBlobs[0]->GetData(), outputBlobs[0]->GetData(),
		inputBlobs[0]->GetObjectCount(), inputBlobs[1]->GetObjectSize(), inputBlobs[1]->GetData() );
}

void CAddToObjectLayer::BackwardOnce()
{
	MathEngine().VectorCopy( inputDiffBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetDataSize() );
	MathEngine().SumMatrixRows( 1, inputDiffBlobs[1]->GetData(), outputDiffBlobs[0]->GetData(),
		outputDiffBlobs[0]->GetObjectCount(), outputDiffBlobs[0]->GetObjectSize() );
}

} // namespace NeoML
