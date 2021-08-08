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

#include <NeoML/Dnn/Layers/CastLayer.h>

namespace NeoML {

CCastLayer::CCastLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCastLayer", false ),
	outputType( CT_Float )
{
}

static const int CastLayerVersion = 0;

void CCastLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( CastLayerVersion );
	CBaseLayer::Serialize( archive );

	int outputTypeInt = static_cast<int>( outputType );
	archive.Serialize( outputTypeInt );
	outputType = static_cast<TBlobType>( outputTypeInt );
}

void CCastLayer::SetOutputType( TBlobType type )
{
	if( outputType == type ) {
		return;
	}

	outputType = type;
	ForceReshape();
}

void CCastLayer::Reshape()
{
	CheckArchitecture( inputDescs.Size() == 1, GetName(), "CCastLayer must have 1 input" );
	CheckArchitecture( outputDescs.Size() == 1, GetName(), "CCastLayer must have 1 output" );
	outputDescs[0] = inputDescs[0];
	outputDescs[0].SetDataType( outputType );
}

void CCastLayer::RunOnce()
{
	if( outputBlobs[0]->GetDataType() == inputBlobs[0]->GetDataType() ) {
		outputBlobs[0]->CopyFrom( inputBlobs[0] );
	} else if( inputBlobs[0]->GetDataType() == CT_Int ) {
		MathEngine().VectorConvert( inputBlobs[0]->GetData<int>(), outputBlobs[0]->GetData(), inputBlobs[0]->GetDataSize() );
	} else {
		MathEngine().VectorConvert( inputBlobs[0]->GetData(), outputBlobs[0]->GetData<int>(), inputBlobs[0]->GetDataSize() );
	}
}

void CCastLayer::BackwardOnce()
{
	NeoAssert( inputBlobs[0]->GetDataType() == CT_Float && outputBlobs[0]->GetDataType() == CT_Float );
	inputDiffBlobs[0]->CopyFrom( outputDiffBlobs[0] );
}

} // namespace NeoML
