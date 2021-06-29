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

#include <NeoML/Dnn/Layers/SpaceToDepthLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

CSpaceToDepthLayer::CSpaceToDepthLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CSpaceToDepthLayer", false ),
	blockSize( 1 )
{
}

void CSpaceToDepthLayer::Reshape()
{
	CheckInput1();
	CheckOutputs();

	CheckArchitecture( blockSize > 1, GetName(), "block size must be more than 1" );
	CheckArchitecture( inputDescs[0].Depth() == 1, GetName(), "input depth must be 1" );

	// The layer needs only one output
	CheckArchitecture( GetOutputCount() == 1, GetName(), "multiple outputs" );

	// The input size should be a multiple of the block size
	CheckArchitecture( inputDescs[0].Height() % blockSize == 0, GetName(),
		"input height must be a multiple of the block size" );
	CheckArchitecture( inputDescs[0].Width() % blockSize == 0, GetName(),
		"input width must be a multiple of the block size" );

	// Calculate the output size
	outputDescs[0] = inputDescs[0];
	outputDescs[0].SetDimSize( BD_Height, outputDescs[0].Height() / blockSize );
	outputDescs[0].SetDimSize( BD_Width, outputDescs[0].Width() / blockSize );
	outputDescs[0].SetDimSize( BD_Channels, outputDescs[0].Channels() * blockSize * blockSize );
}

void CSpaceToDepthLayer::RunOnce()
{
	if( inputBlobs[0]->GetDataType() == CT_Float ) {
		MathEngine().SpaceToDepth( inputBlobs[0]->GetDesc(), inputBlobs[0]->GetData(), blockSize,
			outputBlobs[0]->GetDesc(), outputBlobs[0]->GetData() );
	} else{
		MathEngine().SpaceToDepth( inputBlobs[0]->GetDesc(), inputBlobs[0]->GetData<int>(), blockSize,
			outputBlobs[0]->GetDesc(), outputBlobs[0]->GetData<int>() );
	}
}

void CSpaceToDepthLayer::BackwardOnce()
{
	MathEngine().DepthToSpace( outputDiffBlobs[0]->GetDesc(), outputDiffBlobs[0]->GetData(), blockSize,
		inputDiffBlobs[0]->GetDesc(), inputDiffBlobs[0]->GetData() );
}

static const int SpaceToDepthLayerVersion = 0;

void CSpaceToDepthLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( SpaceToDepthLayerVersion );
	CBaseLayer::Serialize( archive );
	
	archive.Serialize( blockSize );
}

int CSpaceToDepthLayer::GetBlockSize() const
{
	return blockSize;
}

void CSpaceToDepthLayer::SetBlockSize( int _blockSize )
{
	NeoAssert( _blockSize > 1 );
	blockSize = _blockSize;
}

CLayerWrapper<CSpaceToDepthLayer> SpaceToDepth( int blockSize )
{
	return CLayerWrapper<CSpaceToDepthLayer>( "SpaceToDepth", [=]( CSpaceToDepthLayer* result ) {
			result->SetBlockSize( blockSize );
		} );
}

} // namespace NeoML
