/* Copyright © 2017-2022 ABBYY Production LLC

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

#include <NeoML/Dnn/Layers/BroadcastLayer.h>

namespace NeoML {

CBroadcastLayer::CBroadcastLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CBroadcastLayer", false )
{
}

static const int BroadcastLayerVersion = 0;

void CBroadcastLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( BroadcastLayerVersion );
	CBaseLayer::Serialize( archive );
}

void CBroadcastLayer::Reshape()
{
	CheckInputs();
	CheckOutputs();
	CheckArchitecture( GetInputCount() == GetOutputCount(), GetName(),
		"#inputs != #outputs in CBroadcastLayer" );

	CBlobDesc broadcastedDesc = inputDescs[0];

	for( int inputIndex = 1; inputIndex < GetInputCount(); ++inputIndex ) {
		const CBlobDesc& currInput = inputDescs[inputIndex];
		for( int dim = 0; dim < static_cast<int>( BD_Count ); dim++ ) {
			if( currInput.DimSize( dim ) != 1 && currInput.DimSize( dim ) != broadcastedDesc.DimSize( dim ) ) {
				CheckArchitecture( broadcastedDesc.DimSize( dim ) == 1, GetName(), "inputs can't be broadcasted" );
				broadcastedDesc.SetDimSize( dim, currInput.DimSize( dim ) );
			}
		}
	}

	for( int outputIndex = 0; outputIndex < GetOutputCount(); ++outputIndex ) {
		broadcastedDesc.SetDataType( inputDescs[outputIndex].GetDataType() );
		outputDescs[outputIndex] = broadcastedDesc;
	}
}

void CBroadcastLayer::RunOnce()
{
	for( int inputIndex = 0; inputIndex < inputBlobs.Size(); ++inputIndex ) {
		const CDnnBlob& currInput = *inputBlobs[inputIndex];
		CDnnBlob& currOutput = *outputBlobs[inputIndex];

		if( currInput.HasEqualDimensions( &currOutput ) ) {
			currOutput.CopyFrom( &currInput );
			continue;
		}

		if( currInput.GetDataType() == CT_Float ) {
			MathEngine().BroadcastCopy( currOutput.GetData(), currInput.GetData(),
				currOutput.GetDesc(), currInput.GetDesc(), 1 );
		} else {
			MathEngine().BroadcastCopy( currOutput.GetData<int>(), currInput.GetData<int>(),
				currOutput.GetDesc(), currInput.GetDesc(), 1 );
		}
	}
}

void CBroadcastLayer::BackwardOnce()
{
	NeoAssert( false );
}

} // namespace NeoML

