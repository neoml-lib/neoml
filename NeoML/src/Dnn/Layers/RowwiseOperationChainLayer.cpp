/* Copyright © 2017-2023 ABBYY

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

#include <NeoML/Dnn/Layers/RowwiseOperationChainLayer.h>

namespace NeoML {

CRowwiseOperationChainLayer::CRowwiseOperationChainLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CRowwiseOperationChainLayer", false )
{
}

static const int RowwiseOperationChainLayerVersion = 0;

void CRowwiseOperationChainLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( RowwiseOperationChainLayerVersion );
	CBaseLayer::Serialize( archive );
	// TODO: add serialization
}

void CRowwiseOperationChainLayer::Reshape()
{
	CheckInput1();
	CheckLayerArchitecture( inputDescs[0].Depth() == 1, "Non-trivial depth" );

	operationDescs.DeleteAll();
	outputDescs[0] = inputDescs[0];

	for( IRowwiseOperation* operation : operations ) {
		operationDescs.Add( operation->GetDesc( inputDescs[0] ) );
	}

	outputDescs[0] = MathEngine().RowwiseReshape( operationDescs[0], operations.Size(), outputDescs[0]);
	// TODO: support in-place
}

void CRowwiseOperationChainLayer::RunOnce()
{
	MathEngine().RowwiseExecute( inputBlobs[0]->GetDesc(), operationDescs[0], operations.Size(),
		inputBlobs[0]->GetData(), outputBlobs[0]->GetData() );
}

void CRowwiseOperationChainLayer::BackwardOnce()
{
	NeoAssert( false );
}

} // namespace NeoML
