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

#include <NeoML/Dnn/Layers/ActivationLayers.h>
#include <NeoML/Dnn/Layers/ChannelwiseWith1x1Layer.h>
#include <NeoML/Dnn/Layers/ConvLayer.h>
#include <NeoML/Dnn/Layers/MobileNetV2BlockLayer.h>
#include <NeoML/Dnn/Optimization/Graph.h>
#include <NeoML/Dnn/Rowwise/Activation.h>
#include <NeoML/Dnn/Rowwise/ChannelwiseWith1x1.h>
#include <NeoML/Dnn/Rowwise/Conv.h>
#include <NeoML/Dnn/Rowwise/MobileNetV2.h>

namespace NeoML {

CRowwiseOperationChainLayer::CRowwiseOperationChainLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CRowwiseOperationChainLayer", false )
{
	// The whole idea of optimization is CPU only
	// On GPU this optimization worsens the performance
	NeoAssert( mathEngine.GetType() == MET_Cpu );
}

CRowwiseOperationChainLayer::~CRowwiseOperationChainLayer()
{
	deleteRowwiseDescs();
}

static const int RowwiseOperationChainLayerVersion = 0;

void CRowwiseOperationChainLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( RowwiseOperationChainLayerVersion );
	CBaseLayer::Serialize( archive );

	if( archive.IsStoring() ) {
		archive << operations.Size();
		for( IRowwiseOperation* operation : operations ) {
			archive << CString( GetRowwiseOperationName( operation ) );
			operation->Serialize( archive );
		}
	} else {
		operations.DeleteAll();
		int operationCount = 0;
		archive >> operationCount;
		operations.SetBufferSize( operationCount );
		for( int i = 0; i < operationCount; ++i ) {
			CString operationName;
			archive >> operationName;
			operations.Add( CreateRowwiseOperation<IRowwiseOperation>( operationName, MathEngine() ) );
			operations.Last()->Serialize( archive );
		}
	}
}

void CRowwiseOperationChainLayer::Reshape()
{
	CheckInput1();
	CheckLayerArchitecture( inputDescs[0].Depth() == 1, "Non-trivial depth" );

	deleteRowwiseDescs();
	NeoPresume( operationDescs.IsEmpty() );
	outputDescs[0] = inputDescs[0];

	for( IRowwiseOperation* operation : operations ) {
		operationDescs.Add( operation->GetDesc( inputDescs[0] ) );
	}

	outputDescs[0] = MathEngine().RowwiseReshape( operationDescs.GetPtr(), operations.Size(), outputDescs[0] );
}

void CRowwiseOperationChainLayer::RunOnce()
{
	MathEngine().RowwiseExecute( inputBlobs[0]->GetDesc(), operationDescs.GetPtr(), operations.Size(),
		inputBlobs[0]->GetData(), outputBlobs[0]->GetData() );
}

void CRowwiseOperationChainLayer::BackwardOnce()
{
	NeoAssert( false );
}

void CRowwiseOperationChainLayer::deleteRowwiseDescs()
{
	for( int i = 0; i < operationDescs.Size(); ++i ) {
		delete operationDescs[i];
	}
	operationDescs.DeleteAll();
}

//=====================================================================================================================

void OptimizeRowwiseChains( CDnn& dnn, CArray<int>& chains )
{
	chains.DeleteAll();
	optimization::CGraph graph( dnn );

	auto isChainLayer = [] ( const CBaseLayer* layer ) -> bool {
		return dynamic_cast<const CRowwiseOperationChainLayer*>( layer ) != nullptr;
	};

	auto isRowwiseLayer = [] ( const CBaseLayer* layer ) -> bool {
		return dynamic_cast<const CChannelwiseWith1x1Layer*>( layer ) != nullptr
			|| dynamic_cast<const CConvLayer*>( layer ) != nullptr
			|| dynamic_cast<const CHSwishLayer*>( layer ) != nullptr
			|| dynamic_cast<const CMobileNetV2BlockLayer*>( layer ) != nullptr
			|| dynamic_cast<const CReLULayer*>( layer ) != nullptr
			|| dynamic_cast<const CSigmoidLayer*>( layer ) != nullptr;
	};

	auto createOperation = [] ( const CBaseLayer* layer ) -> CPtr<IRowwiseOperation> {
		auto channelwiseWith1x1 = dynamic_cast<const CChannelwiseWith1x1Layer*>( layer );
		if( channelwiseWith1x1 != nullptr ) {
			return new CChannelwiseWith1x1Rowwise( *channelwiseWith1x1 );
		}
		auto conv = dynamic_cast<const CConvLayer*>( layer );
		if( conv != nullptr ) {
			return new CConvRowwise( *conv );
		}
		auto hSwish = dynamic_cast<const CHSwishLayer*>( layer );
		auto relu = dynamic_cast<const CReLULayer*>( layer );
		auto sigmoid = dynamic_cast<const CSigmoidLayer*>( layer );
		if( hSwish != nullptr || relu != nullptr || sigmoid != nullptr ) {
			return new CActivationRowwise( layer->MathEngine(),
				dynamic_cast<const IActivationLayer*>( layer )->GetDesc() );
		}
		auto mobileNetV2 = dynamic_cast<const CMobileNetV2BlockLayer*>( layer );
		if( mobileNetV2 != nullptr ) {
			return new CMobileNetV2Rowwise( *mobileNetV2 );
		}
		NeoAssert( false );
		return nullptr;
	};

	CArray<CBaseLayer*> layers;
	graph.GetLayers( layers );

	for( CBaseLayer* layer : layers ) {
		graph.ClearSelection();

		if( !isRowwiseLayer( layer ) ) {
			continue;
		}

		graph.SelectLayer( *layer );
		CBaseLayer* prevLayer = graph.SelectTheOnlyConnectedOutput<CBaseLayer>( *layer, true );
		if( isChainLayer( prevLayer ) ) {
			dynamic_cast<CRowwiseOperationChainLayer*>( prevLayer )->AddOperation( createOperation( layer ) );
			graph.SwitchOutputs( *layer, 0, *prevLayer, 0 );
			graph.DeleteLayer( *layer );
		} else if( isRowwiseLayer( prevLayer ) ) {
			CPtr<CRowwiseOperationChainLayer> chainLayer = new CRowwiseOperationChainLayer( dnn.GetMathEngine() );
			chainLayer->SetName( graph.GetUniqueName( "RowwiseChain" ) );
			chainLayer->AddOperation( createOperation( prevLayer ) );
			chainLayer->AddOperation( createOperation( layer ) );
			graph.AddLayer( *chainLayer );
			optimization::CLayerOutput<> chainInput = graph.GetConnectedOutput( *prevLayer, 0 );
			graph.Connect( *chainLayer, 0, *chainInput.Layer, chainInput.Index );
			graph.SwitchOutputs( *layer, 0, *chainLayer, 0 );
			graph.DeleteSelectedLayers();
		}
	}

	graph.ClearSelection();

	graph.GetLayers( layers );
	for( CBaseLayer* layer : layers ) {
		if( isChainLayer( layer ) ) {
			chains.Add( dynamic_cast<const CRowwiseOperationChainLayer*>( layer )->OperationCount() );
		}
	}
}

} // namespace NeoML