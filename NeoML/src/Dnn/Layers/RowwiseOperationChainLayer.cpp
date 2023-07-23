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
#include <NeoML/Dnn/Layers/ChannelwiseConvLayer.h>
#include <NeoML/Dnn/Layers/ChannelwiseWith1x1Layer.h>
#include <NeoML/Dnn/Layers/ConvLayer.h>
#include <NeoML/Dnn/Layers/ImageResizeLayer.h>
#include <NeoML/Dnn/Layers/MobileNetV2BlockLayer.h>
#include <NeoML/Dnn/Layers/PoolingLayer.h>
#include <NeoML/Dnn/Optimization/Graph.h>
#include <NeoML/Dnn/Rowwise/Activation.h>
#include <NeoML/Dnn/Rowwise/ChannelwiseConv.h>
#include <NeoML/Dnn/Rowwise/ChannelwiseWith1x1.h>
#include <NeoML/Dnn/Rowwise/Conv.h>
#include <NeoML/Dnn/Rowwise/ImageResize.h>
#include <NeoML/Dnn/Rowwise/MobileNetV2.h>
#include <NeoML/Dnn/Rowwise/Pooling.h>

namespace NeoML {

CRowwiseOperationChainLayer::CRowwiseOperationChainLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CRowwiseOperationChainLayer", false )
{
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
			const CString name = CString( GetRowwiseOperationName( operation ) );
			NeoAssert( operation == nullptr || name != "" ); // assertion on storing not registered rowwise operation
			archive << name;
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
			CheckArchitecture( operationName == "" || operations.Last() != nullptr, operationName,
				"restoring unknown rowwise operation from archive" );
			operations.Last()->Serialize( archive );
		}
	}
}

void CRowwiseOperationChainLayer::Reshape()
{
	// The whole idea of optimization is CPU only
	// On GPU this optimization worsens the performance
	CheckLayerArchitecture( MathEngine().GetType() == MET_Cpu || MathEngine().GetType() == MET_Cuda,
		"Only CPU and CUDA are supported" );

	CheckInput1();
	CheckLayerArchitecture( inputDescs[0].Depth() == 1, "Non-trivial depth" );

	deleteRowwiseDescs();
	NeoPresume( operationDescs.IsEmpty() );
	outputDescs[0] = inputDescs[0];

	for( IRowwiseOperation* operation : operations ) {
		operationDescs.Add( operation->GetDesc() );
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

//---------------------------------------------------------------------------------------------------------------------

template<typename... Targs>
struct IsOneOf
{
	static bool f( const CBaseLayer* ) { return false; }
};

template<typename T, typename... Targs>
struct IsOneOf<T, Targs...>
{
	static bool f( const CBaseLayer* layer )
	{
		return dynamic_cast<const T*>( layer ) != nullptr || IsOneOf<Targs...>::f( layer );
	}
};

static bool isChainLayer( const CBaseLayer* layer )
{
	return IsOneOf<CRowwiseOperationChainLayer>::f( layer );
}

static bool isRowwiseOpLayer( const CBaseLayer* layer )
{
	return IsOneOf<CChannelwiseWith1x1Layer, CChannelwiseConvLayer, CConvLayer, CELULayer, CHardSigmoidLayer,
		CHardTanhLayer, CHSwishLayer, CImageResizeLayer, CLeakyReLULayer, CLinearLayer, CMaxPoolingLayer,
		CMeanPoolingLayer, CMobileNetV2BlockLayer, CReLULayer, CSigmoidLayer, CTanhLayer>::f( layer );
}

static CPtr<IRowwiseOperation> createRowwiseOp( const CBaseLayer* layer )
{
	auto channelwiseWith1x1 = dynamic_cast<const CChannelwiseWith1x1Layer*>( layer );
	if( channelwiseWith1x1 != nullptr ) {
		return new CRowwiseChWith1x1( *channelwiseWith1x1 );
	}
	auto conv = dynamic_cast<const CConvLayer*>( layer );
	if( conv != nullptr ) {
		return new CRowwiseConv( *conv );
	}
	auto chConv = dynamic_cast<const CChannelwiseConvLayer*>( layer );
	if( chConv != nullptr ) {
		return new CRowwiseChConv( *chConv );
	}
	auto imageResize = dynamic_cast<const CImageResizeLayer*>( layer );
	if( imageResize != nullptr ) {
		return new CRowwiseImageResize( *imageResize );
	}
	auto maxPooling = dynamic_cast<const CMaxPoolingLayer*>( layer );
	if( maxPooling != nullptr ) {
		return new CRowwise2DPooling( *maxPooling );
	}
	auto meanPooling = dynamic_cast<const CMeanPoolingLayer*>( layer );
	if( meanPooling != nullptr ) {
		return new CRowwise2DPooling( *meanPooling );
	}
	if( IsOneOf<CELULayer, CHardSigmoidLayer, CHardTanhLayer, CHSwishLayer, CLeakyReLULayer, CLinearLayer,
		CReLULayer, CSigmoidLayer, CTanhLayer>::f( layer ) )
	{
		return new CRowwiseActivation( layer->MathEngine(),
			dynamic_cast<const IActivationLayer*>( layer )->GetDesc() );
	}
	auto mobileNetV2 = dynamic_cast<const CMobileNetV2BlockLayer*>( layer );
	if( mobileNetV2 != nullptr ) {
		return new CRowwiseMobileNetV2( *mobileNetV2 );
	}
	NeoAssert( false );
	return nullptr;
}

void OptimizeRowwiseChains( CDnn& dnn, CArray<int>& chains )
{
	chains.DeleteAll();
	optimization::CGraph graph( dnn );

	CArray<CBaseLayer*> layers;
	graph.GetLayers( layers );

	for( CBaseLayer* layer : layers ) {
		graph.ClearSelection();
		graph.SelectLayer( *layer );

		if( isRowwiseOpLayer( layer ) ) {
			CBaseLayer* prevLayer = graph.SelectTheOnlyConnectedOutput<CBaseLayer>( *layer, true );
			if( isChainLayer( prevLayer ) ) {
				// Append current op to an existing chain
				dynamic_cast<CRowwiseOperationChainLayer*>( prevLayer )->AddOperation( createRowwiseOp( layer ) );
				graph.SwitchOutputs( *layer, 0, *prevLayer, 0 );
				graph.DeleteLayer( *layer );
			} else if( isRowwiseOpLayer( prevLayer ) ) {
				// Merge 2 rowwise ops into chain
				CPtr<CRowwiseOperationChainLayer> chainLayer = new CRowwiseOperationChainLayer( dnn.GetMathEngine() );
				chainLayer->SetName( graph.GetUniqueName( "RowwiseChain" ) );
				chainLayer->AddOperation( createRowwiseOp( prevLayer ) );
				chainLayer->AddOperation( createRowwiseOp( layer ) );
				graph.AddLayer( *chainLayer );
				optimization::CLayerOutput<> chainInput = graph.GetConnectedOutput( *prevLayer, 0 );
				graph.Connect( *chainLayer, 0, *chainInput.Layer, chainInput.Index );
				graph.SwitchOutputs( *layer, 0, *chainLayer, 0 );
				graph.DeleteSelectedLayers();
			}
		} else if( isChainLayer( layer ) ) {
			CRowwiseOperationChainLayer* currChain = dynamic_cast<CRowwiseOperationChainLayer*>( layer );
			CBaseLayer* prevLayer = graph.SelectTheOnlyConnectedOutput<CBaseLayer>( *layer, true );
			if( isChainLayer( prevLayer ) ) {
				// Move operations from currChain into prevChain and delete currChain
				CRowwiseOperationChainLayer* prevChain = dynamic_cast<CRowwiseOperationChainLayer*>( prevLayer );
				for( int i = 0; i < currChain->OperationCount(); ++i ) {
					prevChain->AddOperation( currChain->GetOperation( i ) );
				}
				graph.SwitchOutputs( *currChain, 0, *prevChain, 0 );
				graph.DeleteLayer( *currChain );
			} else if( isRowwiseOpLayer( prevLayer ) ) {
				// Create new chain which starts with operation from prevLayer and then does all the ops from currChain
				CPtr<CRowwiseOperationChainLayer> newChain = new CRowwiseOperationChainLayer( dnn.GetMathEngine() );
				newChain->SetName( graph.GetUniqueName( "RowwiseChain" ) );
				newChain->AddOperation( createRowwiseOp( prevLayer ) );
				for( int i = 0; i < currChain->OperationCount(); ++i ) {
					newChain->AddOperation( currChain->GetOperation( i ) );
				}
				optimization::CLayerOutput<> chainInput = graph.GetConnectedOutput( *prevLayer, 0 );
				graph.AddLayer( *newChain );
				graph.Connect( *newChain, 0, *chainInput.Layer, chainInput.Index );
				graph.SwitchOutputs( *currChain, 0, *newChain, 0 );
				graph.DeleteSelectedLayers();
			}
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
