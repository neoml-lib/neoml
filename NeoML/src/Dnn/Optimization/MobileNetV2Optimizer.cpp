/* Copyright © 2017-2023 ABBYY Production LLC

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

#include "MobileNetV2Optimizer.h"
#include <NeoML/Dnn/Optimization/Graph.h>
#include <NeoML/Dnn/DnnOptimization.h>
#include <NeoML/Dnn/Layers/ChannelwiseWith1x1Layer.h>
#include <NeoML/Dnn/Layers/ConvLayer.h>
#include <NeoML/Dnn/Layers/EltwiseLayer.h>
#include <NeoML/Dnn/Layers/MobileNetV2BlockLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxEltwiseLayer.h>

namespace NeoML {

namespace optimization {

void CMobileNetV2Optimizer::Apply( CDnnOptimizationReport& report )
{
	// Step 1: add residual connections to the CMobileNetV2BlockLayer which are already in the graph
	report.MobileNetV2ResidualBlocks = optimizeResidualConnections();
	// Step 2: merge basic layers into non-residual CMobileNetV2BlockLayer
	report.MobileNetV2NonResidualBlocks = optimizeNonResidualBlocks();
	// Step 3: add residual connections to the block from Step 2 (where possible)
	const int newResidualBlocks = optimizeResidualConnections();
	report.MobileNetV2ResidualBlocks += newResidualBlocks;
	report.MobileNetV2NonResidualBlocks -= newResidualBlocks;
}

// Replaces layers which are equivalent to non-residual mobilenetv2 block
int CMobileNetV2Optimizer::optimizeNonResidualBlocks()
{
	int blocksOptimized = 0;

	CArray<CBaseLayer*> layers;
	graph.GetLayers( layers );

	for( CBaseLayer* layer : layers ) {
		graph.ClearSelection();
		
		if( !graph.HasLayer( layer ) ) {
			// Layer has already been deleted from the graph
			continue;
		}

		CChannelwiseWith1x1Layer* channelwise = dynamic_cast<CChannelwiseWith1x1Layer*>( layer );
		if( channelwise == nullptr || channelwise->Residual() ) {
			continue;
		}
		graph.SelectLayer( *channelwise );

		CBaseLayer* expandActivationLayer = graph.SelectConnectedOutput<>( *channelwise, 0, true ).Layer;
		CActivationDesc expandActivationDesc( AF_None );
		if( expandActivationLayer == nullptr || !isValidActivation( *expandActivationLayer, expandActivationDesc ) ) {
			continue;
		}

		CConvLayer* expandConv = graph.SelectConnectedOutput<CConvLayer>( *expandActivationLayer, 0, true ).Layer;
		if( expandConv == nullptr || !isValid1x1Conv( *expandConv ) ) {
			continue;
		}

		CLayerOutput<> mobileNetBlockData = graph.GetConnectedOutput<>( *expandConv, 0 );
		CPtr<CMobileNetV2BlockLayer> mobileNetV2Block = new CMobileNetV2BlockLayer( graph.MathEngine(),
			expandConv->GetFilterData(), expandConv->GetFreeTermData(), expandActivationDesc,
			channelwise->Stride(), channelwise->ChannelwiseFilter(), channelwise->ChannelwiseFreeTerm(),
			channelwise->Activation(), channelwise->ConvFilter(), channelwise->ConvFreeTerm(), false );
		mobileNetV2Block->SetName( graph.GetUniqueName( "MobileNetV2Block" ) );
		graph.AddLayer( *mobileNetV2Block );
		graph.Connect( *mobileNetV2Block, 0, *mobileNetBlockData.Layer, mobileNetBlockData.Index );
		graph.SwitchOutputs( *channelwise, 0, *mobileNetV2Block, 0 );
		graph.DeleteSelectedLayers();
		blocksOptimized++;
	}

	graph.ClearSelection();

	return blocksOptimized;
}

// Merges residual connection layers into non-residual mobilenetv2 blocks
int CMobileNetV2Optimizer::optimizeResidualConnections()
{
	int blocksOptimized = 0;

	CArray<CBaseLayer*> layers;
	graph.GetLayers( layers );

	for( CBaseLayer* layer : layers ) {
		if( !graph.HasLayer( layer ) ) {
			// Layer has already been deleted from the graph
			continue;
		}

		CEltwiseSumLayer* residual = dynamic_cast<CEltwiseSumLayer*>( layer );
		COnnxEltwiseLayer* onnxResidual = dynamic_cast<COnnxEltwiseLayer*>( layer );
		if( ( onnxResidual == nullptr || onnxResidual->GetOperation() != COnnxEltwiseLayer::TOperation::Add ) 
			&& residual == nullptr )
		{
			continue;
		}

		if( graph.GetInputCount( *layer ) != 2 ) {
			continue;
		}

		for( int i = 0; i < 2; ++i ) {
			CMobileNetV2BlockLayer* mobileNetV2Block = graph.GetConnectedOutput<CMobileNetV2BlockLayer>(
				*layer, i ).Layer;

			// Workaround for some networks which have Linear{1.f, 0.f} between downConv and residual
			CLinearLayer* linear = nullptr;
			if( mobileNetV2Block == nullptr ) {
				linear = graph.GetConnectedOutput<CLinearLayer>( *layer, i ).Layer;
				if( linear == nullptr || linear->GetFreeTerm() != 0.f || linear->GetMultiplier() != 1.f ) {
					continue;
				}
				mobileNetV2Block = graph.GetConnectedOutput<CMobileNetV2BlockLayer>( *linear, 0 ).Layer;
			}

			if( mobileNetV2Block == nullptr || graph.GetInputCount( *mobileNetV2Block ) != 1
				|| graph.GetOutputCount( *mobileNetV2Block ) != 1
				|| graph.GetConnectedInputsCount( *mobileNetV2Block, 0 ) != 1
				|| mobileNetV2Block->Residual() )
			{
				continue;
			}

			CLayerOutput<> mobileNetBlockData = graph.GetConnectedOutput<>( *mobileNetV2Block, 0 );
			CLayerOutput<> otherResidualData = graph.GetConnectedOutput<>( *layer, 1 - i );
			if( mobileNetBlockData == otherResidualData ) {
				graph.SwitchOutputs( *layer, 0, *mobileNetV2Block, 0 );
				mobileNetV2Block->SetResidual( true );
				graph.DeleteLayer( *layer );
				if( linear != nullptr ) {
					graph.DeleteLayer( *linear );
				}
				++blocksOptimized;
				break;
			}
		}
	}

	return blocksOptimized;
}

// Checks that CConvLayer meets the criteria of 1x1 convolution inside MobileNetV2 block
bool CMobileNetV2Optimizer::isValid1x1Conv( CConvLayer& conv ) const
{
	return graph.GetInputCount( conv ) == 1 && conv.GetFilterHeight() == 1 && conv.GetFilterWidth() == 1
		&& conv.GetPaddingHeight() == 0 && conv.GetPaddingWidth() == 0 && conv.GetStrideHeight() == 1
		&& conv.GetStrideWidth() == 1;
}

// Checks that layer meets the criteria for activation function inside MobileNetV2 block
bool CMobileNetV2Optimizer::isValidActivation( CBaseLayer& layer, CActivationDesc& desc ) const
{
	if( graph.GetInputCount( layer ) != 1 ) {
		return false;
	}

	if( dynamic_cast<CReLULayer*>( &layer ) != nullptr || dynamic_cast<CHSwishLayer*>( &layer ) != nullptr ) {
		desc = dynamic_cast<IActivationLayer*>( &layer )->GetDesc();
		return true;
	}

	// Trivial linear == no activation
	desc = CActivationDesc( AF_None );
	CLinearLayer* linear = dynamic_cast<CLinearLayer*>( &layer );
	return linear != nullptr && linear->GetFreeTerm() == 0 && linear->GetMultiplier() == 1;
}

} // namespace optimization

} // namespace NeoML
