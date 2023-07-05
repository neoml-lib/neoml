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

#include "ChannelwiseWith1x1Optimizer.h"
#include <NeoML/Dnn/Optimization/Graph.h>
#include <NeoML/Dnn/DnnOptimization.h>
#include <NeoML/Dnn/Layers/ChannelwiseConvLayer.h>
#include <NeoML/Dnn/Layers/ChannelwiseWith1x1Layer.h>
#include <NeoML/Dnn/Layers/EltwiseLayer.h>
#include <NeoML/Dnn/Layers/MobileNetV2BlockLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxEltwiseLayer.h>

namespace NeoML {

namespace optimization {

void CChannelwiseWith1x1Optimizer::Apply( CDnnOptimizationReport& report )
{
	// Step 1: add residual connections to the CChannelwiseWith1x1Layer which are already in the graph
	report.ChannelwiseWith1x1Residual = optimizeResidualConnections();
	// Step 2: merge basic layers into non-residual CChannelwiseWith1x1Layer
	report.ChannelwiseWith1x1NonResidual = optimizeNonResidualBlocks();
	// Step 3: add residual connections to the blocks from Step 2 (where possible)
	const int newResidualBlocks = optimizeResidualConnections();
	report.ChannelwiseWith1x1Residual += newResidualBlocks;
	report.ChannelwiseWith1x1NonResidual -= newResidualBlocks;
}

// Replaces layers which are equivalent to non-residual mobilenetv2 block
int CChannelwiseWith1x1Optimizer::optimizeNonResidualBlocks()
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

		CConvLayer* downConv = dynamic_cast<CConvLayer*>( layer );
		if( downConv == nullptr || !isValid1x1Conv( *downConv ) ) {
			continue;
		}
		graph.SelectLayer( *downConv );

		CBaseLayer* channelwiseActivation = graph.SelectConnectedOutput<>( *downConv, 0 , true ).Layer;
		if( channelwiseActivation == nullptr ) {
			continue;
		}

		CChannelwiseConvLayer* channelwise = nullptr;
		if( isValidActivation( *channelwiseActivation ) ) {
			channelwise = graph.SelectConnectedOutput<CChannelwiseConvLayer>( *channelwiseActivation, 0, true ).Layer;
		} else {
			channelwise = dynamic_cast<CChannelwiseConvLayer*>( channelwiseActivation );
			channelwiseActivation = nullptr;
		}

		if( channelwise == nullptr || !isValidChannelwise( *channelwise ) ) {
			continue;
		}
		CLayerOutput<> channelwiseWith1x1Data = graph.GetConnectedOutput<>( *channelwise, 0 );

		NeoAssert( channelwiseActivation == nullptr
			|| dynamic_cast<IActivationLayer*>( channelwiseActivation ) != nullptr );
		const CActivationDesc channelwiseActivationDesc = channelwiseActivation == nullptr
			? CActivationDesc( AF_Linear, CLinearLayer::CParam{ 1.f, 0.f } )
			: dynamic_cast<IActivationLayer*>( channelwiseActivation )->GetDesc();
		CPtr<CChannelwiseWith1x1Layer> channelwiseWith1x1 = new CChannelwiseWith1x1Layer( graph.MathEngine(),
			channelwise->GetStrideHeight(), channelwise->GetFilterData(), channelwise->GetFreeTermData(),
			channelwiseActivationDesc, downConv->GetFilterData(), downConv->GetFreeTermData(), false );
		channelwiseWith1x1->SetName( graph.GetUniqueName( "ChannelwiseWith1x1" ) );
		graph.AddLayer( *channelwiseWith1x1 );
		graph.Connect( *channelwiseWith1x1, 0, *channelwiseWith1x1Data.Layer, channelwiseWith1x1Data.Index );
		graph.SwitchOutputs( *downConv, 0, *channelwiseWith1x1, 0 );
		graph.DeleteSelectedLayers();
		blocksOptimized++;
	}

	graph.ClearSelection();

	return blocksOptimized;
}

// Merges residual connection layers into non-residual mobilenetv2 blocks
int CChannelwiseWith1x1Optimizer::optimizeResidualConnections()
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
			CChannelwiseWith1x1Layer* channelwiseWith1x1 = graph.GetConnectedOutput<CChannelwiseWith1x1Layer>(
				*layer, i ).Layer;
			if( channelwiseWith1x1 == nullptr || graph.GetInputCount( *channelwiseWith1x1 ) != 1
				|| graph.GetOutputCount( *channelwiseWith1x1 ) != 1
				|| graph.GetConnectedInputsCount( *channelwiseWith1x1, 0 ) != 1
				|| channelwiseWith1x1->Residual() )
			{
				continue;
			}

			CLayerOutput<> channelwiseWith1x1Data = graph.GetConnectedOutput<>( *channelwiseWith1x1, 0 );
			CLayerOutput<> otherResidualData = graph.GetConnectedOutput<>( *layer, 1 - i );
			if( channelwiseWith1x1Data == otherResidualData ) {
				graph.SwitchOutputs( *layer, 0, *channelwiseWith1x1, 0 );
				channelwiseWith1x1->SetResidual( true );
				graph.DeleteLayer( *layer );
				++blocksOptimized;
				break;
			}
		}
	}

	return blocksOptimized;
}

// Checks that CConvLayer meets the criteria of 1x1 convolution inside MobileNetV2 block
bool CChannelwiseWith1x1Optimizer::isValid1x1Conv( CConvLayer& conv ) const
{
	return graph.GetInputCount( conv ) == 1 && conv.GetFilterHeight() == 1 && conv.GetFilterWidth() == 1
		&& conv.GetPaddingHeight() == 0 && conv.GetPaddingWidth() == 0 && conv.GetStrideHeight() == 1
		&& conv.GetStrideWidth() == 1;
}

// Checks that layer meets the criteria for activation function inside MobileNetV2 block
bool CChannelwiseWith1x1Optimizer::isValidActivation( CBaseLayer& layer ) const
{
	return ( dynamic_cast<CReLULayer*>( &layer ) != nullptr || dynamic_cast<CHSwishLayer*>( &layer ) != nullptr )
		&& graph.GetInputCount( layer ) == 1;
}

// Checks that channelwise layer meets the criteria for channelwise inside MobileNetV2 block
bool CChannelwiseWith1x1Optimizer::isValidChannelwise( CChannelwiseConvLayer& channelwise ) const
{
	return graph.GetInputCount( channelwise ) == 1
		&& channelwise.GetFilterHeight() == 3 && channelwise.GetFilterWidth() == 3
		&& channelwise.GetDilationHeight() == 1 && channelwise.GetDilationWidth() == 1
		&& channelwise.GetPaddingHeight() == 1 && channelwise.GetPaddingWidth() == 1
		&& channelwise.GetStrideHeight() == channelwise.GetStrideWidth()
		&& ( channelwise.GetStrideHeight() == 1 || channelwise.GetStrideHeight() == 2 );
}

} // namespace optimization

} // namespace NeoML
