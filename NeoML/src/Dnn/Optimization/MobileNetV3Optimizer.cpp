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

#include "MobileNetV3Optimizer.h"

#include <NeoML/Dnn/DnnOptimization.h>
#include <NeoML/Dnn/Layers/ConvLayer.h>
#include <NeoML/Dnn/Layers/ChannelwiseConvLayer.h>
#include <NeoML/Dnn/Layers/EltwiseLayer.h>
#include <NeoML/Dnn/Layers/FullyConnectedLayer.h>
#include <NeoML/Dnn/Layers/GlobalMeanPoolingLayer.h>
#include <NeoML/Dnn/Layers/MobileNetV3BlockLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxEltwiseLayer.h>

namespace NeoML {

namespace optimization {

void CMobileNetV3Optimizer::Apply( CDnnOptimizationReport& report )
{
	report.MobileNetV3ResidualBlocks = optimizeResidualBlocks();
	report.MobileNetV3NonResidualBlocks = optimizeNonResidualBlocks();
}

// Optimizes MobileNetV3 blocks with residual connections
int CMobileNetV3Optimizer::optimizeResidualBlocks()
{
	NeoAssert( graph.SelectionSize() == 0 );

	int blocksOptimized = 0;

	CArray<CBaseLayer*> layers;
	graph.GetLayers( layers );
	for( CBaseLayer* layer : layers ) {
		if( !graph.HasLayer( layer ) ) {
			continue;
		}

		graph.ClearSelection();
		CMNv3BlockInfo detectedBlock;
		if( !detectMNv3Residual( *layer, detectedBlock ) ) {
			continue;
		}

		optimizeDetectedBlock( detectedBlock );
		++blocksOptimized;
	}
	graph.ClearSelection();

	NeoAssert( graph.SelectionSize() == 0 );

	return blocksOptimized;
}

// Optimizes MobileNetV3 blocks without residual connections
int CMobileNetV3Optimizer::optimizeNonResidualBlocks()
{
	NeoAssert( graph.SelectionSize() == 0 );

	int blocksOptimized = 0;

	CArray<CBaseLayer*> layers;
	graph.GetLayers( layers );
	for( CBaseLayer* layer : layers ) {
		if( !graph.HasLayer( layer ) ) {
			continue;
		}

		graph.ClearSelection();
		CConvLayer* downConv = dynamic_cast<CConvLayer*>( layer );
		CMNv3BlockInfo detectedBlock;
		if( downConv == nullptr || !detectMNv3NonResidual( *downConv, detectedBlock ) ) {
			continue;
		}

		optimizeDetectedBlock( detectedBlock );
		++blocksOptimized;
	}
	graph.ClearSelection();

	NeoAssert( graph.SelectionSize() == 0 );

	return blocksOptimized;
}

// Checks whether the given layer is a residual layer of MobileNetV3 block
// If it is then writes data about block into detectedBlock, selects layers for future deletion and return true
// If it isn't then returns false (in this case detectedBlock and graph's selection may be in any state)
bool CMobileNetV3Optimizer::detectMNv3Residual( CBaseLayer& residual, CMNv3BlockInfo& detectedBlock )
{
	if( graph.GetInputCount( residual ) != 2 ) {
		return false;
	}

	if( dynamic_cast<CEltwiseSumLayer*>( &residual ) == nullptr ) {
		COnnxEltwiseLayer* onnxSum = dynamic_cast<COnnxEltwiseLayer*>( &residual );
		if( onnxSum == nullptr || onnxSum->GetOperation() != COnnxEltwiseLayer::TOperation::Add ) {
			return false;
		}
	}

	for( int i = 0; i < 2; ++i ) {
		CConvLayer* downConv = graph.GetConnectedOutput<CConvLayer>( residual, i ).Layer;

		// Workaround for some networks which have Linear{1.f, 0.f} between downConv and residual
		CLinearLayer* linear = nullptr;
		if( downConv == nullptr ) {
			linear = graph.GetConnectedOutput<CLinearLayer>( residual, i ).Layer;
			if( linear == nullptr || linear->GetFreeTerm() != 0.f || linear->GetMultiplier() != 1.f ) {
				continue;
			}
			downConv = graph.GetConnectedOutput<CConvLayer>( *linear, 0 ).Layer;
		}

		CLayerOutput<> blockData = graph.GetConnectedOutput<>( residual, 1 - i );
		if( downConv != nullptr && detectMNv3NonResidual( *downConv, detectedBlock )
			&& blockData == detectedBlock.InputData && graph.GetConnectedInputsCount( *downConv, 0 ) == 1 )
		{
			detectedBlock.Residual = &residual;
			graph.SelectLayer( residual );
			if( linear != nullptr ) {
				graph.SelectLayer( *linear );
			}
			return true;
		} else {
			graph.ClearSelection();
		}
	}

	return false;
}

// Checks whether the given layer is a residual layer of MobileNetV3 block
// If it is then writes data about block into detectedBlock, selects layers for future deletion and returns true
// If it isn't then returns false (in this case detectedBlock and graph's selection may be in any state)
bool CMobileNetV3Optimizer::detectMNv3NonResidual( CConvLayer& downConv, CMNv3BlockInfo& detectedBlock )
{
	return detectMNv3PostSE( downConv, detectedBlock )
		&& detectMNv3SE( detectedBlock )
		&& detectMNv3PreSE( detectedBlock );
}

// Checks whether the given layer is a downConv layer of post Squeeze-and-Excite part
// If it is then writes detected layers into detectedBlock, selects layers for future deletion and returns true
// If it isn't then returns false (in this case detectedBlock and graph's selection may be in any state)
bool CMobileNetV3Optimizer::detectMNv3PostSE( CConvLayer& downConv, CMNv3BlockInfo& detectedBlock )
{
	if( !isValid1x1Conv( &downConv ) ) {
		return false;
	}
	detectedBlock.DownConv = &downConv;
	graph.SelectLayer( downConv );

	CBaseLayer* nextLayer = graph.SelectConnectedOutput<>( downConv, 0, true ).Layer;
	if( nextLayer != nullptr && isValidBlockActivation( *nextLayer ) ) {
		detectedBlock.ChannelwisePostSEActivation = dynamic_cast<IActivationLayer&>( *nextLayer ).GetDesc();
		detectedBlock.SEMulVectorInput.Layer = graph.SelectConnectedOutput<>( *nextLayer, 0, true ).Layer;
	} else {
		detectedBlock.SEMulVectorInput.Layer = nextLayer;
	}

	if( detectedBlock.SEMulVectorInput.Layer == nullptr ) {
		return false;
	}

	return true;
}

// Checks whether the given layer is a multiplication layer of Squeeze-and-Excite
// If it is then writes detected layers into detectedBlock, selects layers for future deletion and returns true
// If it isn't then returns false (in this case detectedBlock and graph's selection may be in any state)
bool CMobileNetV3Optimizer::detectMNv3SE( CMNv3BlockInfo& detectedBlock )
{
	CBaseLayer* mulLayer = detectedBlock.SEMulVectorInput.Layer;
	if( !isValidSEMul( *mulLayer ) ) {
		return false;
	}

	for( int mulInput = 0; mulInput < 2; ++mulInput ) {
		detectedBlock.SESecondActivation = graph.GetConnectedOutput<>( *mulLayer, 1 - mulInput ).Layer;
		if( !isValidSEActivation( *detectedBlock.SESecondActivation ) ) {
			continue;
		}

		CBaseLayer* secondFc = graph.GetConnectedOutput<>( *detectedBlock.SESecondActivation, 0 ).Layer;
		if( !isValid1x1Conv( secondFc ) ) {
			return false;
		}

		CBaseLayer* firstActivation = graph.GetConnectedOutput<>( *secondFc, 0 ).Layer;
		if( !isValidSEActivation( *firstActivation ) ) {
			return false;
		}

		detectedBlock.SEFirstFc = graph.GetConnectedOutput<>( *firstActivation, 0 ).Layer;
		if( !isValid1x1Conv( detectedBlock.SEFirstFc ) ) {
			return false;
		}

		detectedBlock.SEPooling = graph.GetConnectedOutput<CGlobalMeanPoolingLayer>( *detectedBlock.SEFirstFc, 0 ).Layer;
		if( detectedBlock.SEPooling == nullptr ) {
			return false;
		}

		CLayerOutput<> poolData = graph.GetConnectedOutput<>( *detectedBlock.SEPooling, 0 );
		CLayerOutput<> seData = graph.SelectConnectedOutput<>( *mulLayer, mulInput, false );
		if( poolData == seData ) {
			detectedBlock.PreSELayer = seData.Layer;
			detectedBlock.SEMulVectorInput.Index = 1 - mulInput;
			return true;
		} else {
			return false;
		}
	}

	return false;
}

// Checks whether the given layer is a downConv layer of post Squeeze-and-Excite part
// If it is then writes detected layers into detectedBlock, selects layers for future deletion and returns true
// If it isn't then returns false (in this case detectedBlock and graph's selection may be in any state)
bool CMobileNetV3Optimizer::detectMNv3PreSE( CMNv3BlockInfo& detectedBlock )
{
	NeoAssert( detectedBlock.PreSELayer != nullptr );
	detectedBlock.Channelwise = dynamic_cast<CChannelwiseConvLayer*>( detectedBlock.PreSELayer );
	if( detectedBlock.Channelwise == nullptr && isValidBlockActivation( *detectedBlock.PreSELayer ) ) {
		detectedBlock.ChannelwisePreSEActivation =
			dynamic_cast<IActivationLayer*>( detectedBlock.PreSELayer )->GetDesc();
		detectedBlock.Channelwise = graph.SelectConnectedOutput<CChannelwiseConvLayer>( *detectedBlock.PreSELayer, 0, true ).Layer;
	}
	
	if( detectedBlock.Channelwise == nullptr || !isValidChannelwise( *detectedBlock.Channelwise ) ) {
		return false;
	}

	CBaseLayer* expandActivation = graph.SelectConnectedOutput<>( *detectedBlock.Channelwise, 0, true ).Layer;
	if( expandActivation == nullptr || !isValidBlockActivation( *expandActivation ) ) {
		return false;
	}
	detectedBlock.ExpandActivation = dynamic_cast<IActivationLayer&>( *expandActivation ).GetDesc();

	detectedBlock.ExpandConv = graph.SelectConnectedOutput<CConvLayer>( *expandActivation, 0, true ).Layer;
	if( detectedBlock.ExpandConv == nullptr || !isValid1x1Conv( detectedBlock.ExpandConv ) ) {
		return false;
	}

	detectedBlock.InputData = graph.GetConnectedOutput<>( *detectedBlock.ExpandConv, 0 );
	return detectedBlock.InputData.Layer != nullptr;
}

bool CMobileNetV3Optimizer::isValidSEMul( CBaseLayer& layer ) const
{
	if( graph.GetInputCount( layer ) != 2 || graph.GetOutputCount( layer ) != 1
		|| graph.GetConnectedInputsCount( layer, 0 ) != 1 )
	{
		return false;
	}

	COnnxEltwiseLayer* onnxMul = dynamic_cast<COnnxEltwiseLayer*>( &layer );
	if( onnxMul != nullptr && onnxMul->GetOperation() == COnnxEltwiseLayer::TOperation::Mul ) {
		return true;
	}

	// Workaround for a layer which is outside of NeoML :-(
	if( GetLayerClass( layer ) == "CnnChannelwiseMultiplicationLayer" ) {
		return true;
	}

	return false;
}

// Checks that CConvLayer meets the criteria of 1x1 convolution inside MobileNetV3 block
bool CMobileNetV3Optimizer::isValid1x1Conv( CBaseLayer* layer ) const
{
	if( dynamic_cast<CFullyConnectedLayer*>( layer ) != nullptr ) {
		// CFullyConnectedLayer is an equivalent of Conv1x1 with Stride == 1 and Padding == 0
		return true;
	}

	CConvLayer* conv = dynamic_cast<CConvLayer*>( layer );
	return conv != nullptr && graph.GetInputCount( *conv ) == 1 && conv->GetFilterHeight() == 1
		&& conv->GetFilterWidth() == 1 && conv->GetPaddingHeight() == 0 && conv->GetPaddingWidth() == 0
		&& conv->GetStrideHeight() == 1 && conv->GetStrideWidth() == 1;
}

// Checks that layer meets the criteria for activation function inside MobileNetV3 block
bool CMobileNetV3Optimizer::isValidBlockActivation( CBaseLayer& layer ) const
{
	if( graph.GetInputCount( layer ) != 1 ) {
		return false;
	}

	if( dynamic_cast<CReLULayer*>( &layer ) != nullptr || dynamic_cast<CHSwishLayer*>( &layer ) != nullptr ) {
		return true;
	}

	CLinearLayer* linear = dynamic_cast<CLinearLayer*>( &layer );
	return linear != nullptr && linear->GetFreeTerm() == 0 && linear->GetMultiplier() == 1;
}

// Checks that layer meets the criteria for activation function inside Squeeze-and-Excite
bool CMobileNetV3Optimizer::isValidSEActivation( CBaseLayer& layer ) const
{
	CReLULayer* relu = dynamic_cast<CReLULayer*>( &layer );
	CHardSigmoidLayer* hardSigmoid = dynamic_cast<CHardSigmoidLayer*>( &layer );
	if( relu == nullptr && hardSigmoid == nullptr ) {
		return false;
	}

	return graph.GetInputCount( layer ) == 1;
}

// Checks that channelwise layer meets the criteria for channelwise inside MobileNetV3 block
bool CMobileNetV3Optimizer::isValidChannelwise( CChannelwiseConvLayer& channelwise ) const
{
	return graph.GetInputCount( channelwise ) == 1
		&& channelwise.GetFilterHeight() == channelwise.GetFilterWidth()
		&& ( channelwise.GetFilterHeight() == 3 || channelwise.GetFilterHeight() == 5 )
		&& channelwise.GetDilationHeight() == 1 && channelwise.GetDilationWidth() == 1
		&& channelwise.GetPaddingHeight() == channelwise.GetPaddingWidth()
		&& channelwise.GetPaddingHeight() == channelwise.GetFilterHeight() / 2
		&& channelwise.GetStrideHeight() == channelwise.GetStrideWidth()
		&& ( channelwise.GetStrideHeight() == 1 || channelwise.GetStrideHeight() == 2 );
}

void CMobileNetV3Optimizer::optimizeDetectedBlock( const CMNv3BlockInfo& detectedBlock )
{
	// optimize pre Squeeze-and-Excite part
	CPtr<CMobileNetV3PreSEBlockLayer> preSEBlock = new CMobileNetV3PreSEBlockLayer( graph.MathEngine(),
		detectedBlock.ExpandConv->GetFilterData(), detectedBlock.ExpandConv->GetFreeTermData(),
		detectedBlock.ExpandActivation, detectedBlock.Channelwise->GetStrideHeight(),
		detectedBlock.Channelwise->GetFilterData(), detectedBlock.Channelwise->GetFreeTermData(),
		detectedBlock.ChannelwisePreSEActivation );
	preSEBlock->SetName( graph.GetUniqueName( "MobileNetV3PreSEBlock" ) );
	graph.AddLayer( *preSEBlock );
	graph.Connect( *preSEBlock, 0, *detectedBlock.InputData.Layer, detectedBlock.InputData.Index );
	graph.Connect( *detectedBlock.SEPooling, 0, *preSEBlock, 0 );

	// optimize Squeeze-and-Excite
	graph.Connect( *detectedBlock.SEFirstFc, 0, *detectedBlock.SEPooling, 0 );
	graph.Connect( *detectedBlock.SEMulVectorInput.Layer, detectedBlock.SEMulVectorInput.Index,
		*detectedBlock.SESecondActivation, 0 );

	// optimzie post Squeeze-and-Excite part
	CPtr<CMobileNetV3PostSEBlockLayer> postSEBlock = new CMobileNetV3PostSEBlockLayer( graph.MathEngine(),
		detectedBlock.ChannelwisePostSEActivation, detectedBlock.DownConv->GetFilterData(),
		detectedBlock.DownConv->GetFreeTermData() );
	postSEBlock->SetName( graph.GetUniqueName( "MobileNetV3PostSEBlock" ) );
	graph.AddLayer( *postSEBlock );
	graph.Connect( *postSEBlock, 0, *preSEBlock, 0 );
	graph.Connect( *postSEBlock, 1, *detectedBlock.SESecondActivation, 0 );
	if( detectedBlock.Residual != nullptr ) {
		graph.Connect( *postSEBlock, 2, *detectedBlock.InputData.Layer,
			detectedBlock.InputData.Index );
		graph.SwitchOutputs( *detectedBlock.Residual, 0, *postSEBlock, 0 );
	} else {
		graph.SwitchOutputs( *detectedBlock.DownConv, 0, *postSEBlock, 0 );
	}

	graph.DeleteSelectedLayers();
}

} // namespace optimization

} // namespace NeoML
