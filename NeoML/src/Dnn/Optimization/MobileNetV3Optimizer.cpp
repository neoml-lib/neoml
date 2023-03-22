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
#include <NeoML/Dnn/Layers/FullyConnectedLayer.h>
#include <NeoML/Dnn/Layers/GlobalMeanPoolingLayer.h>
#include <NeoML/Dnn/Layers/MobileNetV3BlockLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxEltwiseLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxReshapeLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxSourceHelper.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxTransformHelper.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxTransposeHelper.h>

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
	COnnxEltwiseLayer* onnxSum = dynamic_cast<COnnxEltwiseLayer*>( &residual );
	if( onnxSum == nullptr || graph.GetInputCount( residual ) != 2
		|| onnxSum->GetOperation() != COnnxEltwiseLayer::TOperation::Add )
	{
		return false;
	}

	for( int i = 0; i < 2; ++i ) {
		CConvLayer* downConvOutput = graph.GetConnectedOutput<CConvLayer>( residual, i ).Layer;
		if( downConvOutput == nullptr ) {
			continue;
		}
		CLayerOutput<> blockData = graph.GetConnectedOutput<>( residual, 1 - i );
		if( detectMNv3NonResidual( *downConvOutput, detectedBlock ) && blockData == detectedBlock.InputData ) {
			detectedBlock.Residual = &residual;
			graph.SelectLayer( residual );
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
	if( !isValid1x1Conv( downConv ) ) {
		return false;
	}
	detectedBlock.DownConv = &downConv;
	graph.SelectLayer( downConv );

	CBaseLayer* channelwiseActvation = graph.SelectConnectedOutput<>( downConv, 0, true ).Layer;
	if( channelwiseActvation == nullptr || !isValidBlockActivation( *channelwiseActvation ) ) {
		return false;
	}
	detectedBlock.ChannelwiseActivation = dynamic_cast<IActivationLayer&>( *channelwiseActvation ).GetDesc();

	detectedBlock.SEMulVectorInput.Layer = graph.SelectConnectedOutput<>( *channelwiseActvation, 0, true ).Layer;
	return true;
}

// Checks whether the given layer is a multiplication layer of Squeeze-and-Excite
// If it is then writes detected layers into detectedBlock, selects layers for future deletion and returns true
// If it isn't then returns false (in this case detectedBlock and graph's selection may be in any state)
bool CMobileNetV3Optimizer::detectMNv3SE( CMNv3BlockInfo& detectedBlock )
{
	COnnxEltwiseLayer* onnxMul = dynamic_cast<COnnxEltwiseLayer*>( detectedBlock.SEMulVectorInput.Layer );
	if( onnxMul == nullptr || onnxMul->GetOperation() != COnnxEltwiseLayer::TOperation::Mul ) {
		return false;
	}

	for( int mulInput = 0; mulInput < 2; ++mulInput ) {
		CLayerOutput<CChannelwiseConvLayer> channelwiseOutput = graph.GetConnectedOutput<CChannelwiseConvLayer>(
			*detectedBlock.SEMulVectorInput.Layer, mulInput );
		if( channelwiseOutput.Layer == nullptr ) {
			continue;
		}
		detectedBlock.Channelwise = channelwiseOutput.Layer;
		detectedBlock.SEMulVectorInput.Index = 1 - mulInput;

		COnnxTransformHelper* thirdTransform = graph.SelectConnectedOutput<COnnxTransformHelper>(
			*detectedBlock.SEMulVectorInput.Layer, detectedBlock.SEMulVectorInput.Index, true ).Layer;
		if( thirdTransform == nullptr ||
			!isValidOnnxTransform( *thirdTransform,
				{ BD_Count, BD_BatchLength, BD_Count, BD_ListSize, BD_Height, BD_Count, BD_Channels } ) )
		{
			return false;
		}

		COnnxTransposeHelper* secondTranspose = graph.SelectConnectedOutput<COnnxTransposeHelper>(
			*thirdTransform, 0, true ).Layer;
		if( secondTranspose == nullptr || !isValidOnnxTranspose( *secondTranspose, BD_BatchWidth, BD_Channels ) ) {
			return false;
		}

		COnnxTransformHelper* secondTransform = graph.SelectConnectedOutput<COnnxTransformHelper>(
			*secondTranspose, 0, true ).Layer;
		if( secondTransform == nullptr || !isValidOnnxTransform( *secondTransform,
			{ BD_BatchWidth, BD_Height, BD_Width, BD_Channels, BD_Count, BD_Count, BD_Count } ) )
		{
			return false;
		}

		COnnxReshapeLayer* secondReshape = graph.SelectConnectedOutput<COnnxReshapeLayer>(
			*secondTransform, 0, true ).Layer;
		if( secondReshape == nullptr ) {
			return false;
		}

		detectedBlock.SESecondActivation = nullptr;
		for( int reshapeInput = 0; reshapeInput < 2; ++reshapeInput ) {
			COnnxSourceHelper* shapeSource = graph.SelectConnectedOutput<COnnxSourceHelper>(
				*secondReshape, reshapeInput, true ).Layer;
			if( shapeSource == nullptr ) {
				continue;
			}
			if( !isValidOnnxSource( *shapeSource, { 1, 0, 1, 1 } ) ) {
				return false;
			}

			detectedBlock.SESecondActivation = graph.GetConnectedOutput<>( *secondReshape, 1 - reshapeInput ).Layer;
			if( !isValidSEActivation( *detectedBlock.SESecondActivation ) ) {
				return false;
			}
			break;
		}

		CFullyConnectedLayer* secondFc = graph.GetConnectedOutput<CFullyConnectedLayer>(
			*detectedBlock.SESecondActivation, 0 ).Layer;
		if( secondFc == nullptr ) {
			return false;
		}

		CBaseLayer* firstActivation = graph.GetConnectedOutput<>( *secondFc, 0 ).Layer;
		if( !isValidSEActivation( *firstActivation ) ) {
			return false;
		}

		detectedBlock.SEFirstFc = graph.GetConnectedOutput<CFullyConnectedLayer>( *firstActivation, 0 ).Layer;
		if( detectedBlock.SEFirstFc == nullptr ) {
			return false;
		}

		CLayerOutput<COnnxReshapeLayer> firstReshape = graph.SelectConnectedOutput<COnnxReshapeLayer>(
			*detectedBlock.SEFirstFc, 0, false );
		if( firstReshape.Layer == nullptr ) {
			return false;
		}

		CLayerOutput<COnnxTransformHelper> firstTransform;
		CLayerOutput<COnnxSourceHelper> firstSource;
		if( !graph.SelectBothConnectedOutputs( *firstReshape.Layer, firstTransform, firstSource, true )
			|| !isValidOnnxTransform( *firstTransform.Layer, { BD_Count, BD_BatchWidth, BD_Count, BD_ListSize, BD_Height, BD_Count, BD_Width } )
			|| !isValidOnnxSource( *firstSource.Layer, { 1, 0 } ) )
		{
			return false;
		}

		COnnxTransposeHelper* firstTranspose = graph.SelectConnectedOutput<COnnxTransposeHelper>(
			*firstTransform.Layer, 0, true ).Layer;
		if( firstTranspose == nullptr || !isValidOnnxTranspose( *firstTranspose, BD_Channels, BD_ListSize ) )
		{
			return false;
		}

		detectedBlock.SEPooling = graph.GetConnectedOutput<CGlobalMeanPoolingLayer>( *firstTranspose, 0 ).Layer;
		if( detectedBlock.SEPooling == nullptr ) {
			return false;
		}

		CLayerOutput<> poolData = graph.GetConnectedOutput<>( *detectedBlock.SEPooling, 0 );
		if( poolData == channelwiseOutput ) {
			return true;
		}
	}

	return false;
}

// Checks whether the given layer is a downConv layer of post Squeeze-and-Excite part
// If it is then writes detected layers into detectedBlock, selects layers for future deletion and returns true
// If it isn't then returns false (in this case detectedBlock and graph's selection may be in any state)
bool CMobileNetV3Optimizer::detectMNv3PreSE( CMNv3BlockInfo& detectedBlock )
{
	NeoAssert( detectedBlock.Channelwise != nullptr );
	if( !isValidChannelwise( *detectedBlock.Channelwise ) ) {
		return false;
	}

	CBaseLayer* expandActivation = graph.GetConnectedOutput<>( *detectedBlock.Channelwise, 0 ).Layer;
	if( expandActivation == nullptr || !isValidBlockActivation( *expandActivation ) ) {
		return false;
	}
	detectedBlock.ExpandActivation = dynamic_cast<IActivationLayer&>( *expandActivation ).GetDesc();

	detectedBlock.ExpandConv = graph.GetConnectedOutput<CConvLayer>( *expandActivation, 0 ).Layer;
	if( detectedBlock.ExpandConv == nullptr || !isValid1x1Conv( *detectedBlock.ExpandConv ) ) {
		return false;
	}

	detectedBlock.InputData = graph.GetConnectedOutput<>( *detectedBlock.ExpandConv, 0 );
	return detectedBlock.InputData.Layer != nullptr;
}

// Checks that CConvLayer meets the criteria of 1x1 convolution inside MobileNetV3 block
bool CMobileNetV3Optimizer::isValid1x1Conv( CConvLayer& conv ) const
{
	return graph.GetInputCount( conv ) == 1 && conv.GetFilterHeight() == 1 && conv.GetFilterWidth() == 1
		&& conv.GetPaddingHeight() == 0 && conv.GetPaddingWidth() == 0 && conv.GetStrideHeight() == 1
		&& conv.GetStrideWidth() == 1;
}

// Checks that layer meets the criteria for activation function inside MobileNetV3 block
bool CMobileNetV3Optimizer::isValidBlockActivation( CBaseLayer& layer ) const
{
	return ( dynamic_cast<CReLULayer*>( &layer ) != nullptr || dynamic_cast<CHSwishLayer*>( &layer ) != nullptr )
		&& graph.GetInputCount( layer ) == 1;
}

// Checks that ONNX transform has the expected rules
bool CMobileNetV3Optimizer::isValidOnnxTransform( COnnxTransformHelper& transform,
	std::initializer_list<TBlobDim> expectedRules ) const
{
	NeoAssert( expectedRules.size() == 7 );
	auto it = expectedRules.begin();
	for( int i = 0; i < 7; ++i ) {
		if( *it != transform.GetRule( TBlobDim( i ) ) ) {
			return false;
		}
		++it;
	}
	return graph.GetInputCount( transform ) == 1;
}

// Checks that ONNX transposes swaps the expected dimensions
bool CMobileNetV3Optimizer::isValidOnnxTranspose( COnnxTransposeHelper& transpose, TBlobDim firstDim, TBlobDim secondDim ) const
{
	TBlobDim dim0, dim1;
	transpose.GetDims( dim0, dim1 );
	return graph.GetInputCount( transpose ) == 1
		&& ( ( dim0 == firstDim && dim1 == secondDim ) || ( dim1 == firstDim && dim0 == secondDim ) );
}

// Checks that ONNX source contains the expected data
bool CMobileNetV3Optimizer::isValidOnnxSource( COnnxSourceHelper& source,
	std::initializer_list<int> expectedData ) const
{
	CPtr<CDnnBlob> blob = source.Blob();
	if( blob->GetDataType() != CT_Int || blob->GetDataSize() != expectedData.size() ) {
		return false;
	}
	CDnnBlobBuffer<int> buff( *blob, TDnnBlobBufferAccess::Read );
	auto it = expectedData.begin();
	for( int i = 0; i < buff.Size(); ++i ) {
		if( *it != 0 && buff[i] != *it ) {
			return false;
		}
		++it;
	}
	return graph.GetInputCount( source ) == 0;
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
	// optimize Squeeze-and-Excite
	graph.Connect( *detectedBlock.SEFirstFc, 0, *detectedBlock.SEPooling, 0 );
	graph.Connect( *detectedBlock.SEMulVectorInput.Layer, detectedBlock.SEMulVectorInput.Index,
		*detectedBlock.SESecondActivation, 0 );

	// optimzie post Squeeze-and-Excite part
	CPtr<CMobileNetV3PostSEBlockLayer> postSEBlock = new CMobileNetV3PostSEBlockLayer( graph.MathEngine(),
		detectedBlock.ChannelwiseActivation, detectedBlock.DownConv->GetFilterData(),
		!detectedBlock.DownConv->IsZeroFreeTerm() ? detectedBlock.DownConv->GetFreeTermData() : nullptr );
	postSEBlock->SetName( graph.GetUniqueName( "MobileNetV3PostSEBlock" ) );
	graph.AddLayer( *postSEBlock );
	graph.Connect( *postSEBlock, 0, *detectedBlock.Channelwise, 0 );
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
