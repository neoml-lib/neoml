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

#include "SqueezeAndExciteOptimizer.h"

#include <NeoML/Dnn/DnnOptimization.h>
#include <NeoML/Dnn/Layers/ConvLayer.h>
#include <NeoML/Dnn/Layers/FullyConnectedLayer.h>
#include <NeoML/Dnn/Layers/GlobalMeanPoolingLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxEltwiseLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxReshapeLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxSourceHelper.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxTransformHelper.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxTransposeHelper.h>

namespace NeoOnnx {

namespace optimization {

int CSqueezeAndExciteOptimizer::Apply()
{
	return optimizeSEBlocks();
}

int CSqueezeAndExciteOptimizer::optimizeSEBlocks()
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
		CSEBlockInfo detectedBlock;
		if( !detectSqueezAndExcite( *layer, detectedBlock ) ) {
			continue;
		}

		graph.Connect( *detectedBlock.SEFirstFc, 0, *detectedBlock.SEPooling, 0 );
		graph.Connect( *detectedBlock.SEMulVectorInput.Layer, detectedBlock.SEMulVectorInput.Index,
			*detectedBlock.SESecondActivation, 0 );
		graph.DeleteSelectedLayers();

		++blocksOptimized;
	}
	graph.ClearSelection();

	NeoAssert( graph.SelectionSize() == 0 );

	return blocksOptimized;
}

// Checks whether the given layer is a multiplication layer of Squeeze-and-Excite
// If it is then writes detected layers into detectedBlock, selects layers for future deletion and returns true
// If it isn't then returns false (in this case detectedBlock and graph's selection may be in any state)
bool CSqueezeAndExciteOptimizer::detectSqueezAndExcite( CBaseLayer& mulLayer, CSEBlockInfo& detectedBlock )
{
	if( !isValidMul( mulLayer ) ) {
		return false;
	}

	for( int mulInput = 0; mulInput < 2; ++mulInput ) {
		COnnxTransformHelper* thirdTransform = graph.SelectConnectedOutput<COnnxTransformHelper>( mulLayer, mulInput,
			false ).Layer;
		if( thirdTransform == nullptr ||
			!isValidOnnxTransform( *thirdTransform,
				{ BD_Count, BD_BatchLength, BD_Count, BD_ListSize, BD_Height, BD_Count, BD_Channels } ) )
		{
			continue;
		}

		detectedBlock.InputData = graph.GetConnectedOutput<>( mulLayer, 1 - mulInput );
		detectedBlock.SEMulVectorInput.Layer = &mulLayer;
		detectedBlock.SEMulVectorInput.Index = mulInput;

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
		if( firstTranspose == nullptr || !isValidOnnxTranspose( *firstTranspose, BD_Channels, BD_ListSize ) ) {
			return false;
		}

		detectedBlock.SEPooling = graph.GetConnectedOutput<CGlobalMeanPoolingLayer>( *firstTranspose, 0 ).Layer;
		if( detectedBlock.SEPooling == nullptr ) {
			return false;
		}

		CLayerOutput<> poolData = graph.GetConnectedOutput<>( *detectedBlock.SEPooling, 0 );
		if( poolData == detectedBlock.InputData ) {
			return true;
		}
	}

	return false;
}

bool CSqueezeAndExciteOptimizer::isValidMul( CBaseLayer& layer ) const
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

// Checks that layer meets the criteria for activation function inside Squeeze-and-Excite
bool CSqueezeAndExciteOptimizer::isValidSEActivation( CBaseLayer& layer ) const
{
	CReLULayer* relu = dynamic_cast<CReLULayer*>( &layer );
	CHardSigmoidLayer* hardSigmoid = dynamic_cast<CHardSigmoidLayer*>( &layer );
	if( relu == nullptr && hardSigmoid == nullptr ) {
		return false;
	}

	return graph.GetInputCount( layer ) == 1;
}

// Checks that CConvLayer meets the criteria of 1x1 convolution inside MobileNetV3 block
bool CSqueezeAndExciteOptimizer::isValid1x1Conv( CBaseLayer* layer ) const
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

// Checks that ONNX transform has the expected rules
bool CSqueezeAndExciteOptimizer::isValidOnnxTransform( COnnxTransformHelper& transform,
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
bool CSqueezeAndExciteOptimizer::isValidOnnxTranspose( COnnxTransposeHelper& transpose, TBlobDim firstDim, TBlobDim secondDim ) const
{
	TBlobDim dim0, dim1;
	transpose.GetDims( dim0, dim1 );
	return graph.GetInputCount( transpose ) == 1
		&& ( ( dim0 == firstDim && dim1 == secondDim ) || ( dim1 == firstDim && dim0 == secondDim ) );
}

// Checks that ONNX source contains the expected data
bool CSqueezeAndExciteOptimizer::isValidOnnxSource( COnnxSourceHelper& source,
	std::initializer_list<int> expectedData ) const
{
	CPtr<CDnnBlob> blob = source.Blob();
	if( blob->GetDataType() != CT_Int || blob->GetDataSize() != static_cast<int>( expectedData.size() ) ) {
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

} // namespace optimization

} // namespace NeoOnnx
