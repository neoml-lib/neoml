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

#include <NeoML/Dnn/DnnSimpleAPI.h>

namespace NeoML {
namespace SimpleAPI {
//////////////////////////////////////////////////////////////////////////////////////////

CSourceLayer* Source( CDnn& network, const char* name )
{
	CPtr<CSourceLayer> source = new CSourceLayer( network.GetMathEngine() );
	source->SetName( name );
	network.AddLayer( *source );
	return source;
}

CSinkLayer* Sink( const CLayerOutput& layer, const char* name )
{
	NeoAssert( layer.IsValid() );

	CDnn* network = layer.Layer->GetDnn();

	CPtr<CSinkLayer> sink = new CSinkLayer( network->GetMathEngine() );
	sink->SetName( name );

	network->AddLayer( *sink );
	sink->Connect( 0, *layer.Layer, layer.OutputNumber );

	return sink;
}

//////////////////////////////////////////////////////////////////////////////////////////

CLayerWrapper<CDropoutLayer> Dropout( float dropoutRate,
	bool isSpatial, bool isBatchwise )
{
	return CLayerWrapper<CDropoutLayer>( "Dropout", [=]( CDropoutLayer* result ) {
		result->SetSpatial( isSpatial );
		result->SetBatchwise( isBatchwise );
		result->SetDropoutRate( dropoutRate );
	} );
}

CLayerWrapper<CReLULayer> Relu()
{
	return CLayerWrapper<CReLULayer>( "Relu" );
}

CLayerWrapper<CGELULayer> Gelu()
{
	return CLayerWrapper<CGELULayer>( "Gelu" );
}

CLayerWrapper<CLinearLayer> Linear( float multiplier, float freeTerm )
{
	return CLayerWrapper<CLinearLayer>( "Linear", [=]( CLinearLayer* result ) {
		result->SetMultiplier( multiplier );
		result->SetFreeTerm( freeTerm );
	} );
}

CLayerWrapper<CELULayer> Elu( float alpha )
{
	return CLayerWrapper<CELULayer>( "Elu", [=]( CELULayer* result ) {
		result->SetAlpha( alpha );
	} );
}

CLayerWrapper<CLeakyReLULayer> LeakyRelu( float alpha )
{
	return CLayerWrapper<CLeakyReLULayer>( "LeakyRelu", [=]( CLeakyReLULayer* result ) {
		result->SetAlpha( alpha );
	} );
}

CLayerWrapper<CAbsLayer> Abs()
{
	return CLayerWrapper<CAbsLayer>( "Abs" );
}

CLayerWrapper<CSigmoidLayer> Sigmoid()
{
	return CLayerWrapper<CSigmoidLayer>( "Sigmoid" );
}

CLayerWrapper<CTanhLayer> Tanh()
{
	return CLayerWrapper<CTanhLayer>( "Tanh" );
}

CLayerWrapper<CHardTanhLayer> HardTanh()
{
	return CLayerWrapper<CHardTanhLayer>( "HardTanh" );
}

CLayerWrapper<CHardSigmoidLayer> HardSigmoid( float slope, float bias )
{
	return CLayerWrapper<CHardSigmoidLayer>( "HardSigmoid", [=]( CHardSigmoidLayer* result ) {
		result->SetSlope( slope );
		result->SetBias( bias );
	} );
}

CLayerWrapper<CPowerLayer> Power( float exponent )
{
	return CLayerWrapper<CPowerLayer>( "Power", [=]( CPowerLayer* result ) {
		result->SetExponent( exponent );
	} );
}

CLayerWrapper<CHSwishLayer> HSwish()
{
	return CLayerWrapper<CHSwishLayer>( "HSwish" );
}

CLayerWrapper<CFullyConnectedLayer> FullyConnected( int numberOfElements, bool zeroFreeTerm )
{
	return CLayerWrapper<CFullyConnectedLayer>( "FullyConnected", [=]( CFullyConnectedLayer* result ) {
		result->SetNumberOfElements( numberOfElements );
		result->SetZeroFreeTerm( zeroFreeTerm );
	} );
}

CLayerWrapper<CBinaryCrossEntropyLossLayer> BinaryCrossEntropyLoss(
	float positiveWeight, float lossWeight )
{
	return CLayerWrapper<CBinaryCrossEntropyLossLayer>( "BinaryCrossEntropyLoss", [=]( CBinaryCrossEntropyLossLayer* result ) {
		result->SetPositiveWeight( positiveWeight );
		result->SetLossWeight( lossWeight );
	} );
}

CLayerWrapper<CCrossEntropyLossLayer> CrossEntropyLoss(
	bool isSoftmaxApplied, float lossWeight )
{
	return CLayerWrapper<CCrossEntropyLossLayer>( "CrossEntropyLoss", [=]( CCrossEntropyLossLayer* result ) {
		result->SetApplySoftmax( isSoftmaxApplied );
		result->SetLossWeight( lossWeight );
	} );
}

CLayerWrapper<CEuclideanLossLayer> EuclideanLoss( float lossWeight )
{
	return CLayerWrapper<CEuclideanLossLayer>( "EuclideanLoss", [=]( CEuclideanLossLayer* result ) {
		result->SetLossWeight( lossWeight );
	} );
}

CLayerWrapper<CHingeLossLayer> HingeLoss( float lossWeight )
{
	return CLayerWrapper<CHingeLossLayer>( "HingeLoss", [=]( CHingeLossLayer* result ) {
		result->SetLossWeight( lossWeight );
	} );
}

CLayerWrapper<CSquaredHingeLossLayer> SquaredHingeLoss( float lossWeight )
{
	return CLayerWrapper<CSquaredHingeLossLayer>( "SquaredHingeLoss", [=]( CSquaredHingeLossLayer* result ) {
		result->SetLossWeight( lossWeight );
	} );
}

CLayerWrapper<CBinaryFocalLossLayer> BinaryFocalLoss( float focalForce, float lossWeight )
{
	return CLayerWrapper<CBinaryFocalLossLayer>( "BinaryFocalLoss", [=]( CBinaryFocalLossLayer* result ) {
		result->SetFocalForce( focalForce );
		result->SetLossWeight( lossWeight );
	} );
}

CLayerWrapper<CCenterLossLayer> CenterLoss(
	int numberOfClasses, float classCentersConvergenceRate, float lossWeight )
{
	return CLayerWrapper<CCenterLossLayer>( "CenterLoss", [=]( CCenterLossLayer* result ) {
		result->SetNumberOfClasses( numberOfClasses );
		result->SetClassCentersConvergenceRate( classCentersConvergenceRate );
		result->SetLossWeight( lossWeight );
	} );
}

CLayerWrapper<CMultiHingeLossLayer> MultiHingeLoss( float lossWeight )
{
	return CLayerWrapper<CMultiHingeLossLayer>( "MultiHingeLoss", [=]( CMultiHingeLossLayer* result ) {
		result->SetLossWeight( lossWeight );
	} );
}

CLayerWrapper<CMultiSquaredHingeLossLayer> MultiSquaredHingeLoss( float lossWeight )
{
	return CLayerWrapper<CMultiSquaredHingeLossLayer>( "MultiSquaredHingeLoss", [=]( CMultiSquaredHingeLossLayer* result ) {
		result->SetLossWeight( lossWeight );
	} );
}

CLayerWrapper<CPositionalEmbeddingLayer> PositionalEmbedding(
	CPositionalEmbeddingLayer::TPositionalEmbeddingType type )
{
	return CLayerWrapper<CPositionalEmbeddingLayer>( "PositionalEmbedding", [=]( CPositionalEmbeddingLayer* result ) {
		result->SetType( type );
	} );
}

CLayerWrapper<CSoftmaxLayer> Softmax(
	CSoftmaxLayer::TNormalizationArea normalizationArea )
{
	return CLayerWrapper<CSoftmaxLayer>( "Softmax", [=]( CSoftmaxLayer* result ) {
		result->SetNormalizationArea( normalizationArea );
	} );
}

CLayerWrapper<CAddToObjectLayer> AddToObject()
{
	return CLayerWrapper<CAddToObjectLayer>( "AddToObject" );
}

static void applyTransformRule( CTransformLayer* transformLayer, TBlobDim dim, int value )
{
	assert( transformLayer != 0 );
	assert( value > 0 || value == TR_Remainder || value == TR_Same );

	if( value == TR_Same ) {
		transformLayer->SetDimensionRule( dim, CTransformLayer::O_Multiply, 1 );
	} else if( value == TR_Remainder ) {
		transformLayer->SetDimensionRule( dim, CTransformLayer::O_Remainder, 0 );
	} else {
		transformLayer->SetDimensionRule( dim, CTransformLayer::O_SetSize, value );
	}
}

static bool isTransformParamCorrect( int param )
{
	return param > 0 || param == TR_Remainder || param == TR_Same;
}

CLayerWrapper<CTransformLayer> Transform( int batchLength, int batchWidth,
	int listSize, int height, int width, int depth, int channel )
{
	NeoAssert( isTransformParamCorrect( batchLength ) );
	NeoAssert( isTransformParamCorrect( batchWidth ) );
	NeoAssert( isTransformParamCorrect( listSize ) );
	NeoAssert( isTransformParamCorrect( width ) );
	NeoAssert( isTransformParamCorrect( height ) );
	NeoAssert( isTransformParamCorrect( depth ) );
	NeoAssert( isTransformParamCorrect( channel ) );

	return CLayerWrapper<CTransformLayer>( "Transform", [=]( CTransformLayer* result ) {
		const CTransformLayer::CDimensionRule sameRule(
			CTransformLayer::O_Multiply, 1 );
		const CTransformLayer::CDimensionRule remainderRule(
			CTransformLayer::O_Remainder, 0 );

		applyTransformRule( result, BD_BatchLength, batchLength );
		applyTransformRule( result, BD_BatchWidth, batchWidth );
		applyTransformRule( result, BD_ListSize, listSize );
		applyTransformRule( result, BD_Height, height );
		applyTransformRule( result, BD_Width, width );
		applyTransformRule( result, BD_Depth, depth );
		applyTransformRule( result, BD_Channels, channel );
	} );
}

CLayerWrapper<CTransposeLayer> Transpose( TBlobDim d1, TBlobDim d2 )
{
	return CLayerWrapper<CTransposeLayer>( "Transpose", [=]( CTransposeLayer* result ) {
		result->SetTransposedDimensions( d1, d2 );
	} );
}

CLayerWrapper<CConcatBatchWidthLayer> ConcatBatchWidth()
{
	return CLayerWrapper<CConcatBatchWidthLayer>( "ConcatBatchWidth" );
}

CLayerWrapper<CConcatHeightLayer> ConcatHeight()
{
	return CLayerWrapper<CConcatHeightLayer>( "ConcatHeight" );
}

CLayerWrapper<CConcatWidthLayer> ConcatWidth()
{
	return CLayerWrapper<CConcatWidthLayer>( "ConcatWidth" );
}

CLayerWrapper<CConcatChannelsLayer> ConcatChannels()
{
	return CLayerWrapper<CConcatChannelsLayer>( "ConcatChannels" );
}

CLayerWrapper<CConcatDepthLayer> ConcatDepth()
{
	return CLayerWrapper<CConcatDepthLayer>( "ConcatDepth" );
}

CLayerWrapper<CBatchNormalizationLayer> BatchNormalization(
	bool isChannelBased, bool isZeroFreeTerm, float slowConvergenceRate )
{
	return CLayerWrapper<CBatchNormalizationLayer>( "BatchNormalization", [=]( CBatchNormalizationLayer* result ) {
		result->SetChannelBased( isChannelBased );
		result->SetZeroFreeTerm( isZeroFreeTerm );
		result->SetSlowConvergenceRate( slowConvergenceRate );
	} );
}

CLayerWrapper<CObjectNormalizationLayer> ObjectNormalization( float epsilon )
{
	return CLayerWrapper<CObjectNormalizationLayer>( "ObjectNormalization", [=]( CObjectNormalizationLayer* result ) {
		result->SetEpsilon( epsilon );
	} );
}

CLayerWrapper<CMultiheadAttentionLayer> MultiheadAttention(
	int headCount, int hiddenSize, int outputSize, float dropoutRate )
{
	return CLayerWrapper<CMultiheadAttentionLayer>( "MultiheadAttention", [=]( CMultiheadAttentionLayer* result ) {
		result->SetHeadCount( headCount );
		result->SetHiddenSize( hiddenSize );
		result->SetOutputSize( outputSize );
		result->SetDropoutRate( dropoutRate );
	} );
}

CLayerWrapper<CMatrixMultiplicationLayer> MatrixMultiplication()
{
	return CLayerWrapper<CMatrixMultiplicationLayer>( "MatrixMultiplication" );
}

CLayerWrapper<CEltwiseSumLayer> Sum()
{
	return CLayerWrapper<CEltwiseSumLayer>( "Sum" );
}

CLayerWrapper<CEltwiseMulLayer> Mul()
{
	return CLayerWrapper<CEltwiseMulLayer>( "Mul" );
}

CLayerWrapper<CEltwiseNegMulLayer> NegMul()
{
	return CLayerWrapper<CEltwiseNegMulLayer>( "NegMul" );
}

CLayerWrapper<CEltwiseMaxLayer> Max()
{
	return CLayerWrapper<CEltwiseMaxLayer>( "Max" );
}

CLayerWrapper<CGlobalMaxPoolingLayer> GlobalMaxPooling( int maxCount )
{
	return CLayerWrapper<CGlobalMaxPoolingLayer>( "", [=]( CGlobalMaxPoolingLayer* result ) {
		result->SetMaxCount( maxCount );
	} );
}

CLayerWrapper<CGlobalMeanPoolingLayer> GlobalMeanPooling()
{
	return CLayerWrapper<CGlobalMeanPoolingLayer>( "GlobalMeanPooling" );
}

CLayerWrapper<CLstmLayer> Lstm(
	int hiddenSize, float dropoutRate, bool isInCompatibilityMode )
{
	return CLayerWrapper<CLstmLayer>( "", [=]( CLstmLayer* result ) {
		result->SetHiddenSize( hiddenSize );
		result->SetDropoutRate( dropoutRate );
		result->SetCompatibilityMode( isInCompatibilityMode );
	} );
}

CLayerWrapper<CGruLayer> Gru( int hiddenSize )
{
	return CLayerWrapper<CGruLayer>( "Gru", [=]( CGruLayer* result ) {
		result->SetHiddenSize( hiddenSize );
	} );
}

CLayerWrapper<CSubSequenceLayer> SubSequence( int startPos, int length )
{
	return CLayerWrapper<CSubSequenceLayer>{ "SubSequence", [=]( CSubSequenceLayer* result ) {
		result->SetStartPos( startPos );
		result->SetLength( length );
	} };
}

CLayerWrapper<CSubSequenceLayer> ReverseSubSequence()
{
	return CLayerWrapper<CSubSequenceLayer>( "ReverseSubSequence", [=]( CSubSequenceLayer* result ) {
		result->SetReverse();
	} );
}

CLayerWrapper<CArgmaxLayer> Argmax( TBlobDim dim )
{
	return CLayerWrapper<CArgmaxLayer>( "Argmax", [=]( CArgmaxLayer* result ) {
		result->SetDimension( dim );
	} );
}

CLayerWrapper<CConvLayer> Conv( int filterCount,
	const CConvAxisParams& heightParams, const CConvAxisParams& widthParams,
	bool isZeroFreeTerm )
{
	return CLayerWrapper<CConvLayer>( "Conv", [=]( CConvLayer* result ) {

		result->SetFilterCount( filterCount );

		result->SetFilterHeight( heightParams.Size );
		result->SetPaddingHeight( heightParams.Padding );
		result->SetStrideWidth( heightParams.Stride );
		result->SetDilationHeight( heightParams.Dilation );

		result->SetFilterWidth( widthParams.Size );
		result->SetPaddingWidth( widthParams.Padding );
		result->SetStrideHeight( widthParams.Stride );
		result->SetDilationWidth( widthParams.Dilation );

		result->SetZeroFreeTerm( isZeroFreeTerm );
	} );
}

CLayerWrapper<CChannelwiseConvLayer> ChannelwiseConv( int filterCount,
	const CConvAxisParams& heightParams, const CConvAxisParams& widthParams,
	bool isZeroFreeTerm )
{
	return CLayerWrapper<CChannelwiseConvLayer>( "ChannelwiseConv", [=]( CChannelwiseConvLayer* result ) {
		result->SetFilterCount( filterCount );

		result->SetFilterHeight( heightParams.Size );
		result->SetPaddingHeight( heightParams.Padding );
		result->SetStrideWidth( heightParams.Stride );
		result->SetDilationHeight( heightParams.Dilation );

		result->SetFilterWidth( widthParams.Size );
		result->SetPaddingWidth( widthParams.Padding );
		result->SetStrideHeight( widthParams.Stride );
		result->SetDilationWidth( widthParams.Dilation );

		result->SetZeroFreeTerm( isZeroFreeTerm );
	} );
}

CLayerWrapper<CTimeConvLayer> TimeConv( int filterCount, int size, int padding,
	int stride, int dilation )
{
	return CLayerWrapper<CTimeConvLayer>( "ChannelwiseConv", [=]( CTimeConvLayer* result ) {
		result->SetFilterCount( filterCount );
		result->SetFilterSize( size );
		result->SetPadding( padding );
		result->SetStride( stride );
		result->SetDilation( dilation );
	} );
}

CLayerWrapper<CMultichannelLookupLayer> MultichannelLookup(
	const CArray<CLookupDimension>& lookupDimensions, bool useFrameworkLearning )
{
	return CLayerWrapper<CMultichannelLookupLayer>( "MultichannelLookupLayer", [=, &lookupDimensions]( CMultichannelLookupLayer* result ) {
		result->SetDimensions( lookupDimensions );
		result->SetUseFrameworkLearning( useFrameworkLearning );
	} );
}

CLayerWrapper<CMultichannelLookupLayer> Embeddings( int count, int size )
{
	return CLayerWrapper<CMultichannelLookupLayer>( "MultichannelLookupLayer", [=]( CMultichannelLookupLayer* result ) {
		CArray<CLookupDimension> lookupDimensions;
		lookupDimensions.Add( CLookupDimension( count, size ) );
		result->SetDimensions( lookupDimensions );
		result->SetUseFrameworkLearning( true );
	} );
}

CLayerWrapper<CUpsampling2DLayer> Upsampling2d( int heightCopyCount, int widthCopyCount )
{
	return CLayerWrapper<CUpsampling2DLayer>( "Upsampling2d", [=]( CUpsampling2DLayer* result ) {
		result->SetHeightCopyCount( heightCopyCount );
		result->SetWidthCopyCount( widthCopyCount );
	} );
}

CLayerWrapper<CSplitBatchWidthLayer> SplitBatchWidth( const CArray<int>& outputCounts )
{
	return CLayerWrapper<CSplitBatchWidthLayer>( "SplitBatchWidth", [&outputCounts]( CSplitBatchWidthLayer* result ) {
		result->SetOutputCounts( outputCounts );
	} );
}

CLayerWrapper<CSplitBatchWidthLayer> SplitBatchWidth( int output0, int output1, int output2 )
{
	return CLayerWrapper<CSplitBatchWidthLayer>( "SplitBatchWidth", [=]( CSplitBatchWidthLayer* result ) {
		if( output1 == 0 ) {
			result->SetOutputCounts2( output0 );
		} else if( output2 == 0 ) {
			result->SetOutputCounts3( output0, output1 );
		} else {
			result->SetOutputCounts4( output0, output1, output2 );
		}
	} );
}

CLayerWrapper<CSplitChannelsLayer> SplitChannels( const CArray<int>& outputCounts )
{
	return CLayerWrapper<CSplitChannelsLayer>( "SplitChannels", [&outputCounts]( CSplitChannelsLayer* result ) {
		result->SetOutputCounts( outputCounts );
	} );
}

CLayerWrapper<CSplitChannelsLayer> SplitChannels( int output0, int output1, int output2 )
{
	return CLayerWrapper<CSplitChannelsLayer>( "SplitChannels", [=]( CSplitChannelsLayer* result ) {
		if( output1 == 0 ) {
			result->SetOutputCounts2( output0 );
		} else if( output2 == 0 ) {
			result->SetOutputCounts3( output0, output1 );
		} else {
			result->SetOutputCounts4( output0, output1, output2 );
		}
	} );
}

CLayerWrapper<CSplitDepthLayer> SplitDepth( const CArray<int>& outputCounts )
{
	return CLayerWrapper<CSplitDepthLayer>( "SplitDepth", [&outputCounts]( CSplitDepthLayer* result ) {
		result->SetOutputCounts( outputCounts );
	} );
}

CLayerWrapper<CSplitDepthLayer> SplitDepth( int output0, int output1, int output2 )
{
	return CLayerWrapper<CSplitDepthLayer>( "SplitDepth", [=]( CSplitDepthLayer* result ) {
		if( output1 == 0 ) {
			result->SetOutputCounts2( output0 );
		} else if( output2 == 0 ) {
			result->SetOutputCounts3( output0, output1 );
		} else {
			result->SetOutputCounts4( output0, output1, output2 );
		}
	} );
}

CLayerWrapper<CSplitHeightLayer> SplitHeight( const CArray<int>& outputCounts )
{
	return CLayerWrapper<CSplitHeightLayer>( "SplitHeight", [&outputCounts]( CSplitHeightLayer* result ) {
		result->SetOutputCounts( outputCounts );
	} );
}

CLayerWrapper<CSplitHeightLayer> SplitHeight( int output0, int output1, int output2 )
{
	return CLayerWrapper<CSplitHeightLayer>( "SplitHeight", [=]( CSplitHeightLayer* result ) {
		if( output1 == 0 ) {
			result->SetOutputCounts2( output0 );
		} else if( output2 == 0 ) {
			result->SetOutputCounts3( output0, output1 );
		} else {
			result->SetOutputCounts4( output0, output1, output2 );
		}
	} );
}

CLayerWrapper<CSplitWidthLayer> SplitWidth( const CArray<int>& outputCounts )
{
	return CLayerWrapper<CSplitWidthLayer>( "SplitWidth", [&outputCounts]( CSplitWidthLayer* result ) {
		result->SetOutputCounts( outputCounts );
	} );
}

CLayerWrapper<CSplitWidthLayer> SplitWidth( int output0, int output1, int output2 )
{
	return CLayerWrapper<CSplitWidthLayer>( "SplitWidth", [=]( CSplitWidthLayer* result ) {
		if( output1 == 0 ) {
			result->SetOutputCounts2( output0 );
		} else if( output2 == 0 ) {
			result->SetOutputCounts3( output0, output1 );
		} else {
			result->SetOutputCounts4( output0, output1, output2 );
		}
	} );
}

CLayerWrapper<C3dConvLayer> Conv3d( int filterCount,
	const CConvAxisParams& heightParams, const CConvAxisParams& widthParams,
	const CConvAxisParams& depthParams, bool isZeroFreeTerm )
{
	return CLayerWrapper<C3dConvLayer>( "Conv3d", [=]( C3dConvLayer* result ) {
		result->SetFilterCount( filterCount );

		result->SetFilterHeight( heightParams.Size );
		result->SetPaddingHeight( heightParams.Padding );
		result->SetStrideWidth( heightParams.Stride );
		result->SetDilationHeight( heightParams.Dilation );

		result->SetFilterWidth( widthParams.Size );
		result->SetPaddingWidth( widthParams.Padding );
		result->SetStrideHeight( widthParams.Stride );
		result->SetDilationWidth( widthParams.Dilation );

		result->SetFilterDepth( depthParams.Size );
		result->SetPaddingDepth( depthParams.Padding );
		result->SetStrideDepth( depthParams.Stride );
		// layer has no dilation

		result->SetZeroFreeTerm( isZeroFreeTerm );
	} );
}

CLayerWrapper<CTransposedConvLayer> TransposedConv( int filterCount,
	const CConvAxisParams& heightParams, const CConvAxisParams& widthParams,
	bool isZeroFreeTerm )
{
	return CLayerWrapper<CTransposedConvLayer>( "TransposedConv", [=]( CTransposedConvLayer* result ) {
		result->SetFilterCount( filterCount );

		result->SetFilterHeight( heightParams.Size );
		result->SetPaddingHeight( heightParams.Padding );
		result->SetStrideWidth( heightParams.Stride );
		result->SetDilationHeight( heightParams.Dilation );

		result->SetFilterWidth( widthParams.Size );
		result->SetPaddingWidth( widthParams.Padding );
		result->SetStrideHeight( widthParams.Stride );
		result->SetDilationWidth( widthParams.Dilation );

		result->SetZeroFreeTerm( isZeroFreeTerm );
	} );
}

CLayerWrapper<CMaxOverTimePoolingLayer> MaxOverTimePooling(
	int filterLength, int strideLength )
{
	return CLayerWrapper<CMaxOverTimePoolingLayer>( "MaxOverTimePooling", [=]( CMaxOverTimePoolingLayer* result ) {
		result->SetFilterLength( filterLength );
		result->SetStrideLength( strideLength );
	} );
}

CLayerWrapper<C3dMaxPoolingLayer> Pooling3dMax( int filterHeight, int filterWidth, int filterDepth,
	int strideHeight, int strideWidth, int strideDepth )
{
	return CLayerWrapper<C3dMaxPoolingLayer>( "Pooling3D", [=]( C3dMaxPoolingLayer* result ) {
		result->SetFilterHeight( filterHeight );
		result->SetFilterWidth( filterWidth );
		result->SetFilterDepth( filterDepth );
		result->SetStrideHeight( strideHeight );
		result->SetStrideWidth( strideWidth );
		result->SetStrideDepth( strideDepth );
	} );
}

CLayerWrapper<C3dMeanPoolingLayer> Pooling3dMean( int filterHeight, int filterWidth, int filterDepth,
	int strideHeight, int strideWidth, int strideDepth )
{
	return CLayerWrapper<C3dMeanPoolingLayer>( "Pooling3dMean", [=]( C3dMeanPoolingLayer* result ) {
		result->SetFilterHeight( filterHeight );
		result->SetFilterWidth( filterWidth );
		result->SetFilterDepth( filterDepth );
		result->SetStrideHeight( strideHeight );
		result->SetStrideWidth( strideWidth );
		result->SetStrideDepth( strideDepth );
	} );
}

NEOML_API CLayerWrapper<NeoML::CAccumulativeLookupLayer> AccumulativeLookup(
	int count, int size )
{
	return CLayerWrapper<CAccumulativeLookupLayer>( "AccumulativeLookup", [=]( CAccumulativeLookupLayer* result ) {
		return result->SetDimension( CLookupDimension( count, size ) );
	} );
}

CLayerWrapper<CAccuracyLayer> Accuracy()
{
	return CLayerWrapper<CAccuracyLayer>( "Accuracy" );
}

CLayerWrapper<CConfusionMatrixLayer> ConfusionMatrix()
{
	return CLayerWrapper<CConfusionMatrixLayer>( "ConfusionMatrix" );
}

CLayerWrapper<CAttentionDecoderLayer> AttentionDecoder(
	TAttentionScore score, int outObjectSize, int outSeqLen, int hiddenSize )
{
	return CLayerWrapper<CAttentionDecoderLayer>( "AttentionDecoder", [=]( CAttentionDecoderLayer* result ) {
		result->SetAttentionScore( score );
		result->SetOutputObjectSize( outObjectSize );
		result->SetOutputSequenceLen( outSeqLen );
		result->SetHiddenLayerSize( hiddenSize );
	} );
}

CLayerWrapper<CCtcLossLayer> CtcLoss( int blankLabel, bool allowBlankLabelSkip,
	float lossWeight )
{
	return CLayerWrapper<CCtcLossLayer>( "CtcLoss", [=]( CCtcLossLayer* result ) {
		result->SetBlankLabel( blankLabel );
		result->SetAllowBlankLabelSkips( allowBlankLabelSkip );
		result->SetLossWeight( lossWeight );
	} );
}

CLayerWrapper<CDotProductLayer> DotProduct()
{
	return CLayerWrapper<CDotProductLayer>( "DotProduct" );
}

CLayerWrapper<CEnumBinarizationLayer> EnumBinarization( int enumSize )
{
	return CLayerWrapper<CEnumBinarizationLayer>( "EnumBinarization", [=]( CEnumBinarizationLayer* result ) {
		result->SetEnumSize( enumSize );
	} );
}

CLayerWrapper<CFullyConnectedSourceLayer> FullyConnectedSource( TBlobType labelType,
	int batchSize, int maxBatchCount, IProblem* problem )
{
	return CLayerWrapper<CFullyConnectedSourceLayer>( "FullyConnectedSource", [=, &problem]( CFullyConnectedSourceLayer* result ) {
		result->SetLabelType( labelType );
		result->SetBatchSize( batchSize );
		result->SetMaxBatchCount( maxBatchCount );
		result->SetProblem( problem );
	} );
}

CLayerWrapper<CPixelToImageLayer> PixelToImage( int imageHeight, int imageWidth )
{
	return CLayerWrapper<CPixelToImageLayer>( "PixelToImage", [=]( CPixelToImageLayer* result ) {
		result->SetImageHeight( imageHeight );
		result->SetImageWidth( imageWidth );
	} );
}

CLayerWrapper<CImageToPixelLayer> ImageToPixel()
{
	return CLayerWrapper<CImageToPixelLayer>( "ImageToPixel" );
}

CLayerWrapper<CImageResizeLayer> ImageResize( int deltaLeft, int deltaRight, int deltaTop,
	int deltaBottom, float defaultValue )
{
	return CLayerWrapper<CImageResizeLayer>( "ImageResize", [=]( CImageResizeLayer* result ) {
		result->SetDelta( CImageResizeLayer::IS_Left, deltaLeft );
		result->SetDelta( CImageResizeLayer::IS_Right, deltaRight );
		result->SetDelta( CImageResizeLayer::IS_Bottom, deltaBottom );
		result->SetDelta( CImageResizeLayer::IS_Top, deltaTop );
		result->SetDefalutValue( defaultValue );
	} );
}

CLayerWrapper<CPrecisionRecallLayer> PrecisionRecall()
{
	return CLayerWrapper<CPrecisionRecallLayer>( "PrecisionRecall" );
}

CLayerWrapper<CReorgLayer> Reorg()
{
	return CLayerWrapper<CReorgLayer>( "Reorg" );
}

CLayerWrapper<CRepeatSequenceLayer> RepeatSequence( int repeatCount )
{
	return CLayerWrapper<CRepeatSequenceLayer>( "RepeatSequence", [=]( CRepeatSequenceLayer* result ) {
		result->SetRepeatCount( repeatCount );
	} );
}

CLayerWrapper<CSequenceSumLayer> SequenceSum()
{
	return CLayerWrapper<CSequenceSumLayer>( "SequenceSum" );
}

//////////////////////////////////////////////////////////////////////////////////////////
} // namespace SimpleAPI 
} // namespace NeoML
