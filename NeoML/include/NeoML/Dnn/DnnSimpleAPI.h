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

#pragma once

#include <NeoML/Dnn/DnnLambdaHolder.h>
#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Dnn.h>
#include <NeoML/Dnn/Layers/DropoutLayer.h>
#include <NeoML/Dnn/Layers/ActivationLayers.h>
#include <NeoML/Dnn/Layers/GELULayer.h>
#include <NeoML/Dnn/Layers/FullyConnectedLayer.h>
#include <NeoML/Dnn/Layers/LossLayer.h>
#include <NeoML/Dnn/Layers/SourceLayer.h>
#include <NeoML/Dnn/Layers/SinkLayer.h>
#include <NeoML/Dnn/Layers/PositionalEmbeddingLayer.h>
#include <NeoML/Dnn/Layers/SoftmaxLayer.h>
#include <NeoML/Dnn/Layers/AddToObjectLayer.h>
#include <NeoML/Dnn/Layers/TransformLayer.h>
#include <NeoML/Dnn/Layers/TransposeLayer.h>
#include <NeoML/Dnn/Layers/ConcatLayer.h>
#include <NeoML/Dnn/Layers/BatchNormalizationLayer.h>
#include <NeoML/Dnn/Layers/ObjectNormalizationLayer.h>
#include <NeoML/Dnn/Layers/MultiheadAttentionLayer.h>
#include <NeoML/Dnn/Layers/MatrixMultiplicationLayer.h>
#include <NeoML/Dnn/Layers/EltwiseLayer.h>
#include <NeoML/Dnn/Layers/GlobalMaxPoolingLayer.h>
#include <NeoML/Dnn/Layers/GlobalMeanPoolingLayer.h>
#include <NeoML/Dnn/Layers/LstmLayer.h>
#include <NeoML/Dnn/Layers/GruLayer.h>
#include <NeoML/Dnn/Layers/SubSequenceLayer.h>
#include <NeoML/Dnn/Layers/ArgmaxLayer.h>
#include <NeoML/Dnn/Layers/MultichannelLookupLayer.h>
#include <NeoML/Dnn/Layers/ConvLayer.h>
#include <NeoML/Dnn/Layers/MaxOverTimePoolingLayer.h>
#include <NeoML/Dnn/Layers/3dConvLayer.h>
#include <NeoML/Dnn/Layers/3dPoolingLayer.h>
#include <NeoML/Dnn/Layers/TransposedConvLayer.h>
#include <NeoML/Dnn/Layers/AccumulativeLookupLayer.h>
#include <NeoML/Dnn/Layers/AccuracyLayer.h>
#include <NeoML/Dnn/Layers/AttentionLayer.h>
#include <NeoML/Dnn/Layers/BinaryFocalLossLayer.h>
#include <NeoML/Dnn/Layers/CenterLossLayer.h>
#include <NeoML/Dnn/Layers/ChannelwiseConvLayer.h>
#include <NeoML/Dnn/Layers/CrfLayer.h>
#include <NeoML/Dnn/Layers/CtcLayer.h>
#include <NeoML/Dnn/Layers/DotProductLayer.h>
#include <NeoML/Dnn/Layers/EnumBinarizationLayer.h>
#include <NeoML/Dnn/Layers/FocalLossLayer.h>
#include <NeoML/Dnn/Layers/FullyConnectedSourceLayer.h>
#include <NeoML/Dnn/Layers/ImageAndPixelConversionLayer.h>
#include <NeoML/Dnn/Layers/ImageResizeLayer.h>
#include <NeoML/Dnn/Layers/MaxOverTimePoolingLayer.h>
#include <NeoML/Dnn/Layers/MultiHingeLossLayer.h>
#include <NeoML/Dnn/Layers/PrecisionRecallLayer.h>
#include <NeoML/Dnn/Layers/QualityControlLayer.h>
#include <NeoML/Dnn/Layers/ReorgLayer.h>
#include <NeoML/Dnn/Layers/RepeatSequenceLayer.h>
#include <NeoML/Dnn/Layers/SequenceSumLayer.h>
#include <NeoML/Dnn/Layers/SequenceSumLayer.h>
#include <NeoML/Dnn/Layers/SplitLayer.h>
#include <NeoML/Dnn/Layers/TimeConvLayer.h>
#include <NeoML/Dnn/Layers/TransposedConvLayer.h>
#include <NeoML/Dnn/Layers/Upsampling2DLayer.h>

namespace NeoML {
namespace SimpleAPI {

// Layer output.
class NEOML_API CLayerOutput {
public:
	// Layer
	CBaseLayer* Layer;
	// Output number
	int OutputNumber;
	
	// Default value for optional inputs.
	CLayerOutput() : Layer( 0 ), OutputNumber( -1 ) {}
	CLayerOutput( const CLayerOutput& other ) :
		Layer( other.Layer ), OutputNumber( other.OutputNumber ) {}
	CLayerOutput( CBaseLayer* layer, int outputNumber ) :
		Layer( layer ),
		OutputNumber( outputNumber )
	{
		NeoAssert( Layer != 0 );
		NeoAssert( OutputNumber >= 0 );
	}

	// Converting constructor
	CLayerOutput( CBaseLayer* layer ) :
		Layer( layer ), OutputNumber( 0 ) {}

	// Is this layer optional, i.e. created by CLayerOutout() default constructor.
	bool IsOptional() const { return Layer == 0 && OutputNumber == -1; }
	// Check if the layer output is valid.
	bool IsValid() const { return Layer != 0 && OutputNumber >= 0 && Layer->GetDnn() != 0; }
};

//////////////////////////////////////////////////////////////////////////////////////////

// Wrapper for the layer. Store layer type, init function and initialization params.
template<typename T>
class CLayerWrapper {
public:
	CLayerWrapper( const char* prefix, CLambda<void( T* )> lambda );
	explicit CLayerWrapper( const char* prefix );
	CLayerWrapper( const CLayerWrapper<T>& other ) :
		prefix( other.prefix ), initFunc( other.initFunc ) {}

	// Connect inputs to the layer and change layer name.
	T* operator()( const char* name, const CLayerOutput& layer1,
		const CLayerOutput& layer2 = CLayerOutput(),
		const CLayerOutput& layer3 = CLayerOutput(),
		const CLayerOutput& layer4 = CLayerOutput(),
		const CLayerOutput& layer5 = CLayerOutput(),
		const CLayerOutput& layer6 = CLayerOutput() );

	// Connect inputs to the layer.
	T* operator()( const CLayerOutput& layer1,
		const CLayerOutput& layer2 = CLayerOutput(),
		const CLayerOutput& layer3 = CLayerOutput(),
		const CLayerOutput& layer4 = CLayerOutput(),
		const CLayerOutput& layer5 = CLayerOutput(),
		const CLayerOutput& layer6 = CLayerOutput() );

private:
	// Prefix for create layer name.
	const char* prefix;
	// Init function for new layer.
	CLambda<void( T* )> initFunc;
	// New layer.
	CPtr<T> layer;

	CString findFreeLayerName( const CDnn& first, const char* prefix ) const;
};

template<typename T>
CLayerWrapper<T>::CLayerWrapper( const char* _prefix, CLambda<void( T* )> _initFunc ) :
	prefix( _prefix ),
	initFunc( _initFunc )
{
}

template<typename T>
CLayerWrapper<T>::CLayerWrapper( const char* _prefix ) :
	prefix( _prefix )
{
	NeoAssert( prefix != 0 );
}

template<typename T>
T* CLayerWrapper<T>::operator()( const char* name,
	const CLayerOutput& layer1, const CLayerOutput& layer2,
	const CLayerOutput& layer3, const CLayerOutput& layer4,
	const CLayerOutput& layer5, const CLayerOutput& layer6 )
{
	NeoAssert( !layer1.IsOptional() );
	NeoAssert( layer1.IsValid() );
	NeoAssert( name != 0 );

	if( layer == 0 ) {
		CDnn* network = layer1.Layer->GetDnn();
		layer = new T( network->GetMathEngine() );
		if( !initFunc.IsEmpty() ) {
			initFunc( layer );
		}
		layer->SetName( name );
		network->AddLayer( *layer );
	}

	CArray<CLayerOutput> inputLayers;
	inputLayers.Add( layer1 );
	inputLayers.Add( layer2 );
	inputLayers.Add( layer3 );
	inputLayers.Add( layer4 );
	inputLayers.Add( layer5 );
	inputLayers.Add( layer6 );

	const int startIndex = layer->GetInputCount();
	for( int i = 0; i < inputLayers.Size(); i++ ) {
		const CLayerOutput& inputLayer = inputLayers[i];
		if( inputLayer.IsOptional() ) {
			break;
		}
		NeoAssert( inputLayer.IsValid() );
		layer->Connect( startIndex + i, *inputLayer.Layer, inputLayer.OutputNumber );
	}

	return layer;
}

template<typename T>
T* CLayerWrapper<T>::operator()(
	const CLayerOutput& layer1, const CLayerOutput& layer2,
	const CLayerOutput& layer3, const CLayerOutput& layer4,
	const CLayerOutput& layer5, const CLayerOutput& layer6 )
{
	NeoAssert( layer1.IsValid() );
	CDnn* network = layer1.Layer->GetDnn();
	const CString name = findFreeLayerName( *network, prefix );
	return operator()( name, layer1, layer2, layer3, layer4, layer5, layer6 );
}

template<typename T>
CString CLayerWrapper<T>::findFreeLayerName(
	const CDnn& network, const char* prefix ) const
{
	const CString prefixStr( prefix );

	int index = 0;
	while( true ) {
		const CString newName = prefixStr + "_" + Str( index++ );
		if( !network.HasLayer( newName ) ) {
			return newName;
		}
	}
	NeoAssert( false );
	// make compiler happy
	return CString();
}

//////////////////////////////////////////////////////////////////////////////////////////

/*
Simple API for create neural networks.

Example 0:

	CDnn net{ rnd, engine };

	auto x = Source( net, "Input0" );
	auto labels = Source( net, "Labels" );

	x = FullyConnected( 100 )( x );
	x = Relu() ( x );
	x = FullyConnected( 200 )( x );
	x = Gelu()( x );
	x = Dropout( 0.5f )( x )
	x = FullyConnected( 1 )( x );
	BinaryCrossEntropy()( x, labels );

Example 2:

	CDnn net{ rnd, engine };

	auto x = Source( net, "Input0" );
	auto y = Source( net, "Input1" );
	auto labels = Source( net, "Labels" );
	auto weights = Source( net, "Weightss" );

	auto fc = FullyConnected( 100 );

	x = fc( x ); // 1.
	y = fc( y ); // 2. Share weights with 1.
	x = Concat() ( x, y );
	x = Relu() ( x );
	x = fc( x );
	x = Gelu()( x );
	x = Dropout( 0.5f )( x )
	x = FullyConnected( 1 )( x );
	BinaryCrossEntropy( 2.0f )( x, labels, weights );

*/


//////////////////////////////////////////////////////////////////////////////////////////

// Create CSourceLayer with name
NEOML_API CBaseLayer* Source( CDnn& network, const char* name );
// Create CSinkLayer with name
NEOML_API CSinkLayer* Sink( const CLayerOutput& layer, const char* name );

NEOML_API CLayerWrapper<CDropoutLayer> Dropout( float dropoutRate,
	bool isSpatial = false, bool isBatchwise = false );

// Activation functions
NEOML_API CLayerWrapper<CReLULayer> Relu();
NEOML_API CLayerWrapper<CGELULayer> Gelu();
NEOML_API CLayerWrapper<CLinearLayer> Linear( float multiplier, float freeTerm );
NEOML_API CLayerWrapper<CELULayer> Elu( float alpha = 0.01f );
NEOML_API CLayerWrapper<CLeakyReLULayer> LeakyRelu( float alpha = 0.01f );
NEOML_API CLayerWrapper<CAbsLayer> Abs();
NEOML_API CLayerWrapper<CSigmoidLayer> Sigmoid();
NEOML_API CLayerWrapper<CTanhLayer> Tanh();
NEOML_API CLayerWrapper<CHardTanhLayer> HardTanh();
NEOML_API CLayerWrapper<CHardSigmoidLayer> HardSigmoid( float slope, float bias );
NEOML_API CLayerWrapper<CPowerLayer> Power( float exponent );
NEOML_API CLayerWrapper<CHSwishLayer> HSwish();

NEOML_API CLayerWrapper<CFullyConnectedLayer> FullyConnected(
	int numberOfElements, bool zeroFreeTerm = false );

NEOML_API CLayerWrapper<CBinaryCrossEntropyLossLayer> BinaryCrossEntropyLoss(
	float positiveWeight = 1.0f, float lossWeight = 1.0f );
NEOML_API CLayerWrapper<CCrossEntropyLossLayer> CrossEntropyLoss(
	bool isSoftmaxApplied = true, float lossWeight = 1.0f );
NEOML_API CLayerWrapper<CEuclideanLossLayer> EuclideanLoss( float lossWeight = 1.0f );
NEOML_API CLayerWrapper<CHingeLossLayer> HingeLoss( float lossWeight = 1.0f );
NEOML_API CLayerWrapper<CSquaredHingeLossLayer> SquaredHingeLoss(
	float lossWeight = 1.0f );
NEOML_API CLayerWrapper<CBinaryFocalLossLayer> BinaryFocalLoss( 
	float focalForce, float lossWeight = 1.0f );
NEOML_API CLayerWrapper<CCenterLossLayer> CenterLoss( int numberOfClasses,
	float classCentersConvergenceRate, float lossWeight = 1.0f );
NEOML_API CLayerWrapper<CMultiHingeLossLayer> MultiHingeLoss( float lossWeight = 1.0f );
NEOML_API CLayerWrapper<CMultiSquaredHingeLossLayer> MultiSquaredHingeLoss(
	float lossWeight = 1.0f );

NEOML_API CLayerWrapper<CPositionalEmbeddingLayer> PositionalEmbedding(
	CPositionalEmbeddingLayer::TPositionalEmbeddingType type );

NEOML_API CLayerWrapper<CSoftmaxLayer> Softmax(
	CSoftmaxLayer::TNormalizationArea normalizationArea );
NEOML_API CLayerWrapper<CAddToObjectLayer> AddToObject();

// Special values for parameters in Transform
enum TTransformRule {
	TR_Remainder = -1, // Shape inference, only one layer should have this value.
	TR_Same = -2 // Rest dimension the same.
};

NEOML_API CLayerWrapper<CTransformLayer> Transform( int batchLength,
	int batchWidth, int listSize, int height, int width, int depth, int channel );

NEOML_API CLayerWrapper<CTransposeLayer> Transpose( TBlobDim d1, TBlobDim d2 );

NEOML_API CLayerWrapper<CConcatBatchWidthLayer> ConcatBatchWidth();
NEOML_API CLayerWrapper<CConcatHeightLayer> ConcatHeight();
NEOML_API CLayerWrapper<CConcatWidthLayer> ConcatWidth();
NEOML_API CLayerWrapper<CConcatDepthLayer> ConcatDepth();
NEOML_API CLayerWrapper<CConcatChannelsLayer> ConcatChannels();

NEOML_API CLayerWrapper<CBatchNormalizationLayer> BatchNormalization(
	bool isChannelBased, bool isZeroFreeTerm = false, float slowConvergenceRate = 1.0f );

NEOML_API CLayerWrapper<CObjectNormalizationLayer> ObjectNormalization(
	float epsilon = 1e-5f );

NEOML_API CLayerWrapper<CMultiheadAttentionLayer> MultiheadAttention(
	int headCount, int hiddenSize, int outputSize, float dropoutRate );

NEOML_API CLayerWrapper<CMatrixMultiplicationLayer> MatrixMultiplication();

NEOML_API CLayerWrapper<CEltwiseSumLayer> Sum();
NEOML_API CLayerWrapper<CEltwiseMulLayer> Mul();
NEOML_API CLayerWrapper<CEltwiseNegMulLayer> NegMul();
NEOML_API CLayerWrapper<CEltwiseMaxLayer> Max();

NEOML_API CLayerWrapper<CGlobalMaxPoolingLayer> GlobalMaxPooling( int maxCount );
NEOML_API CLayerWrapper<CGlobalMeanPoolingLayer> GlobalMeanPooling();

NEOML_API CLayerWrapper<CLstmLayer> Lstm(
	int hiddenSize, float dropoutRate, bool isInCompatibilityMode = false );

NEOML_API CLayerWrapper<CGruLayer> Gru( int hiddenSize );

NEOML_API CLayerWrapper<CSubSequenceLayer> SubSequence(
	int startPos, int length );
// CSubSequenceLayer with SetReverse()
NEOML_API CLayerWrapper<CSubSequenceLayer> ReverseSubSequence();

NEOML_API CLayerWrapper<CArgmaxLayer> Argmax( TBlobDim dim );

// Convolution parameterss along one of the axes.
struct NEOML_API CConvAxisParams {
	// FilterSize
	int Size;
	// Padding
	int Padding;
	// Stride
	int Stride;
	// Dilation
	int Dilation;

	CConvAxisParams() : Size( 1 ), Padding( 0 ), Stride( 1 ), Dilation( 1 ) {}
	explicit CConvAxisParams( int size, int padding = 0, int stride = 1, int dilation = 1 ) :
		Size( size ),
		Padding( padding ),
		Stride( stride ),
		Dilation( dilation )
	{
	}
};

NEOML_API CLayerWrapper<CConvLayer> Conv( int filterCount,
	const CConvAxisParams& heightParams, const CConvAxisParams& widthParams,
	bool isZeroFreeTerm = false );

// Convolution along width dimension, i.e:
//	filter height = 1
//	stride height = 1
//	dilation height = 1
//	padding height = 0
inline CLayerWrapper<CConvLayer> ConvByWidth( int filterCount,
	int size, int padding = 0, int stride = 1, int dilation = 1,
	bool isZeroFreeTerm = false )
{
	return Conv( filterCount, CConvAxisParams(),
		CConvAxisParams( size, padding, stride, dilation ), isZeroFreeTerm );
}

// Convolution along height dimension, i.e:
//	filter width = 1
//	stride width = 1
//	dilation width = 1
//	padding width = 0
inline CLayerWrapper<CConvLayer> ConvByHeight( int filterCount,
	int size, int padding = 0, int stride = 1, int dilation = 1,
	bool isZeroFreeTerm = false )
{
	return Conv( filterCount, CConvAxisParams( size, padding, stride, dilation ),
		CConvAxisParams(), isZeroFreeTerm );
}

NEOML_API CLayerWrapper<CChannelwiseConvLayer> ChannelwiseConv( int filterCount,
	const CConvAxisParams& heightParams, const CConvAxisParams& widthParams,
	bool isZeroFreeTerm = false );

inline CLayerWrapper<CTimeConvLayer> TimeConv( int filterCount,
	int size, int padding = 0, int stride = 1, int dilation = 1 );

// N.B. Layer does not support dilation! 
// The Dilation parameter in the CConvAxisParams will be ignored.
NEOML_API CLayerWrapper<C3dConvLayer> Conv3d( int filterCount,
	const CConvAxisParams& heightParams,
	const CConvAxisParams& widthParams,
	const CConvAxisParams& depthParams,
	bool isZeroFreeTerm = false );

NEOML_API CLayerWrapper<CTransposedConvLayer> TransposedConv( int filterCount,
	const CConvAxisParams& heightParams, const CConvAxisParams& widthParams,
	bool isZeroFreeTerm = false );

NEOML_API CLayerWrapper<CMultichannelLookupLayer> MultichannelLookup(
	const CArray<CLookupDimension>& lookupDimensions, bool useFrameworkLearning );

// Standard embeddings.
NEOML_API CLayerWrapper<CMultichannelLookupLayer> Embeddings( int count, int size );

NEOML_API CLayerWrapper<CUpsampling2DLayer> Upsampling2d( int heightCopyCount,
	int widthCopyCount );

NEOML_API CLayerWrapper<CSplitBatchWidthLayer> SplitBatchWidth(
	const CArray<int>& outputCounts );
NEOML_API CLayerWrapper<CSplitBatchWidthLayer> SplitBatchWidth( int output0,
	int output1 = 0, int output2 = 0 );

NEOML_API CLayerWrapper<CSplitChannelsLayer> SplitChannels(
	const CArray<int>& outputCounts );
NEOML_API CLayerWrapper<CSplitChannelsLayer> SplitChannels( int output0,
	int output1 = 0, int output2 = 0 );

NEOML_API CLayerWrapper<CSplitDepthLayer> SplitDepth(
	const CArray<int>& outputCounts );
NEOML_API CLayerWrapper<CSplitDepthLayer> SplitDepth( int output0,
	int output1 = 0, int output2 = 0 );

NEOML_API CLayerWrapper<CSplitHeightLayer> SplitHeight(
	const CArray<int>& outputCounts );
NEOML_API CLayerWrapper<CSplitHeightLayer> SplitHeight( int output0,
	int output1 = 0, int output2 = 0 );

NEOML_API CLayerWrapper<CSplitWidthLayer> SplitWidth(
	const CArray<int>& outputCounts );
NEOML_API CLayerWrapper<CSplitWidthLayer> SplitWidth( int output0,
	int output1 = 0, int output2 = 0 );

NEOML_API CLayerWrapper<CMaxOverTimePoolingLayer> MaxOverTimePooling(
	int filterLength, int strideLength );

NEOML_API CLayerWrapper<C3dMaxPoolingLayer> Pooling3dMax(
	int filterHeight, int filterWidth, int filterDepth,
	int strideHeight = 1, int strideWidth = 1, int strideDepth = 1 );

NEOML_API CLayerWrapper<C3dMeanPoolingLayer> Pooling3dMean(
	int filterHeight, int filterWidth, int filterDepth,
	int strideHeight = 1, int strideWidth = 1, int strideDepth = 1 );

NEOML_API CLayerWrapper<CAccumulativeLookupLayer> AccumulativeLookup(
	int count, int size );

NEOML_API CLayerWrapper<CAccuracyLayer> Accuracy();
NEOML_API CLayerWrapper<CConfusionMatrixLayer> ConfusionMatrix();

NEOML_API CLayerWrapper<CAttentionDecoderLayer> AttentionDecoder(
	TAttentionScore score, int outObjectSize, int outSeqLen, int hiddenSize );

NEOML_API CLayerWrapper<CCtcLossLayer> CtcLoss( int blankLabel, bool allowBlankLabelSkip,
	float lossWeight = 1.0f );

NEOML_API CLayerWrapper<CDotProductLayer> DotProduct();

NEOML_API CLayerWrapper<CEnumBinarizationLayer> EnumBinarization( int enumSize );

NEOML_API CLayerWrapper<CFullyConnectedSourceLayer> FullyConnectedSource(
	TBlobType labelType, int batchSize, int maxBatchCount, IProblem* problem );

NEOML_API CLayerWrapper<CPixelToImageLayer> PixelToImage( int imageHeight, int imageWidth );
NEOML_API CLayerWrapper<CImageToPixelLayer> ImageToPixel();
NEOML_API CLayerWrapper<CImageResizeLayer> ImageResize( int deltaLeft, int deltaRight,
	int deltaTop, int deltaBottom, float defaultValue );

NEOML_API CLayerWrapper<CPrecisionRecallLayer> PrecisionRecall();
NEOML_API CLayerWrapper<CReorgLayer> Reorg();
NEOML_API CLayerWrapper<CRepeatSequenceLayer> RepeatSequence( int repeatCount );
NEOML_API CLayerWrapper<CSequenceSumLayer> SequenceSum();

//////////////////////////////////////////////////////////////////////////////////////////
} // namespace SimpleAPI 
} // namespace NeoML
