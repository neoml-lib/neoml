/* Copyright © 2017-2024 ABBYY

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

#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoML/Dnn/Dnn.h>
#include <NeoML/Dnn/Layers/3dConvLayer.h>
#include <NeoML/Dnn/Layers/ActivationLayers.h>
#include <NeoML/Dnn/Layers/AddToObjectLayer.h>
#include <NeoML/Dnn/Layers/BackLinkLayer.h>
#include <NeoML/Dnn/Layers/BaseInPlaceLayer.h>
#include <NeoML/Dnn/Layers/BatchNormalizationLayer.h>
#include <NeoML/Dnn/Layers/BroadcastLayer.h>
#include <NeoML/Dnn/Layers/CastLayer.h>
#include <NeoML/Dnn/Layers/ChannelwiseConvLayer.h>
#include <NeoML/Dnn/Layers/CompositeLayer.h>
#include <NeoML/Dnn/Layers/ConcatLayer.h>
#include <NeoML/Dnn/Layers/ConvLayer.h>
#include <NeoML/Dnn/Layers/DataLayer.h>
#include <NeoML/Dnn/Layers/DropoutLayer.h>
#include <NeoML/Dnn/Layers/EltwiseLayer.h>
#include <NeoML/Dnn/Layers/FullyConnectedLayer.h>
#include <NeoML/Dnn/Layers/GlobalMeanPoolingLayer.h>
#include <NeoML/Dnn/Layers/IndRnnLayer.h>
#include <NeoML/Dnn/Layers/LossLayer.h>
#include <NeoML/Dnn/Layers/LstmLayer.h>
#include <NeoML/Dnn/Layers/MatrixMultiplicationLayer.h>
#include <NeoML/Dnn/Layers/MobileNetV2BlockLayer.h>
#include <NeoML/Dnn/Layers/MultichannelLookupLayer.h>
#include <NeoML/Dnn/Layers/MultiheadAttentionLayer.h>
#include <NeoML/Dnn/Layers/ObjectNormalizationLayer.h>
#include <NeoML/Dnn/Layers/ParameterLayer.h>
#include <NeoML/Dnn/Layers/PoolingLayer.h>
#include <NeoML/Dnn/Layers/RecurrentLayer.h>
#include <NeoML/Dnn/Layers/QrnnLayer.h>
#include <NeoML/Dnn/Layers/QualityControlLayer.h>
#include <NeoML/Dnn/Layers/SinkLayer.h>
#include <NeoML/Dnn/Layers/SoftmaxLayer.h>
#include <NeoML/Dnn/Layers/SourceLayer.h>
#include <NeoML/Dnn/Layers/SplitLayer.h>
#include <NeoML/Dnn/Layers/TimeConvLayer.h>
#include <NeoML/Dnn/Layers/TransformLayer.h>
#include <NeoML/Dnn/Layers/TransposedConvLayer.h>
#include <NeoML/Dnn/Layers/TransposeLayer.h>
#ifndef NEOML_COMPACT
#include <NeoML/Dnn/Layers/3dPoolingLayer.h>
#include <NeoML/Dnn/Layers/3dTransposedConvLayer.h>
#include <NeoML/Dnn/Layers/AccumulativeLookupLayer.h>
#include <NeoML/Dnn/Layers/AccuracyLayer.h>
#include <NeoML/Dnn/Layers/ArgmaxLayer.h>
#include <NeoML/Dnn/Layers/AttentionLayer.h>
#include <NeoML/Dnn/Layers/BertConvLayer.h>
#include <NeoML/Dnn/Layers/BinaryFocalLossLayer.h>
#include <NeoML/Dnn/Layers/CenterLossLayer.h>
#include <NeoML/Dnn/Layers/ChannelwiseWith1x1Layer.h>
#include <NeoML/Dnn/Layers/CrfLayer.h>
#include <NeoML/Dnn/Layers/CtcLayer.h>
#include <NeoML/Dnn/Layers/CumSumLayer.h>
#include <NeoML/Dnn/Layers/DepthToSpaceLayer.h>
#include <NeoML/Dnn/Layers/DotProductLayer.h>
#include <NeoML/Dnn/Layers/EnumBinarizationLayer.h>
#include <NeoML/Dnn/Layers/FocalLossLayer.h>
#include <NeoML/Dnn/Layers/FullyConnectedSourceLayer.h>
#include <NeoML/Dnn/Layers/GlobalMaxPoolingLayer.h>
#include <NeoML/Dnn/Layers/GlobalSumPoolingLayer.h>
#include <NeoML/Dnn/Layers/GrnLayer.h>
#include <NeoML/Dnn/Layers/GruLayer.h>
#include <NeoML/Dnn/Layers/ImageAndPixelConversionLayer.h>
#include <NeoML/Dnn/Layers/ImageResizeLayer.h>
#include <NeoML/Dnn/Layers/InterpolationLayer.h>
#include <NeoML/Dnn/Layers/IrnnLayer.h>
#include <NeoML/Dnn/Layers/LogicalLayers.h>
#include <NeoML/Dnn/Layers/LoraFullyConnectedLayer.h>
#include <NeoML/Dnn/Layers/LrnLayer.h>
#include <NeoML/Dnn/Layers/MaxOverTimePoolingLayer.h>
#include <NeoML/Dnn/Layers/MobileNetV3BlockLayer.h>
#include <NeoML/Dnn/Layers/ModelWrapperLayer.h>
#include <NeoML/Dnn/Layers/MultiHingeLossLayer.h>
#include <NeoML/Dnn/Layers/PositionalEmbeddingLayer.h>
#include <NeoML/Dnn/Layers/PrecisionRecallLayer.h>
#include <NeoML/Dnn/Layers/ProjectionPoolingLayer.h>
#include <NeoML/Dnn/Layers/ReorgLayer.h>
#include <NeoML/Dnn/Layers/RepeatSequenceLayer.h>
#include <NeoML/Dnn/Layers/RowwiseOperationChainLayer.h>
#include <NeoML/Dnn/Layers/ScatterGatherLayers.h>
#include <NeoML/Dnn/Layers/SequenceSumLayer.h>
#include <NeoML/Dnn/Layers/SpaceToDepthLayer.h>
#include <NeoML/Dnn/Layers/SubSequenceLayer.h>
#include <NeoML/Dnn/Layers/TiedEmbeddingsLayer.h>
#include <NeoML/Dnn/Layers/TransformerLayer.h>
#include <NeoML/Dnn/Layers/TransformerSourceMaskLayer.h>
#include <NeoML/Dnn/Layers/Upsampling2DLayer.h>
#endif //!NEOML_COMPACT

namespace NeoML {

// The minimum size of temporary data blobs to start reusing memory
static const size_t MinReuseMemoryModeNetSize = 4 * 1024 * 1024;

static CMap<CString, TCreateLayerFunction, CDefaultHash<CString>, RuntimeHeap>& getRegisteredLayers()
{
	static CMap<CString, TCreateLayerFunction, CDefaultHash<CString>, RuntimeHeap> registeredLayers;
	return registeredLayers;
}


// Class name hash to compare type_info
struct CTypeInfoNameHash final {
	static int HashKey( const std::type_info* key )
	{
		return GetMBCStringHash( key->name() );
	}

	static bool IsEqual( const std::type_info* first, const std::type_info* second )
	{
		return ( ::strcmp( first->name(), second->name() ) == 0 );
	}
};

static CMap<const std::type_info*, CString, CTypeInfoNameHash, RuntimeHeap>& getLayerClasses()
{
	static CMap<const std::type_info*, CString, CTypeInfoNameHash, RuntimeHeap> layerClasses;
	return layerClasses;
}

void RegisterLayerClass( const char* className, const char* additionalName, const std::type_info& typeInfo, TCreateLayerFunction function )
{
	NeoAssert( !getRegisteredLayers().Has( className ) );
	getRegisteredLayers().Add( className, function );
	if( additionalName != 0 ) {
		NeoAssert( !getRegisteredLayers().Has( additionalName ) );
		getRegisteredLayers().Add( additionalName, function );
	}
	getLayerClasses().Add( &typeInfo, className );
}

void UnregisterLayerClass( const std::type_info& typeInfo )
{
	getRegisteredLayers().Delete( getLayerClasses().Get( &typeInfo ) );
	getLayerClasses().Delete( &typeInfo );
}

bool IsRegisteredLayerClass( const char* className )
{
	return getRegisteredLayers().Has( className );
}

void GetRegisteredLayerClasses( CArray<const char*>& layerNames )
{
	const CMap<CString, TCreateLayerFunction, CDefaultHash<CString>, RuntimeHeap>& registeredLayers = getRegisteredLayers();
	layerNames.DeleteAll();
	layerNames.SetBufferSize( registeredLayers.Size() );
	for( int pos = registeredLayers.GetFirstPosition(); pos != NotFound; pos = registeredLayers.GetNextPosition( pos ) ) {
		layerNames.Add( registeredLayers.GetKey( pos ) );
	}
}

CPtr<CBaseLayer> CreateLayer( const char* className, IMathEngine& mathEngine )
{
	NeoAssert( getRegisteredLayers().Has( className ) );
	return getRegisteredLayers()[className]( mathEngine );
}

static CPtr<CBaseLayer> createLayer( IMathEngine& mathEngine, const CString& className )
{
	TMapPosition pos = getRegisteredLayers().GetFirstPosition( className );
	if( pos == NotFound ) {
		return 0;
	}
	return getRegisteredLayers().GetValue( pos )( mathEngine );
}

static CString getLayerClass( const CBaseLayer* layer )
{
	if( layer == 0 ) {
		return CString();
	}
	const std::type_info& layerType = typeid( *layer );
	TMapPosition pos = getLayerClasses().GetFirstPosition( &layerType );
	if( pos == NotFound ) {
		return CString();
	}
	return getLayerClasses().GetValue( pos );
}

CString GetLayerClass( const CBaseLayer& layer )
{
	return getLayerClass( &layer );
}

void SerializeLayer( CArchive& archive, IMathEngine& mathEngine, CPtr<CBaseLayer>& layer )
{
	if( archive.IsStoring() ) {
		CString name = getLayerClass( layer );
		NeoAssertMsg( layer == nullptr || name != "", "Try to store non-registered layer" );
		archive << name;
		if( layer != nullptr ) {
			layer->Serialize( archive );
		}
	} else if( archive.IsLoading() ) {
		CString name;
		archive >> name;
		layer = createLayer( mathEngine, name );
		CheckArchitecture( name == "" || layer != nullptr, name, "Try to restore unknown layer from archive" );
		if( layer != nullptr ) {
			layer->Serialize( archive );
		}
	} else {
		NeoAssert( false );
	}
}

//---------------------------------------------------------------------------------------------------------

// Register all layer types
namespace {

// {{ CNN
REGISTER_NEOML_LAYER( C3dConvLayer, "FmlCnn3dConvLayer" )
REGISTER_NEOML_LAYER( CBackLinkLayer, "FmlCnnBackLink" )
REGISTER_NEOML_LAYER( CBatchNormalizationLayer, "FmlCnnBatchNormalizationLayer" )
REGISTER_NEOML_LAYER( CChannelwiseConvLayer, "FmlCnnChannelwiseConvLayer" )
REGISTER_NEOML_LAYER( CConcatBatchLengthLayer, "FmlCnnConcatBatchLengthLayer" )
REGISTER_NEOML_LAYER( CConcatBatchWidthLayer, "FmlCnnConcatBatchWidthLayer" )
REGISTER_NEOML_LAYER( CConcatChannelsLayer, "FmlCnnConcatChannelsLayer" )
REGISTER_NEOML_LAYER( CConcatDepthLayer, "FmlCnnConcatDepthLayer" )
REGISTER_NEOML_LAYER( CConcatHeightLayer, "FmlCnnConcatHeightLayer" )
REGISTER_NEOML_LAYER( CConcatListSizeLayer, "FmlCnnConcatListSizeLayer" )
REGISTER_NEOML_LAYER( CConcatObjectLayer, "FmlCnnConcatObjectLayer" )
REGISTER_NEOML_LAYER( CConcatWidthLayer, "FmlCnnConcatWidthLayer" )
REGISTER_NEOML_LAYER( CCompositeLayer, "FmlCnnCompositeLayer" )
REGISTER_NEOML_LAYER( CConvLayer, "FmlCnnConvLayer" )
REGISTER_NEOML_LAYER( CDropoutLayer, "FmlCnnDropoutLayer" )
REGISTER_NEOML_LAYER( CEltwiseMaxLayer, "FmlCnnEltwiseMaxLayer" )
REGISTER_NEOML_LAYER( CEltwiseMulLayer, "FmlCnnEltwiseMulLayer" )
REGISTER_NEOML_LAYER( CEltwiseNegMulLayer, "FmlCnnEltwiseNegMulLayer" )
REGISTER_NEOML_LAYER( CEltwiseSumLayer, "FmlCnnEltwiseSumLayer" )
REGISTER_NEOML_LAYER( CFullyConnectedLayer, "FmlCnnFullyConnectedLayer" )
REGISTER_NEOML_LAYER_EX( CGlobalMeanPoolingLayer, "FmlCnnGlobalMainPoolingLayer", "FmlCnnGlobalAveragePoolingLayer" )
REGISTER_NEOML_LAYER( CLinearLayer, "FmlCnnLinearLayer" )
REGISTER_NEOML_LAYER( CLstmLayer, "FmlCnnLstmLayer" )
REGISTER_NEOML_LAYER( CMaxPoolingLayer, "FmlCnnMaxPoolingLayer" )
REGISTER_NEOML_LAYER( CMeanPoolingLayer, "FmlCnnMeanPoolingLayer" )
REGISTER_NEOML_LAYER( CMultichannelLookupLayer, "FmlCnnMultychannelLookupLayer" )
REGISTER_NEOML_LAYER( CRecurrentLayer, "FmlCnnRecurrentLayer" )
REGISTER_NEOML_LAYER( CRleConvLayer, "FmlCnnRleConvLayer" )
REGISTER_NEOML_LAYER( CSinkLayer, "FmlCnnSinkLayer" )
REGISTER_NEOML_LAYER_EX( CSoftmaxLayer, "FmlCnnSoftmaxLayer", "FmlCCnnChannelwiseSoftmaxLayer" )
REGISTER_NEOML_LAYER( CSourceLayer, "FmlCnnSourceLayer" )
REGISTER_NEOML_LAYER( CSplitBatchWidthLayer, "FmlCnnSplitBatchWidthLayer" )
REGISTER_NEOML_LAYER( CSplitChannelsLayer, "FmlCnnSplitChannelsLayer" )
REGISTER_NEOML_LAYER( CSplitDepthLayer, "FmlCnnSplitDepthLayer" )
REGISTER_NEOML_LAYER( CSplitHeightLayer, "FmlCnnSplitHeightLayer" )
REGISTER_NEOML_LAYER( CSplitWidthLayer, "FmlCnnSplitWidthLayer" )
REGISTER_NEOML_LAYER( CTimeConvLayer, "FmlCnnTimeConvLayer" )
REGISTER_NEOML_LAYER( CTransformLayer, "FmlCnnTransformWithoutTransposeLayer" )
REGISTER_NEOML_LAYER( CTransposeLayer, "FmlCnnTransposeLayer" )
REGISTER_NEOML_LAYER( CTransposedConvLayer, "FmlCnnTransposedConvLayer" )
// }} CNN

// {{ DNN
REGISTER_NEOML_LAYER( CAddToObjectLayer, "NeoMLDnnAddToObjectLayer" )
REGISTER_NEOML_LAYER( CBroadcastLayer, "NeoMLDnnBroadcastLayer" )
REGISTER_NEOML_LAYER( CCastLayer, "NeoMLDnnCastLayer" )
REGISTER_NEOML_LAYER( CDataLayer, "NeoMLDnnDataLayer" )
REGISTER_NEOML_LAYER( CEltwiseSubLayer, "NeoMLDnnEltwiseSubLayer" )
REGISTER_NEOML_LAYER( CEltwiseDivLayer, "NeoMLDnnEltwiseDivLayer" )
REGISTER_NEOML_LAYER( CGELULayer, "NeoMLDnnGELULayer" )
REGISTER_NEOML_LAYER( CIndRnnLayer, "NeoMLDnnIndRnnLayer" )
REGISTER_NEOML_LAYER( CIndRnnRecurrentLayer, "NeoMLDnnIndRnnRecurrentLayer" )
REGISTER_NEOML_LAYER( CMatrixMultiplicationLayer, "NeoMLDnnMatrixMultiplicationLayer" )
REGISTER_NEOML_LAYER( CMobileNetV2BlockLayer, "NeoMLDnnMobileNetV2BlockLayer" )
REGISTER_NEOML_LAYER( CMultiheadAttentionLayer, "NeoMLDnnMultiheadAttentionLayer" )
REGISTER_NEOML_LAYER( CObjectNormalizationLayer, "NeoMLDnnObjectNormalizationLayer" )
REGISTER_NEOML_LAYER( CParameterLayer, "NeoMLDnnParameterLayer" )
REGISTER_NEOML_LAYER( CQrnnFPoolingLayer, "NeoMLDnnQrnnFPoolingLayer" )
REGISTER_NEOML_LAYER( CQrnnIfPoolingLayer, "NeoMLDnnQrnnIfPoolingLayer" )
REGISTER_NEOML_LAYER( CQrnnLayer, "NeoMLDnnQrnnLayer" )
REGISTER_NEOML_LAYER( CSplitBatchLengthLayer, "NeoMLDnnSplitBatchLengthLayer" )
REGISTER_NEOML_LAYER( CSplitListSizeLayer, "NeoMLDnnSplitListSizeLayer" )
// }} DNN

// {{ Activation Layers
REGISTER_NEOML_LAYER( CAbsLayer, "FmlCnnAbsLayer" )
REGISTER_NEOML_LAYER( CELULayer, "FmlCnnELULayer" )
REGISTER_NEOML_LAYER( CHardSigmoidLayer, "FmlCnnSigmoidTanhLayer" )
REGISTER_NEOML_LAYER( CHardTanhLayer, "FmlCnnHardTanhLayer" )
REGISTER_NEOML_LAYER( CHSwishLayer, "FmlCnnHSwishLayer" )
REGISTER_NEOML_LAYER( CLeakyReLULayer, "FmlCnnLeakyReLULayer" )
REGISTER_NEOML_LAYER( CPowerLayer, "FmlCnnPowerLayer" )
REGISTER_NEOML_LAYER( CReLULayer, "FmlCnnReLULayer" )
REGISTER_NEOML_LAYER( CSigmoidLayer, "FmlCnnSigmoidLayer" )
REGISTER_NEOML_LAYER( CTanhLayer, "FmlCnnTanhLayer" )

REGISTER_NEOML_LAYER( CErfLayer, "NeoMLDnnErfLayer" )
REGISTER_NEOML_LAYER( CExpLayer, "NeoMLDnnExpLayer" )
REGISTER_NEOML_LAYER( CLogLayer, "NeoMLDnnLogLayer" )
// }} Activation Layers

// {{ Loss Layers
REGISTER_NEOML_LAYER( CBinaryCrossEntropyLossLayer, "FmlCnnBinaryCrossEntropyLossLayer" )
REGISTER_NEOML_LAYER( CCrossEntropyLossLayer, "FmlCnnCrossEntropyLossLayer" )
REGISTER_NEOML_LAYER( CEuclideanLossLayer, "FmlCnnEuclideanLossLayer" )
REGISTER_NEOML_LAYER( CHingeLossLayer, "FmlCnnHingeLossLayer" )
REGISTER_NEOML_LAYER( CL1LossLayer, "NeoMLDnnL1LossLayer" )
REGISTER_NEOML_LAYER( CSquaredHingeLossLayer, "FmlCnnSquaredHingeLossLayer" )
// }} Loss Layers

#ifndef NEOML_COMPACT
// {{ CNN
REGISTER_NEOML_LAYER( C3dMaxPoolingLayer, "FmlCnn3dMaxPoolingLayer" )
REGISTER_NEOML_LAYER( C3dMeanPoolingLayer, "FmlCnn3dMeanPoolingLayer" )
REGISTER_NEOML_LAYER( C3dTransposedConvLayer, "FmlCnn3dTransposedConvLayer" )
REGISTER_NEOML_LAYER( CAccumulativeLookupLayer, "FmlCnnAccumulativeLookupLayer" )
REGISTER_NEOML_LAYER( CAccuracyLayer, "FmlCnnAccuracyLayer" )
REGISTER_NEOML_LAYER( CArgmaxLayer, "FmlCnnArgmaxLayer" )
REGISTER_NEOML_LAYER( CAttentionDecoderLayer, "FmlCnnAttentionDecoderLayer" )
REGISTER_NEOML_LAYER( CAttentionDotProductLayer, "FmlCnnAttentionDotProductLayer" )
REGISTER_NEOML_LAYER( CAttentionRecurrentLayer, "FmlCnnAttentionRecurrentLayer" )
REGISTER_NEOML_LAYER( CAttentionLayer, "FmlCnnAttentionLayer" )
REGISTER_NEOML_LAYER( CAttentionSumLayer, "FmlCnnAttentionSumLayer" )
REGISTER_NEOML_LAYER( CAttentionWeightedSumLayer, "FmlCnnAttentionWeightedSumLayer" )
REGISTER_NEOML_LAYER( CBestSequenceLayer, "FmlCnnBestSequenceLayer" )
REGISTER_NEOML_LAYER( CBinaryFocalLossLayer, "FmlCnnBinaryFocalLossLayer" )
REGISTER_NEOML_LAYER( CBitSetVectorizationLayer, "FmlCnnBitSetVectorizationLayerClassName" )
REGISTER_NEOML_LAYER( CCaptureSinkLayer, "FmlCnnCaptureSink" )
REGISTER_NEOML_LAYER( CCenterLossLayer, "FmlCnnCenterLossLayer" )
REGISTER_NEOML_LAYER( CChannelwiseWith1x1Layer, "NeoMLDnnChannelwiseWith1x1Layer" )
REGISTER_NEOML_LAYER( CCompositeSinkLayer, "FmlCompositeCnnSinkLayer" )
REGISTER_NEOML_LAYER( CCompositeSourceLayer, "FmlCnnCompositeSourceLayer" )
REGISTER_NEOML_LAYER( CConfusionMatrixLayer, "FmlCnnConfusionMatrixLayer" )
REGISTER_NEOML_LAYER( CCrfCalculationLayer, "FmlCnnCrfCalculationLayer" )
REGISTER_NEOML_LAYER( CCrfInternalLossLayer, "FmlCnnCrfInternalLossLayer" )
REGISTER_NEOML_LAYER( CCrfLayer, "FmlCnnCrfLayer" )
REGISTER_NEOML_LAYER( CCrfLossLayer, "FmlCnnCrfLossLayer" )
REGISTER_NEOML_LAYER( CCtcDecodingLayer, "FmlCnnCtcDecodingLayer" )
REGISTER_NEOML_LAYER( CCtcLossLayer, "FmlCnnCtcLossLayer" )
REGISTER_NEOML_LAYER( CDotProductLayer, "FmlCnnDotProductLayer" )
REGISTER_NEOML_LAYER( CEnumBinarizationLayer, "FmlCnnEnumBinarizationLayer" )
REGISTER_NEOML_LAYER( CGlobalMaxPoolingLayer, "FmlCnnGlobalMaxPoolingLayer" )
REGISTER_NEOML_LAYER( CGrnLayer, "NeoMLDnnGrnLayer" )
REGISTER_NEOML_LAYER( CGruLayer, "FmlCnnGruLayer" )
REGISTER_NEOML_LAYER( CImageResizeLayer, "FmlCnnImageResizeLayer" )
REGISTER_NEOML_LAYER( CImageToPixelLayer, "FmlCnnImageToPixelLayerClass" )
REGISTER_NEOML_LAYER( CFocalLossLayer, "FmlCnnFocalLossLayer" )
REGISTER_NEOML_LAYER( CFullyConnectedSourceLayer, "FmlCnnFullyConnectedSourceLayer" )
REGISTER_NEOML_LAYER( CLoraFullyConnectedLayer, "NeoMLDnnLoraFullyConnectedLayer" )
REGISTER_NEOML_LAYER( CMaxOverTimePoolingLayer, "FmlCnnMaxOverTimePoolingLayer" )
REGISTER_NEOML_LAYER( CMobileNetV3PreSEBlockLayer, "NeoMLDnnMobileNetV3PreSEBlockLayer" )
REGISTER_NEOML_LAYER( CMobileNetV3PostSEBlockLayer, "NeoMLDnnMobileNetV3PostSEBlockLayer" )
REGISTER_NEOML_LAYER( CMultiHingeLossLayer, "FmlCnnMultyHingeLossLayer" )
REGISTER_NEOML_LAYER( CMultiSquaredHingeLossLayer, "FmlCnnMultySquaredHingeLossLayer" )
REGISTER_NEOML_LAYER( CPixelToImageLayer, "FmlCnnPixelToImageLayerClass" )
REGISTER_NEOML_LAYER( CPrecisionRecallLayer, "FmlCnnPrecisionRecallLayer" )
REGISTER_NEOML_LAYER( CProblemSourceLayer, "FmlCnnProblemSourceLayer" )
REGISTER_NEOML_LAYER( CProjectionPoolingLayer, "FmlCnnProjectionPoolingLayerClass" )
REGISTER_NEOML_LAYER( CReorgLayer, "FmlCnnReorgLayerClass" )
REGISTER_NEOML_LAYER( CRepeatSequenceLayer, "FmlCnnRepeatSequenceLayer" )
REGISTER_NEOML_LAYER( CSequenceSumLayer, "FmlCnnSequenceSumLayer" )
REGISTER_NEOML_LAYER( CSubSequenceLayer, "FmlCnnSubSequenceLayer" )
REGISTER_NEOML_LAYER( CTiedEmbeddingsLayer, "TiedEmbeddingsLayer" )
REGISTER_NEOML_LAYER( CUpsampling2DLayer, "FmlCnnUpsampling2DLayer" )
// }} CNN

// {{ DNN
REGISTER_NEOML_LAYER( CBertConvLayer, "NeoMLDnnBertConvLayer" )
REGISTER_NEOML_LAYER( CCumSumLayer, "NeoMLDnnCumSumLayer" )
REGISTER_NEOML_LAYER( CDepthToSpaceLayer, "NeoMLDnnDepthToSpaceLayer" )
REGISTER_NEOML_LAYER( CEqualLayer, "NeoMLDnnEqualLayer" )
REGISTER_NEOML_LAYER( CGlobalSumPoolingLayer, "NeoMLDnnGlobalSumPoolingLayer" )
REGISTER_NEOML_LAYER( CInterpolationLayer, "NeoMLDnnInterpolationLayer" )
REGISTER_NEOML_LAYER( CIrnnLayer, "NeoMLDnnIrnnLayer" )
REGISTER_NEOML_LAYER( CLessLayer, "NeoMLDnnLessLayer" )
REGISTER_NEOML_LAYER( CLrnLayer, "NeoMLDnnLrnLayer" )
REGISTER_NEOML_LAYER( CNotLayer, "NeoMLDnnNotLayer" )
REGISTER_NEOML_LAYER( CPositionalEmbeddingLayer, "NeoMLDnnPositionalEmbeddingLayer" )
REGISTER_NEOML_LAYER( CRowwiseOperationChainLayer, "NeoMLDnnRowwiseOperationChainLayer" )
REGISTER_NEOML_LAYER( CScatterNDLayer, "NeoMLDnnScatterNDLayer" )
REGISTER_NEOML_LAYER( CSpaceToDepthLayer, "NeoMLDnnSpaceToDepthLayer" )
REGISTER_NEOML_LAYER( CTransformerEncoderLayer, "NeoMLDnnTransformerEncoderLayer" )
REGISTER_NEOML_LAYER( CTransformerSourceMaskLayer, "NeoMLDnnTransformerSourceMaskLayer" )
REGISTER_NEOML_LAYER( CWhereLayer, "NeoMLDnnWhereLayer" )
// }} DNN
#endif //!NEOML_COMPACT

} // namespace

//---------------------------------------------------------------------------------------------------------

CDnn::CDnn( CRandom& _random, IMathEngine& _mathEngine, const CCompositeLayer* _owner ) :
	random( _random ),
	mathEngine( _mathEngine ),
	solver( FINE_DEBUG_NEW CDnnSimpleGradientSolver( mathEngine ) ),
	initializer( FINE_DEBUG_NEW CDnnXavierInitializer( random ) ),
	owner( _owner )
{
}

CDnn::~CDnn()
{
	referenceDnnInfo.Release();
	for( int i = layers.Size() - 1; i >= 0; i-- ) {
		CPtr<CBaseLayer> layer = layers[i];
		DeleteLayer( *layer );
		layer->setDnn( 0 );
	}
}

void CDnn::GetLayerList( CArray<const char*>& layerList ) const
{
	layerList.SetSize( layers.Size() );

	for( int i = 0; i < layers.Size(); ++i ) {
		layerList[i] = layers[i]->GetName();
	}
}

CPtr<CBaseLayer> CDnn::GetLayer( const char* name )
{
	CBaseLayer* layer = getLayer( name );

	NeoAssertMsg( !IsReferenceDnn()
		|| dynamic_cast<CSourceLayer*>( layer ) != nullptr
		|| dynamic_cast<CSinkLayer*>( layer ) != nullptr,
		"For ReferenceDnn changing layers is restricted. Use const version instead." );
	return layer;
}

CBaseLayer* CDnn::getLayer( const char* name )
{
	CheckArchitecture( layerMap.Has( name ), name, "layer is not in this dnn" );
	return layerMap.Get( name );
}

CPtr<const CBaseLayer> CDnn::GetLayer( const char* name ) const
{
	CheckArchitecture( layerMap.Has( name ), name, "layer is not in this dnn" );
	return layerMap.Get( name );
}

CPtr<CBaseLayer> CDnn::GetLayer( const CArray<CString>& path )
{
	CBaseLayer* layer = getLayer( path );

	NeoAssertMsg( !IsReferenceDnn()
		|| dynamic_cast<CSourceLayer*>( layer ) != nullptr
		|| dynamic_cast<CSinkLayer*>( layer ) != nullptr,
		"For ReferenceDnn changing layers is restricted. Use const version instead." );
	return layer;
}

CBaseLayer* CDnn::getLayer( const CArray<CString>& path )
{
	CheckArchitecture(path.Size() > 0, "NULL", "can not find layer - empty path");
	if (path.Size() == 1) {
		return getLayer( path[0] );
	} else {
		CheckArchitecture(layerMap.Has(path[0]), path[0], "layer is not in this dnn");
		CPtr<CCompositeLayer> currComp = CheckCast<CCompositeLayer>( getLayer( path[0] ) );
		for (int i = 1; i < path.Size() - 1; ++i) {
			CheckArchitecture(currComp->HasLayer(path[i]), path[i], "layer is not in this composite layer");
			currComp = CheckCast<CCompositeLayer>(currComp->GetLayer(path[i]).Ptr());
		}
		CheckArchitecture(currComp->HasLayer(path.Last()), path.Last(), "layer is not contained by this path");
		return currComp->GetLayer(path.Last());
	}
}

CPtr<const CBaseLayer> CDnn::GetLayer(const CArray<CString>& path) const
{
	return const_cast<CDnn*>(this)->getLayer(path);
}

void CDnn::AddLayerImpl( CBaseLayer& layer )
{
	NeoAssertMsg( !IsReferenceDnn()
		|| dynamic_cast<CCompositeSourceLayer*>( &layer ) != nullptr
		|| dynamic_cast<CCompositeSinkLayer*>( &layer ) != nullptr,
		"For ReferenceDnn adding layers is restricted" );
	layer.CheckLayerArchitecture( !layerMap.Has( layer.GetName() ), "layer already in this dnn" );
	layer.CheckLayerArchitecture( layer.GetDnn() == 0, "layer already added to other dnn" );

	// Set the flag that indicates the network must be rebuilt (configuration has changed)
	ForceRebuild();

	// Add a layer
	layerMap.Add( layer.GetName(), &layer );
	layers.Add( &layer );

	// Set the layer network
	layer.setDnn( this );
}

void CDnn::ForceRebuild()
{
	isRebuildNeeded = true;
	sinkLayers.SetSize( 0 );
	sourceLayers.SetSize( 0 );
}

void CDnn::DeleteLayerImpl( CBaseLayer& layer )
{
	NeoAssertMsg( !IsReferenceDnn()
		|| dynamic_cast<CCompositeSourceLayer*>( &layer ) != nullptr
		|| dynamic_cast<CCompositeSinkLayer*>( &layer ) != nullptr,
		"For ReferenceDnn deleting layers is restricted" );
	layer.CheckLayerArchitecture( HasLayer( layer.GetName() ), "deletion of the layer which is not in this dnn" );

	// Set the flag that indicates the network should be rebuilt (configuration has changed)
	ForceRebuild();
	// Unlink all layer connections
	layer.unlink();
	// Delete the layer from the table
	layerMap.Delete( layer.GetName() );

	// Set the network for the layer
	layer.setDnn( 0 );
	// Delete the layer from the array that owns the data
	int oldSize = layers.Size();
	for( int i = 0; i < layers.Size(); i++ ) {
		if( layers[i] == &layer ) {
			layers.DeleteAt( i );
			break;
		}
	}
	NeoAssert( layers.Size() < oldSize );
}

void CDnn::RestartSequence()
{
	for( int i = 0; i < layers.Size(); i++ ) {
		layers[i]->RestartSequence();
	}
}

void CDnn::DisableLearning()
{
	if( !isLearningEnabled ) {
		return;
	}
	isLearningEnabled = false;
	RequestReshape( /*forcedReshape*/true );
}

void CDnn::EnableLearning()
{
	NeoAssertMsg( !IsReferenceDnn(), "For ReferenceDnn learning is restricted" );
	if( isLearningEnabled ) {
		return;
	}
	isLearningEnabled = true;
	RequestReshape( /*forcedReshape*/true );
}

void CDnn::RequestReshape( bool forcedReshape )
{
	for( int i = 0; i < layers.Size(); i++ ) {
		layers[i]->isReshapeNeeded = true;
		layers[i]->forcedReshape = layers[i]->forcedReshape || forcedReshape;
	}
}

void CDnn::SetSolver( CDnnSolver* _solver )
{
	if( solver.Ptr() == _solver ) {
		return;
	}
	solver = _solver;
}

// Sets the network operation parameters
void CDnn::setProcessingParams( bool _isRecurrentMode, int sequenceLength, bool _isReverseSequense, bool _isBackwardPerformed )
{
	isRecurrentMode = _isRecurrentMode;
	maxSequenceLength = sequenceLength;
	NeoAssert( isRecurrentMode || maxSequenceLength == 1 );
	isReverseSequense = _isReverseSequense;
	if( !isReverseSequense ) {
		currentSequencePos = 0;
	} else {
		currentSequencePos = sequenceLength - 1;
	}
	isBackwardPerformed = _isBackwardPerformed;
}

void CDnn::runOnce( int curSequencePos )
{
	currentSequencePos = curSequencePos;
	++runNumber;

	if( IsLogging() ) {
		*log << "Run " << runNumber << " : " << currentSequencePos;
	}
	// Run the network for each sink layer; they will recursively call RunOnce for all their inputs
	for( int i = 0; i < sinkLayers.Size(); ++i ) {
		sinkLayers[i]->runOnce();

		if( IsLogging() ) {
			CLossLayer* loss = dynamic_cast<CLossLayer*>( sinkLayers[i] );
			if( loss != 0 ) {
				*log << ", loss = " << loss->GetLastLoss();
			}
		}
	}
	if( IsLogging() ) {
		*log << "\n";
	}
}

void CDnn::RunOnce()
{
	try {
		NeoAssert( maxSequenceLength == 1 );
		if( isBackwardPerformed ) {
			// The layer Reshape methods depend on IsBackwardPerformed()
			RequestReshape( /*forcedReshape*/true );
		}
		isBackwardPerformed = false;
		if( autoRestartMode ) {
			RestartSequence();
		}
		reshape(); // rebuild the network if necessary

		// During inference we turning reuseMemoryMode on when the net is big enough
		isReuseMemoryMode = ( getOutputBlobsSize() > MinReuseMemoryModeNetSize );
		runOnce( 0 );
#ifdef NEOML_USE_FINEOBJ
	} catch( CCheckException* exception ) {
#else
	} catch( CCheckException& exception ) {
#endif
		if( IsLogging() ) {
			*log << "CCheckException in RunOnce\n";
#ifdef NEOML_USE_FINEOBJ
			*log << "\t" << exception->MessageText() << "\n";
#else
			*log << "\t" << exception.what() << "\n";
#endif
		}
		throw;
	}
}

void CDnn::RunAndBackwardOnce()
{
	try {
		NeoAssert( maxSequenceLength == 1 );
		if( !isBackwardPerformed ) {
			// The layer Reshape methods depend on IsBackwardPerformed()
			RequestReshape( /*forcedReshape*/true );
		}
		isBackwardPerformed = true;
		if( autoRestartMode ) {
			RestartSequence();
		}
		reshape(); // rebuild the network if necessary

		// During training we don't reuse memory only when training on nonDistributed CPU
		CMathEngineInfo info;
		mathEngine.GetMathEngineInfo( info );
		isReuseMemoryMode = info.Type != MET_Cpu || mathEngine.IsDistributed();
		runOnce( 0 );
		backwardRunAndLearnOnce( 0 );
#ifdef NEOML_USE_FINEOBJ
	} catch( CCheckException* exception ) {
#else
	} catch( CCheckException& exception ) {
#endif
		if( IsLogging() ) {
			*log << "CCheckException in RunAndLearnOnce\n";
#ifdef NEOML_USE_FINEOBJ
			*log << "\t" << exception->MessageText() << "\n";
#else
			*log << "\t" << exception.what() << "\n";
#endif
		}
		throw;
	}
}

void CDnn::RunAndLearnOnce()
{
	RunAndBackwardOnce();
	solver->Train();
}

void CDnn::CleanUp( bool totalCleanUp )
{
	for( int i = 0; i < layers.Size(); i++ ) {
		layers[i]->CleanUp( totalCleanUp );
	}
}

void CDnn::backwardRunAndLearnOnce( int curSequencePos )
{
	currentSequencePos = curSequencePos;
	if( IsLogging() ) {
		*log << "Backward & Learn " << runNumber << " : " << currentSequencePos;
	}
	// Run the network for each sink layer; they will recursively call RunOnce for all their inputs
	for( int i = 0; i < sinkLayers.Size(); ++i ) {
		sinkLayers[i]->backwardRunAndLearnOnce();
	}
	if( IsLogging() ) {
		*log << "\n";
	}
}

void CDnn::reshape()
{
	rebuild(); // rebuild the network if necessary

	// Check if backward propagation is required
	for( int i = 0; i < layers.Size(); ++i ) {
		layers[i]->isBackwardNeeded = CBaseLayer::BS_Unknown;
	}
	for( int i = 0; i < sinkLayers.Size(); ++i ) {
		sinkLayers[i]->recheckBackwardNeeded();
	}
	// Call reshape for each sink layer; they will recursively reshape all their inputs
	for( int i = 0; i < sinkLayers.Size(); ++i ) {
		sinkLayers[i]->reshape();
	}
}

// Rebuilds the network
void CDnn::rebuild()
{
	if( !isRebuildNeeded ) {
		return;
	}
	isRebuildNeeded = false;

	if( solver != 0 ) {
		solver->Reset();
	}

	// Unlink all layers
	for( int i = 0; i < layers.Size(); i++ ) {
		layers[i]->unlink();
	}
	// Clear up the sink layers
	sinkLayers.DeleteAll();
	sourceLayers.DeleteAll();

	// Link the layers again
	for( int i = 0; i < layers.Size(); i++ ) {
		layers[i]->link();
	}

	// Recalculate the source and sink layers
	for( int i = 0; i < layers.Size(); i++ ) {
		const CBaseLayer* layer = layers[i];
		if( layer->GetInputCount() == 0 ) {
			sourceLayers.Add( layers[i] );
		}
		if( layer->GetOutputCount() == 0 ) {
			sinkLayers.Add( layers[i] );
		}
	}

	for( int i = 0; i < sinkLayers.Size(); ++i ) {
		sinkLayers[i]->buildOrder();
	}

	RequestReshape( /*forcedReshape*/true );
}

size_t CDnn::getOutputBlobsSize() const
{
	size_t result = 0;
	for( int i = 0; i < layers.Size(); i++ ) {
		result += layers[i]->GetOutputBlobsSize();
	}
	return result;
}

void CDnn::FilterLayersParams( float threshold )
{
	NeoAssertMsg( !IsReferenceDnn(), "For ReferenceDnn filtering layers parameters is restricted" );
	for( int i = 0; i < layers.Size(); ++i ) {
		layers[i]->FilterLayerParams( threshold );
	}
}

void CDnn::FilterLayersParams( const CArray<const char*>& layers, float threshold )
{
	NeoAssertMsg( !IsReferenceDnn(), "For ReferenceDnn filtering layers parameters is restricted" );
	for( int i = 0; i < layers.Size(); ++i ) {
		GetLayer( layers[i] )->FilterLayerParams( threshold );
	}
}

static constexpr int dnnVersion = 2000;

void CDnn::Serialize( CArchive& archive )
{
	NeoAssertMsg( !IsReferenceDnn(), "For ReferenceDnn serializing is restricted" );

	int version = dnnVersion;
	archive.Serialize( version );

	if( archive.IsLoading() ) {
		// Calculate the data
		if( version < 0 ) {
			version = -version;
		}
		if( version < CDnn::ArchiveMinSupportedVersion || version > dnnVersion ) {
			check( false, ERR_BAD_ARCHIVE_VERSION, archive.Name() );
		}

		// Clean up the network
		while( layers.Size() > 0 ) {
			DeleteLayer( *layers[0] );
		}
		runNumber = 0;
		isRebuildNeeded = false;
		isBackwardPerformed = false;
	}

	archive.Serialize( logFrequency );

	int layersSize = layers.Size();
	archive.Serialize( layersSize );
	for( int i = 0; i < layersSize; ++i ) {
		CPtr<CBaseLayer> layer;
		if( archive.IsStoring() ) {
			layer = layers[i];
		}
		SerializeLayer( archive, mathEngine, layer );
		if( archive.IsLoading() ) {
			AddLayer( *layer );
		}
	}

	archive.Serialize( isLearningEnabled );

	if( archive.IsLoading() ) {
		// In order to avoid the CDnnSolver::Reset for the next solver
		rebuild();
	}
}

void CDnn::SerializeCheckpoint( CArchive& archive )
{
	Serialize( archive );
	CPtr<CDnnSolver> solverPtr = nullptr;
	if( archive.IsStoring() ) {
		solverPtr = GetSolver();
	}
	SerializeSolver( archive, *this, solverPtr );
	if( archive.IsLoading() ) {
		SetSolver( solverPtr );
	}
}

void CDnn::EnableProfile( bool profile )
{
	for( int i = 0; i < layers.Size(); ++i ) {
		layers[i]->EnableProfile( profile );
	}
}

} // namespace NeoML
