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

#include <NeoML/Dnn/Dnn.h>
#include <NeoML/Dnn/Layers/BaseInPlaceLayer.h>
#include <NeoML/Dnn/Layers/SourceLayer.h>
#include <NeoML/Dnn/Layers/SinkLayer.h>
#include <NeoML/Dnn/Layers/ConcatLayer.h>
#include <NeoML/Dnn/Layers/SplitLayer.h>
#include <NeoML/Dnn/Layers/EltwiseLayer.h>
#include <NeoML/Dnn/Layers/LossLayer.h>
#include <NeoML/Dnn/Layers/ConvLayer.h>
#include <NeoML/Dnn/Layers/FullyConnectedLayer.h>
#include <NeoML/Dnn/Layers/FullyConnectedSourceLayer.h>
#include <NeoML/Dnn/Layers/ActivationLayers.h>
#include <NeoML/Dnn/Layers/PoolingLayer.h>
#include <NeoML/Dnn/Layers/PositionalEmbeddingLayer.h>
#include <NeoML/Dnn/Layers/ModelWrapperLayer.h>
#include <NeoML/Dnn/Layers/BatchNormalizationLayer.h>
#include <NeoML/Dnn/Layers/ObjectNormalizationLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoML/Dnn/Layers/DropoutLayer.h>
#include <NeoML/Dnn/Layers/MultichannelLookupLayer.h>
#include <NeoML/Dnn/Layers/ImageResizeLayer.h>
#include <NeoML/Dnn/Layers/CompositeLayer.h>
#include <NeoML/Dnn/Layers/RecurrentLayer.h>
#include <NeoML/Dnn/Layers/BackLinkLayer.h>
#include <NeoML/Dnn/Layers/SubSequenceLayer.h>
#include <NeoML/Dnn/Layers/EnumBinarizationLayer.h>
#include <NeoML/Dnn/Layers/SoftmaxLayer.h>
#include <NeoML/Dnn/Layers/GlobalMeanPoolingLayer.h>
#include <NeoML/Dnn/Layers/GlobalMaxPoolingLayer.h>
#include <NeoML/Dnn/Layers/GruLayer.h>
#include <NeoML/Dnn/Layers/LstmLayer.h>
#include <NeoML/Dnn/Layers/MaxOverTimePoolingLayer.h>
#include <NeoML/Dnn/Layers/3dConvLayer.h>
#include <NeoML/Dnn/Layers/3dPoolingLayer.h>
#include <NeoML/Dnn/Layers/TimeConvLayer.h>
#include <NeoML/Dnn/Layers/TransposedConvLayer.h>
#include <NeoML/Dnn/Layers/3dTransposedConvLayer.h>
#include <NeoML/Dnn/Layers/AttentionLayer.h>
#include <NeoML/Dnn/Layers/CrfLayer.h>
#include <NeoML/Dnn/Layers/SequenceSumLayer.h>
#include <NeoML/Dnn/Layers/CtcLayer.h>
#include <NeoML/Dnn/Layers/MultiHingeLossLayer.h>
#include <NeoML/Dnn/Layers/Upsampling2DLayer.h>
#include <NeoML/Dnn/Layers/ChannelwiseConvLayer.h>
#include <NeoML/Dnn/Layers/AccumulativeLookupLayer.h>
#include <NeoML/Dnn/Layers/QualityControlLayer.h>
#include <NeoML/Dnn/Layers/AccuracyLayer.h>
#include <NeoML/Dnn/Layers/PrecisionRecallLayer.h>
#include <NeoML/Dnn/Layers/CenterLossLayer.h>
#include <NeoML/Dnn/Layers/FocalLossLayer.h>
#include <NeoML/Dnn/Layers/BinaryFocalLossLayer.h>
#include <NeoML/Dnn/Layers/ImageAndPixelConversionLayer.h>
#include <NeoML/Dnn/Layers/TransposeLayer.h>
#include <NeoML/Dnn/Layers/TransformLayer.h>
#include <NeoML/Dnn/Layers/ArgmaxLayer.h>
#include <NeoML/Dnn/Layers/RepeatSequenceLayer.h>
#include <NeoML/Dnn/Layers/DotProductLayer.h>
#include <NeoML/Dnn/Layers/ReorgLayer.h>
#include <NeoML/Dnn/Layers/AddToObjectLayer.h>
#include <NeoML/Dnn/Layers/MatrixMultiplicationLayer.h>
#include <NeoML/Dnn/Layers/MultiheadAttentionLayer.h>
#include <NeoML/Dnn/Layers/GELULayer.h>

namespace NeoML {

// The minimum size of temporary data blobs to start reusing memory
static const size_t MinReuseMemoryModeNetSize = 4 * 1024 * 1024;

static CMap<CString, TCreateLayerFunction, CDefaultHash<CString>, RuntimeHeap>& getRegisteredLayers()
{
	static CMap<CString, TCreateLayerFunction, CDefaultHash<CString>, RuntimeHeap> registeredLayers;
	return registeredLayers;
}


// Class name hash to compare type_info
struct CTypeInfoNameHash {
	static int HashKey( const std::type_info* key )
	{ 
		return GetMBCStringHash( key->name() );
	}

	static bool IsEqual( const std::type_info* first, const std::type_info* second )
	{
		return ( ::strcmp( first->name(), second->name() ) == 0 );
	}
};

static CMap<const std::type_info*, CString, CTypeInfoNameHash, RuntimeHeap>& getLayerNames()
{
	static CMap<const std::type_info*, CString, CTypeInfoNameHash, RuntimeHeap> layerNames;
	return layerNames;
}

void RegisterLayerName( const char* mainName, const char* additionalName, const std::type_info& typeInfo, TCreateLayerFunction function )
{
	NeoAssert( !getRegisteredLayers().Has( mainName ) );
	getRegisteredLayers().Add( mainName, function );
	if( additionalName != 0 ) {
		NeoAssert( !getRegisteredLayers().Has( additionalName ) );
		getRegisteredLayers().Add( additionalName, function );
	}
	getLayerNames().Add( &typeInfo, mainName );
}

void UnregisterLayerName( const std::type_info& typeInfo )
{
	getRegisteredLayers().Delete( getLayerNames().Get( &typeInfo ) );
	getLayerNames().Delete( &typeInfo );
}

static CPtr<CBaseLayer> createLayer( IMathEngine& mathEngine, const CString& className )
{
	TMapPosition pos = getRegisteredLayers().GetFirstPosition( className );
	if( pos == NotFound ) {
		return 0;
	}
	return getRegisteredLayers().GetValue( pos )( mathEngine );
}

static CString getLayerName( CBaseLayer* layer )
{
	if( layer == 0 ) {
		return CString();
	}
	const std::type_info& layerType = typeid( *layer );
	TMapPosition pos = getLayerNames().GetFirstPosition( &layerType );
	if( pos == NotFound ) {
		return CString();
	}
	return getLayerNames().GetValue( pos );
}

void NEOML_API SerializeLayer( CArchive& archive, IMathEngine& mathEngine, CPtr<CBaseLayer>& layer )
{
	if( archive.IsStoring() ) {
		archive << getLayerName( layer );
		if( layer != 0 ) {
			layer->Serialize( archive );
		}
	} else if( archive.IsLoading() ) {
		CString name;
		archive >> name;
		layer = createLayer( mathEngine, name );
		if( layer != 0 ) {
			layer->Serialize( archive );
		}
	} else {
		NeoAssert( false );
	}
}

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

// Register all layer types
namespace {

REGISTER_NEOML_LAYER( CSourceLayer, "FmlCnnSourceLayer" )
REGISTER_NEOML_LAYER( CSinkLayer, "FmlCnnSinkLayer" )
REGISTER_NEOML_LAYER( CConcatChannelsLayer, "FmlCnnConcatChannelsLayer" )
REGISTER_NEOML_LAYER( CConcatDepthLayer, "FmlCnnConcatDepthLayer" )
REGISTER_NEOML_LAYER( CConcatWidthLayer, "FmlCnnConcatWidthLayer" )
REGISTER_NEOML_LAYER( CConcatHeightLayer, "FmlCnnConcatHeightLayer" )
REGISTER_NEOML_LAYER( CConcatBatchWidthLayer, "FmlCnnConcatBatchWidthLayer" )
REGISTER_NEOML_LAYER( CConcatObjectLayer, "FmlCnnConcatObjectLayer" )
REGISTER_NEOML_LAYER( CSplitChannelsLayer, "FmlCnnSplitChannelsLayer" )
REGISTER_NEOML_LAYER( CSplitDepthLayer, "FmlCnnSplitDepthLayer" )
REGISTER_NEOML_LAYER( CSplitWidthLayer, "FmlCnnSplitWidthLayer" )
REGISTER_NEOML_LAYER( CSplitHeightLayer, "FmlCnnSplitHeightLayer" )
REGISTER_NEOML_LAYER( CSplitBatchWidthLayer, "FmlCnnSplitBatchWidthLayer" )
REGISTER_NEOML_LAYER( CEltwiseSumLayer, "FmlCnnEltwiseSumLayer" )
REGISTER_NEOML_LAYER( CEltwiseMulLayer, "FmlCnnEltwiseMulLayer" )
REGISTER_NEOML_LAYER( CEltwiseNegMulLayer, "FmlCnnEltwiseNegMulLayer" )
REGISTER_NEOML_LAYER( CEltwiseMaxLayer, "FmlCnnEltwiseMaxLayer" )
REGISTER_NEOML_LAYER( CELULayer, "FmlCnnELULayer" )
REGISTER_NEOML_LAYER( CReLULayer, "FmlCnnReLULayer" )
REGISTER_NEOML_LAYER( CLeakyReLULayer, "FmlCnnLeakyReLULayer" )
REGISTER_NEOML_LAYER( CAbsLayer, "FmlCnnAbsLayer" )
REGISTER_NEOML_LAYER( CSigmoidLayer, "FmlCnnSigmoidLayer" )
REGISTER_NEOML_LAYER( CTanhLayer, "FmlCnnTanhLayer" )
REGISTER_NEOML_LAYER( CHardTanhLayer, "FmlCnnHardTanhLayer" )
REGISTER_NEOML_LAYER( CHardSigmoidLayer, "FmlCnnSigmoidTanhLayer" )
REGISTER_NEOML_LAYER( CHSwishLayer, "FmlCnnHSwishLayer" )
REGISTER_NEOML_LAYER( CPowerLayer, "FmlCnnPowerLayer" )
REGISTER_NEOML_LAYER( CConvLayer, "FmlCnnConvLayer" )
REGISTER_NEOML_LAYER( CRleConvLayer, "FmlCnnRleConvLayer" )
REGISTER_NEOML_LAYER( CMaxPoolingLayer, "FmlCnnMaxPoolingLayer" )
REGISTER_NEOML_LAYER( CMeanPoolingLayer, "FmlCnnMeanPoolingLayer" )
REGISTER_NEOML_LAYER( CFullyConnectedLayer, "FmlCnnFullyConnectedLayer" )
REGISTER_NEOML_LAYER( CFullyConnectedSourceLayer, "FmlCnnFullyConnectedSourceLayer" )
REGISTER_NEOML_LAYER( CCrossEntropyLossLayer, "FmlCnnCrossEntropyLossLayer" )
REGISTER_NEOML_LAYER( CBinaryCrossEntropyLossLayer, "FmlCnnBinaryCrossEntropyLossLayer" )
REGISTER_NEOML_LAYER( CEuclideanLossLayer, "FmlCnnEuclideanLossLayer" )
REGISTER_NEOML_LAYER( CHingeLossLayer, "FmlCnnHingeLossLayer" )
REGISTER_NEOML_LAYER( CSquaredHingeLossLayer, "FmlCnnSquaredHingeLossLayer" )
REGISTER_NEOML_LAYER( CProblemSourceLayer, "FmlCnnProblemSourceLayer" )
REGISTER_NEOML_LAYER( CBatchNormalizationLayer, "FmlCnnBatchNormalizationLayer" )
REGISTER_NEOML_LAYER( CObjectNormalizationLayer, "NeoMLDnnObjectNormalizationLayer" )
REGISTER_NEOML_LAYER( CLinearLayer, "FmlCnnLinearLayer" )
REGISTER_NEOML_LAYER( CDropoutLayer, "FmlCnnDropoutLayer" )
REGISTER_NEOML_LAYER( CImageResizeLayer, "FmlCnnImageResizeLayer" )
REGISTER_NEOML_LAYER( CMultichannelLookupLayer, "FmlCnnMultychannelLookupLayer" )
REGISTER_NEOML_LAYER( CCompositeLayer, "FmlCnnCompositeLayer" )
REGISTER_NEOML_LAYER( CRecurrentLayer, "FmlCnnRecurrentLayer" )
REGISTER_NEOML_LAYER( CSubSequenceLayer, "FmlCnnSubSequenceLayer" )
REGISTER_NEOML_LAYER( CBackLinkLayer, "FmlCnnBackLink" )
REGISTER_NEOML_LAYER( CCaptureSinkLayer, "FmlCnnCaptureSink" )
REGISTER_NEOML_LAYER( CEnumBinarizationLayer, "FmlCnnEnumBinarizationLayer" )
REGISTER_NEOML_LAYER( CBitSetVectorizationLayer, "FmlCnnBitSetVectorizationLayerClassName" )
REGISTER_NEOML_LAYER_EX( CSoftmaxLayer, "FmlCnnSoftmaxLayer", "FmlCCnnChannelwiseSoftmaxLayer" )
REGISTER_NEOML_LAYER_EX( CGlobalMeanPoolingLayer, "FmlCnnGlobalMainPoolingLayer", "FmlCnnGlobalAveragePoolingLayer" )
REGISTER_NEOML_LAYER( CGlobalMaxPoolingLayer, "FmlCnnGlobalMaxPoolingLayer" )
REGISTER_NEOML_LAYER( CLstmLayer, "FmlCnnLstmLayer" )
REGISTER_NEOML_LAYER( CGruLayer, "FmlCnnGruLayer" )
REGISTER_NEOML_LAYER( CMaxOverTimePoolingLayer, "FmlCnnMaxOverTimePoolingLayer" )
REGISTER_NEOML_LAYER( CTimeConvLayer, "FmlCnnTimeConvLayer" )
REGISTER_NEOML_LAYER( C3dConvLayer, "FmlCnn3dConvLayer" )
REGISTER_NEOML_LAYER( C3dMaxPoolingLayer, "FmlCnn3dMaxPoolingLayer" )
REGISTER_NEOML_LAYER( C3dMeanPoolingLayer, "FmlCnn3dMeanPoolingLayer" )
REGISTER_NEOML_LAYER( CTransposedConvLayer, "FmlCnnTransposedConvLayer" )
REGISTER_NEOML_LAYER( C3dTransposedConvLayer, "FmlCnn3dTransposedConvLayer" )
REGISTER_NEOML_LAYER( CCrfLayer, "FmlCnnCrfLayer" )
REGISTER_NEOML_LAYER( CCrfCalculationLayer, "FmlCnnCrfCalculationLayer" )
REGISTER_NEOML_LAYER( CCrfLossLayer, "FmlCnnCrfLossLayer" )
REGISTER_NEOML_LAYER( CCrfInternalLossLayer, "FmlCnnCrfInternalLossLayer" )
REGISTER_NEOML_LAYER( CSequenceSumLayer, "FmlCnnSequenceSumLayer" )
REGISTER_NEOML_LAYER( CBestSequenceLayer, "FmlCnnBestSequenceLayer" )
REGISTER_NEOML_LAYER( CCtcLossLayer, "FmlCnnCtcLossLayer" )
REGISTER_NEOML_LAYER( CCtcDecodingLayer, "FmlCnnCtcDecodingLayer" )
REGISTER_NEOML_LAYER( CMultiHingeLossLayer, "FmlCnnMultyHingeLossLayer" )
REGISTER_NEOML_LAYER( CMultiSquaredHingeLossLayer, "FmlCnnMultySquaredHingeLossLayer" )
REGISTER_NEOML_LAYER( CUpsampling2DLayer, "FmlCnnUpsampling2DLayer" )
REGISTER_NEOML_LAYER( CChannelwiseConvLayer, "FmlCnnChannelwiseConvLayer" )
REGISTER_NEOML_LAYER( CAccumulativeLookupLayer, "FmlCnnAccumulativeLookupLayer" )
REGISTER_NEOML_LAYER( CAccuracyLayer, "FmlCnnAccuracyLayer" )
REGISTER_NEOML_LAYER( CConfusionMatrixLayer, "FmlCnnConfusionMatrixLayer" )
REGISTER_NEOML_LAYER( CPrecisionRecallLayer, "FmlCnnPrecisionRecallLayer" )
REGISTER_NEOML_LAYER( CCenterLossLayer, "FmlCnnCenterLossLayer" )
REGISTER_NEOML_LAYER( CFocalLossLayer, "FmlCnnFocalLossLayer" )
REGISTER_NEOML_LAYER( CBinaryFocalLossLayer, "FmlCnnBinaryFocalLossLayer" )
REGISTER_NEOML_LAYER( CImageToPixelLayer, "FmlCnnImageToPixelLayerClass" )
REGISTER_NEOML_LAYER( CPixelToImageLayer, "FmlCnnPixelToImageLayerClass" )
REGISTER_NEOML_LAYER( CTransposeLayer, "FmlCnnTransposeLayer" )
REGISTER_NEOML_LAYER( CTransformLayer, "FmlCnnTransformWithoutTransposeLayer" )
REGISTER_NEOML_LAYER( CArgmaxLayer, "FmlCnnArgmaxLayer" )
REGISTER_NEOML_LAYER( CAttentionDecoderLayer, "FmlCnnAttentionDecoderLayer" )
REGISTER_NEOML_LAYER( CAttentionRecurrentLayer, "FmlCnnAttentionRecurrentLayer" )
REGISTER_NEOML_LAYER( CAttentionLayer, "FmlCnnAttentionLayer" )
REGISTER_NEOML_LAYER( CRepeatSequenceLayer, "FmlCnnRepeatSequenceLayer" )
REGISTER_NEOML_LAYER( CDotProductLayer, "FmlCnnDotProductLayer" )
REGISTER_NEOML_LAYER( CReorgLayer, "FmlCnnReorgLayerClass" )
REGISTER_NEOML_LAYER( CCompositeSourceLayer, "FmlCnnCompositeSourceLayer" )
REGISTER_NEOML_LAYER( CCompositeSinkLayer, "FmlCompositeCnnSinkLayer" )
REGISTER_NEOML_LAYER( CAttentionWeightedSumLayer, "FmlCnnAttentionWeightedSumLayer" )
REGISTER_NEOML_LAYER( CAttentionDotProductLayer, "FmlCnnAttentionDotProductLayer" )
REGISTER_NEOML_LAYER( CAttentionSumLayer, "FmlCnnAttentionSumLayer" )
REGISTER_NEOML_LAYER( CAddToObjectLayer, "NeoMLDnnAddToObjectLayer" )
REGISTER_NEOML_LAYER( CMatrixMultiplicationLayer, "NeoMLDnnMatrixMultiplicationLayer" )
REGISTER_NEOML_LAYER( CMultiheadAttentionLayer, "NeoMLDnnMultiheadAttentionLayer" )
REGISTER_NEOML_LAYER( CPositionalEmbeddingLayer, "NeoMLDnnPositionalEmbeddingLayer" )
REGISTER_NEOML_LAYER( CGELULayer, "NeoMLDnnGELULayer" )

}

///////////////////////////////////////////////////////////////////////////////////////////

CDnn::CDnn( CRandom& _random, IMathEngine& _mathEngine ) :
	log( 0 ),
	logFrequency( 100 ),
	random( _random ),
	mathEngine( _mathEngine ),
	runNumber( -1 ),
	isRebuildNeeded( false ),
	isBackwardPerformed( false ),
	isLearningEnabled( true ),
	isRecurrentMode( false ),	
	maxSequenceLength( 1 ),
	currentSequencePos( 0 ),	
	isReverseSequense( false ),	
	autoRestartMode( true ),
	isReuseMemoryMode( false )
{
	solver = FINE_DEBUG_NEW CDnnSimpleGradientSolver( mathEngine );
	initializer = FINE_DEBUG_NEW CDnnXavierInitializer( random );
}

CDnn::~CDnn()
{
	for( int i = layers.Size() - 1; i >= 0; i-- ) {
		CPtr<CBaseLayer> layer = layers[i];
		DeleteLayer(*layer);
		layer->setDnn(0);
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
	CheckArchitecture( layerMap.Has( name ), name, "layer is not in this dnn" );
	return layerMap.Get( name );
}

CPtr<const CBaseLayer> CDnn::GetLayer( const char* name ) const
{
	CheckArchitecture( layerMap.Has( name ), name, "layer is not in this dnn" );
	return layerMap.Get( name );
}

void CDnn::AddLayerImpl( CBaseLayer& layer )
{
	CheckArchitecture( !layerMap.Has( layer.GetName() ), layer.GetName(), "layer already in this dnn" );
	CheckArchitecture( layer.GetDnn() == 0, layer.GetName(), "layer already added to other dnn" );

	// Set the flag that indicates the network must be rebuilt (configuration has changed)
	ForceRebuild();

	// Add a layer
	layerMap.Add(layer.GetName(), &layer);
	layers.Add(&layer);

	// Set the layer network
	layer.setDnn( this );
}

void CDnn::ForceRebuild()
{
	isRebuildNeeded = true;
	sinkLayers.SetSize(0);
	sourceLayers.SetSize(0);
}

void CDnn::DeleteLayerImpl( CBaseLayer& layer )
{
	CheckArchitecture( HasLayer( layer.GetName() ),
		layer.GetName(), "deletion of the layer which is not in this dnn" );

	// Set the flag that indicates the network should be rebuilt (configuration has changed)
	ForceRebuild();
	// Unlink all layer connections
	layer.unlink();
	// Delete the layer from the table
	layerMap.Delete(layer.GetName());

	// Set the network for the layer
	layer.setDnn(0);
	// Delete the layer from the array that owns the data
	int oldSize = layers.Size();
	for( int i = 0; i < layers.Size(); i++ ) {
		if( layers[i] == &layer ) {
			layers.DeleteAt(i);
			break;
		}
	}
	NeoAssert(layers.Size() < oldSize);
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
	RequestReshape(true);
}

void CDnn::EnableLearning()
{
	if( isLearningEnabled ) {
		return;
	}
	isLearningEnabled = true;
	RequestReshape(true);
}

void CDnn::RequestReshape(bool forcedReshape)
{
	for( int i = 0; i < layers.Size(); i++ ) {
		layers[i]->isReshapeNeeded = true;
		layers[i]->forcedReshape = layers[i]->forcedReshape || forcedReshape;
	}
}

void CDnn::SetSolver(CDnnSolver* _solver)
{
	if(solver.Ptr() == _solver) {
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

void CDnn::runOnce(int curSequencePos)
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
		NeoAssert(maxSequenceLength == 1);
		if(isBackwardPerformed) {
			// The layer Reshape methods depend on IsBackwardPerformed()
			RequestReshape(true);
		}
		isBackwardPerformed = false;
		if(autoRestartMode) {
			RestartSequence();
		}
		reshape(); // rebuild the network if necessary
		
		isReuseMemoryMode = ( getOutputBlobsSize() > MinReuseMemoryModeNetSize );
		runOnce(0);
	}
#ifdef NEOML_USE_FINEOBJ
	catch( CCheckException* exception ) {
		if( IsLogging() ) {
			*log << "CCheckException in RunOnce\n";
			*log << "\t" << exception->MessageText() << "\n";
		}
		throw;
	}
#else
	catch( CCheckException& exception ) {
		if( IsLogging() ) {
			*log << "CCheckException in RunOnce\n";
			*log << "\t" << exception.what() << "\n";
		}
		throw;
	}
#endif
}

void CDnn::RunAndBackwardOnce()
{
	try {
		NeoAssert(maxSequenceLength == 1);
		if(!isBackwardPerformed) {
			// The layer Reshape methods depend on IsBackwardPerformed()
			RequestReshape(true);
		}
		isBackwardPerformed = true;
		if(autoRestartMode) {
			RestartSequence();
		}
		reshape(); // rebuild the network if necessary
		isReuseMemoryMode = false;
		runOnce(0);
		backwardRunAndLearnOnce(0);
	} catch( CCheckException* exception ) {
		if( IsLogging() ) {
			*log << "CCheckException in RunAndLearnOnce\n";
#ifdef NEOML_USE_FINEOBJ
			*log << "\t" << exception->MessageText() << "\n";
#else
			*log << "\t" << exception->what() << "\n";
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

void CDnn::backwardRunAndLearnOnce(int curSequencePos)
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
		if(layer->GetInputCount() == 0) {
			sourceLayers.Add(layers[i]);
		}
		if(layer->GetOutputCount() == 0) {
			sinkLayers.Add(layers[i]);
		}
	}
	RequestReshape(true);
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
	for( int i = 0; i < layers.Size(); ++i ) {
		layers[i]->FilterLayerParams( threshold );
	}
}

void CDnn::FilterLayersParams( const CArray<const char*>& layers, float threshold )
{
	for( int i = 0; i < layers.Size(); ++i ) {
		GetLayer( layers[i] )->FilterLayerParams( threshold );
	}
}

static const int DnnVersion = 2000;

void CDnn::Serialize( CArchive& archive )
{
	if( archive.IsStoring() ) {
		archive << DnnVersion;
		archive << logFrequency;

		archive << layers.Size();
		for( int i = 0; i < layers.Size(); ++i) {
			archive << getLayerName( layers[i] );
			if( layers[i] != 0 ) {
				layers[i]->Serialize( archive );
			}
		}
		archive << isLearningEnabled;
	} else if( archive.IsLoading() ) {
		// Clean up the network
		while( layers.Size() > 0 ) {
			DeleteLayer( *layers[0] );
		}
		runNumber = 0;
		isRebuildNeeded = false;
		isBackwardPerformed = false;

		// Calculate the data
		int version;
		archive >> version;
		if( version < 0 ) {
			version = -version;
		}

		if( version < CDnn::ArchiveMinSupportedVersion || version > DnnVersion ) {
			check( false, ERR_BAD_ARCHIVE_VERSION, archive.Name() );
		}

		archive >> logFrequency;

		int layerCount;
		archive >> layerCount;
		for( int i = 0; i < layerCount; ++i) {
			CString className;
			archive >> className;
			CPtr<CBaseLayer> layer = createLayer( GetMathEngine(), className );
			check( layer != 0, ERR_BAD_ARCHIVE, archive.Name() );
			layer->Serialize( archive );
			AddLayer( *layer );
		}
		archive >> isLearningEnabled;
		// In order to avoid the CDnnSolver::Reset for the next solver
		rebuild();
	} else {
		NeoAssert( false );
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

} // namespace NeoML
