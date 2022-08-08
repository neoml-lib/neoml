/* Copyright © 2017-2021 ABBYY Production LLC

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

#include "PyDnn.h"
#include "PyMathEngine.h"
#include "PySolver.h"
#include "PyInitializer.h"
#include "PyDnnBlob.h"
#include "PyLayer.h"
#include "PyMemoryFile.h"

struct CPyClass {
	CString ModuleName;
	CString ClassName;
};

static CMap<CString, CPyClass, CDefaultHash<CString>, RuntimeHeap>& getRegisteredPyLayers()
{
	static CMap<CString, CPyClass, CDefaultHash<CString>, RuntimeHeap> registeredPyLayers;
	return registeredPyLayers;
}

class CPyLayerClassRegistrar {
public:
	CPyLayerClassRegistrar( const CString& pyModuleName, const CString& pyClassName, const CString& internalClassName )
	{
		CPyClass c;
		c.ModuleName = pyModuleName;
		c.ClassName = pyClassName;
		getRegisteredPyLayers().Add( internalClassName, c );
	}
};

#define REGISTER_NEOML_PYLAYER( pyClassName, internalClassName ) static CPyLayerClassRegistrar __merge__1( _RegisterLayer, __LINE__ )( pyClassName, pyClassName, internalClassName );
#define REGISTER_NEOML_PYLAYER_EX( pyModuleName, pyClassName, internalClassName ) static CPyLayerClassRegistrar __merge__1( _RegisterLayer, __LINE__ )( pyModuleName, pyClassName, internalClassName );

// Register all layer names
namespace {

REGISTER_NEOML_PYLAYER( "Source", "FmlCnnSourceLayer" )
REGISTER_NEOML_PYLAYER( "Sink", "FmlCnnSinkLayer" )
REGISTER_NEOML_PYLAYER_EX( "Concat", "ConcatChannels", "FmlCnnConcatChannelsLayer" )
REGISTER_NEOML_PYLAYER_EX( "Concat", "ConcatDepth", "FmlCnnConcatDepthLayer" )
REGISTER_NEOML_PYLAYER_EX( "Concat", "ConcatWidth", "FmlCnnConcatWidthLayer" )
REGISTER_NEOML_PYLAYER_EX( "Concat", "ConcatHeight", "FmlCnnConcatHeightLayer" )
REGISTER_NEOML_PYLAYER_EX( "Concat", "ConcatBatchWidth", "FmlCnnConcatBatchWidthLayer" )
REGISTER_NEOML_PYLAYER_EX( "Concat", "ConcatBatchLength", "FmlCnnConcatBatchLengthLayer" )
REGISTER_NEOML_PYLAYER_EX( "Concat", "ConcatListSize", "FmlCnnConcatListSizeLayer" )
REGISTER_NEOML_PYLAYER_EX( "Concat", "ConcatObject", "FmlCnnConcatObjectLayer" )
REGISTER_NEOML_PYLAYER_EX( "Split", "SplitChannels", "FmlCnnSplitChannelsLayer" )
REGISTER_NEOML_PYLAYER_EX( "Split", "SplitDepth", "FmlCnnSplitDepthLayer" )
REGISTER_NEOML_PYLAYER_EX( "Split", "SplitWidth", "FmlCnnSplitWidthLayer" )
REGISTER_NEOML_PYLAYER_EX( "Split", "SplitHeight", "FmlCnnSplitHeightLayer" )
REGISTER_NEOML_PYLAYER_EX( "Split", "SplitListSize", "NeoMLDnnSplitListSizeLayer" )
REGISTER_NEOML_PYLAYER_EX( "Split", "SplitBatchWidth", "FmlCnnSplitBatchWidthLayer" )
REGISTER_NEOML_PYLAYER_EX( "Split", "SplitBatchLength", "NeoMLDnnSplitBatchLengthLayer" )
REGISTER_NEOML_PYLAYER_EX( "Eltwise", "EltwiseSum", "FmlCnnEltwiseSumLayer" )
REGISTER_NEOML_PYLAYER_EX( "Eltwise", "EltwiseSub", "NeoMLDnnEltwiseSubLayer" )
REGISTER_NEOML_PYLAYER_EX( "Eltwise", "EltwiseMul", "FmlCnnEltwiseMulLayer" )
REGISTER_NEOML_PYLAYER_EX( "Eltwise", "EltwiseDiv", "NeoMLDnnEltwiseDivLayer" )
REGISTER_NEOML_PYLAYER_EX( "Eltwise", "EltwiseNegMul", "FmlCnnEltwiseNegMulLayer" )
REGISTER_NEOML_PYLAYER_EX( "Eltwise", "EltwiseMax", "FmlCnnEltwiseMaxLayer" )
REGISTER_NEOML_PYLAYER_EX( "Activation", "LinearLayer", "FmlCnnLinearLayer")
REGISTER_NEOML_PYLAYER_EX( "Activation", "ELU", "FmlCnnELULayer" )
REGISTER_NEOML_PYLAYER_EX( "Activation", "ReLU", "FmlCnnReLULayer" )
REGISTER_NEOML_PYLAYER_EX( "Activation", "GELU", "NeoMLDnnGELULayer")
REGISTER_NEOML_PYLAYER_EX( "Activation", "LeakyReLU", "FmlCnnLeakyReLULayer" )
REGISTER_NEOML_PYLAYER_EX( "Activation", "Abs", "FmlCnnAbsLayer" )
REGISTER_NEOML_PYLAYER_EX( "Activation", "Sigmoid", "FmlCnnSigmoidLayer" )
REGISTER_NEOML_PYLAYER_EX( "Activation", "Tanh", "FmlCnnTanhLayer" )
REGISTER_NEOML_PYLAYER_EX( "Activation", "HardTanh", "FmlCnnHardTanhLayer" )
REGISTER_NEOML_PYLAYER_EX( "Activation", "HardSigmoid", "FmlCnnSigmoidTanhLayer" )
REGISTER_NEOML_PYLAYER_EX( "Activation", "HSwish", "FmlCnnHSwishLayer" )
REGISTER_NEOML_PYLAYER_EX( "Activation", "Power", "FmlCnnPowerLayer" )
REGISTER_NEOML_PYLAYER_EX( "Activation", "Exp", "NeoMLDnnExpLayer" )
REGISTER_NEOML_PYLAYER_EX( "Activation", "Log", "NeoMLDnnLogLayer" )
REGISTER_NEOML_PYLAYER_EX( "Activation", "Erf", "NeoMLDnnErfLayer" )
REGISTER_NEOML_PYLAYER( "RleConv", "FmlCnnRleConvLayer" )
REGISTER_NEOML_PYLAYER_EX( "Pooling", "MaxPooling", "FmlCnnMaxPoolingLayer" )
REGISTER_NEOML_PYLAYER_EX( "Pooling", "MeanPooling", "FmlCnnMeanPoolingLayer" )
REGISTER_NEOML_PYLAYER_EX( "Pooling", "GlobalMeanPooling", "FmlCnnGlobalMainPoolingLayer")
REGISTER_NEOML_PYLAYER_EX( "Pooling", "GlobalMaxPooling", "FmlCnnGlobalMaxPoolingLayer")
REGISTER_NEOML_PYLAYER_EX( "Pooling", "GlobalSumPooling", "NeoMLDnnGlobalSumPoolingLayer")
REGISTER_NEOML_PYLAYER_EX( "Pooling", "MaxOverTimePooling", "FmlCnnMaxOverTimePoolingLayer")
REGISTER_NEOML_PYLAYER_EX( "Pooling", "MaxPooling3D", "FmlCnn3dMaxPoolingLayer")
REGISTER_NEOML_PYLAYER_EX( "Pooling", "MeanPooling3D", "FmlCnn3dMeanPoolingLayer")
REGISTER_NEOML_PYLAYER( "FullyConnected", "FmlCnnFullyConnectedLayer" )
REGISTER_NEOML_PYLAYER( "FullyConnectedSource", "FmlCnnFullyConnectedSourceLayer" )
REGISTER_NEOML_PYLAYER_EX( "Logical", "Equal", "NeoMLDnnEqualLayer" )
REGISTER_NEOML_PYLAYER_EX( "Logical", "Not", "NeoMLDnnNotLayer" )
REGISTER_NEOML_PYLAYER_EX( "Logical", "Less", "NeoMLDnnLessLayer" )
REGISTER_NEOML_PYLAYER_EX( "Logical", "Where", "NeoMLDnnWhereLayer" )
REGISTER_NEOML_PYLAYER_EX( "Loss", "CrossEntropyLoss", "FmlCnnCrossEntropyLossLayer" )
REGISTER_NEOML_PYLAYER_EX( "Loss", "BinaryCrossEntropyLoss", "FmlCnnBinaryCrossEntropyLossLayer" )
REGISTER_NEOML_PYLAYER_EX( "Loss", "EuclideanLoss", "FmlCnnEuclideanLossLayer" )
REGISTER_NEOML_PYLAYER_EX( "Loss", "L1Loss", "NeoMLDnnL1LossLayer" )
REGISTER_NEOML_PYLAYER_EX( "Loss", "HingeLoss", "FmlCnnHingeLossLayer" )
REGISTER_NEOML_PYLAYER_EX( "Loss", "SquaredHingeLoss", "FmlCnnSquaredHingeLossLayer" )
REGISTER_NEOML_PYLAYER_EX( "Loss", "CustomLoss", "NeoMLCustomLossLayer" )
REGISTER_NEOML_PYLAYER( "ProblemSource", "FmlCnnProblemSourceLayer" )
REGISTER_NEOML_PYLAYER( "BatchNormalization", "FmlCnnBatchNormalizationLayer" )
REGISTER_NEOML_PYLAYER( "ObjectNormalization", "NeoMLDnnObjectNormalizationLayer" )
REGISTER_NEOML_PYLAYER( "Dropout", "FmlCnnDropoutLayer" )
REGISTER_NEOML_PYLAYER( "MultichannelLookup", "FmlCnnMultychannelLookupLayer" )
REGISTER_NEOML_PYLAYER( "Composite", "FmlCnnCompositeLayer" )
REGISTER_NEOML_PYLAYER( "Recurrent", "FmlCnnRecurrentLayer" )
REGISTER_NEOML_PYLAYER( "SubSequence", "FmlCnnSubSequenceLayer" )
REGISTER_NEOML_PYLAYER( "BackLink", "FmlCnnBackLink" )
REGISTER_NEOML_PYLAYER( "CaptureSink", "FmlCnnCaptureSink" )
REGISTER_NEOML_PYLAYER_EX( "Binarization", "EnumBinarization", "FmlCnnEnumBinarizationLayer" )
REGISTER_NEOML_PYLAYER_EX( "Binarization", "BitSetVectorization", "FmlCnnBitSetVectorizationLayerClassName" )
REGISTER_NEOML_PYLAYER( "Softmax", "FmlCnnSoftmaxLayer" )
REGISTER_NEOML_PYLAYER( "Lstm", "FmlCnnLstmLayer" )
REGISTER_NEOML_PYLAYER( "Gru", "FmlCnnGruLayer" )
REGISTER_NEOML_PYLAYER_EX( "Conv", "Conv", "FmlCnnConvLayer")
REGISTER_NEOML_PYLAYER_EX( "Conv", "TimeConv", "FmlCnnTimeConvLayer" )
REGISTER_NEOML_PYLAYER_EX( "Conv", "Conv3D", "FmlCnn3dConvLayer" )
REGISTER_NEOML_PYLAYER_EX( "Conv", "TransposedConv", "FmlCnnTransposedConvLayer")
REGISTER_NEOML_PYLAYER_EX( "Conv", "TransposedConv3D", "FmlCnn3dTransposedConvLayer")
REGISTER_NEOML_PYLAYER_EX( "Conv", "ChannelwiseConv", "FmlCnnChannelwiseConvLayer")
REGISTER_NEOML_PYLAYER_EX( "Crf", "Crf", "FmlCnnCrfLayer" )
REGISTER_NEOML_PYLAYER_EX( "Crf", "CrfLoss", "FmlCnnCrfLossLayer")
REGISTER_NEOML_PYLAYER_EX( "Crf", "BestSequence", "FmlCnnBestSequenceLayer")
REGISTER_NEOML_PYLAYER( "CrfCalculation", "FmlCnnCrfCalculationLayer" )
REGISTER_NEOML_PYLAYER( "CrfInternalLoss", "FmlCnnCrfInternalLossLayer" )
REGISTER_NEOML_PYLAYER( "SequenceSum", "FmlCnnSequenceSumLayer" )
REGISTER_NEOML_PYLAYER_EX( "Ctc", "CtcLoss", "FmlCnnCtcLossLayer" )
REGISTER_NEOML_PYLAYER_EX( "Ctc", "CtcDecoding", "FmlCnnCtcDecodingLayer" )
REGISTER_NEOML_PYLAYER_EX( "Loss", "MultiHingeLoss", "FmlCnnMultyHingeLossLayer" )
REGISTER_NEOML_PYLAYER_EX( "Loss", "MultiSquaredHingeLoss", "FmlCnnMultySquaredHingeLossLayer" )
REGISTER_NEOML_PYLAYER( "Upsampling2D", "FmlCnnUpsampling2DLayer" )
REGISTER_NEOML_PYLAYER( "AccumulativeLookup", "FmlCnnAccumulativeLookupLayer" )
REGISTER_NEOML_PYLAYER_EX( "Accuracy", "Accuracy", "FmlCnnAccuracyLayer" )
REGISTER_NEOML_PYLAYER_EX( "Accuracy", "ConfusionMatrix", "FmlCnnConfusionMatrixLayer")
REGISTER_NEOML_PYLAYER("TiedEmbeddings", "TiedEmbeddingsLayer")
REGISTER_NEOML_PYLAYER( "PrecisionRecall", "FmlCnnPrecisionRecallLayer" )
REGISTER_NEOML_PYLAYER_EX( "Loss", "CenterLoss", "FmlCnnCenterLossLayer" )
REGISTER_NEOML_PYLAYER_EX( "Loss", "FocalLoss", "FmlCnnFocalLossLayer" )
REGISTER_NEOML_PYLAYER_EX( "Loss", "BinaryFocalLoss", "FmlCnnBinaryFocalLossLayer" )
REGISTER_NEOML_PYLAYER_EX( "ImageConversion", "ImageResize", "FmlCnnImageResizeLayer")
REGISTER_NEOML_PYLAYER_EX( "ImageConversion", "ImageToPixel", "FmlCnnImageToPixelLayerClass" )
REGISTER_NEOML_PYLAYER_EX( "ImageConversion", "PixelToImage", "FmlCnnPixelToImageLayerClass" )
REGISTER_NEOML_PYLAYER( "Transpose", "FmlCnnTransposeLayer" )
REGISTER_NEOML_PYLAYER( "Transform", "FmlCnnTransformWithoutTransposeLayer" )
REGISTER_NEOML_PYLAYER( "Argmax", "FmlCnnArgmaxLayer" )
REGISTER_NEOML_PYLAYER( "AttentionDecoder", "FmlCnnAttentionDecoderLayer" )
REGISTER_NEOML_PYLAYER( "AttentionRecurrent", "FmlCnnAttentionRecurrentLayer" )
REGISTER_NEOML_PYLAYER( "Attention", "FmlCnnAttentionLayer" )
REGISTER_NEOML_PYLAYER( "RepeatSequence", "FmlCnnRepeatSequenceLayer" )
REGISTER_NEOML_PYLAYER( "DotProduct", "FmlCnnDotProductLayer" )
REGISTER_NEOML_PYLAYER( "Reorg", "FmlCnnReorgLayerClass" )
REGISTER_NEOML_PYLAYER( "CompositeSource", "FmlCnnCompositeSourceLayer" )
REGISTER_NEOML_PYLAYER( "CompositeSink", "FmlCompositeCnnSinkLayer" )
REGISTER_NEOML_PYLAYER( "AttentionWeightedSum", "FmlCnnAttentionWeightedSumLayer" )
REGISTER_NEOML_PYLAYER( "AttentionDotProduct", "FmlCnnAttentionDotProductLayer" )
REGISTER_NEOML_PYLAYER( "AttentionSum", "FmlCnnAttentionSumLayer" )
REGISTER_NEOML_PYLAYER( "AddToObject", "NeoMLDnnAddToObjectLayer" )
REGISTER_NEOML_PYLAYER( "MatrixMultiplication", "NeoMLDnnMatrixMultiplicationLayer" )
REGISTER_NEOML_PYLAYER( "MultiheadAttention", "NeoMLDnnMultiheadAttentionLayer" )
REGISTER_NEOML_PYLAYER( "PositionalEmbedding", "NeoMLDnnPositionalEmbeddingLayer" )
REGISTER_NEOML_PYLAYER( "ProjectionPooling", "FmlCnnProjectionPoolingLayerClass" )
REGISTER_NEOML_PYLAYER( "Irnn", "NeoMLDnnIrnnLayer" )
REGISTER_NEOML_PYLAYER( "IndRnn", "NeoMLDnnIndRnnLayer" )
REGISTER_NEOML_PYLAYER( "Qrnn", "NeoMLDnnQrnnLayer" )
REGISTER_NEOML_PYLAYER( "Lrn", "NeoMLDnnLrnLayer" )
REGISTER_NEOML_PYLAYER( "Cast", "NeoMLDnnCastLayer" )
REGISTER_NEOML_PYLAYER( "Data", "NeoMLDnnDataLayer" )
REGISTER_NEOML_PYLAYER( "TransformerEncoder", "NeoMLDnnTransformerEncoderLayer" )
REGISTER_NEOML_PYLAYER( "BertConv", "NeoMLDnnBertConvLayer" )
REGISTER_NEOML_PYLAYER( "Broadcast", "NeoMLDnnBroadcastLayer" )
REGISTER_NEOML_PYLAYER( "CumSum", "NeoMLDnnCumSumLayer" )
REGISTER_NEOML_PYLAYER_EX( "ScatterGather", "ScatterND", "NeoMLDnnScatterNDLayer" )

}

py::object createLayer( CBaseLayer& layer, CPyMathEngineOwner& mathEngineOwner )
{
	CPyLayer pyLayer( layer, mathEngineOwner );

	CMap<CString, CPyClass, CDefaultHash<CString>, RuntimeHeap>& layers = getRegisteredPyLayers();
	CString layerClass( GetLayerClass( layer ) );

	if( !layers.Has( layerClass ) ) {
		NeoML::GetMathEngineExceptionHandler()->OnAssert( "Class '" + layerClass + "' isn't known to Python module",
			__UNICODEFILE__, __LINE__, 0 );
	}
	CPyClass c = layers.Get( layerClass );

	CString wrapperModuleName = "neoml.PythonWrapper";
	py::object wrapperModule = py::module::import( wrapperModuleName );
	py::object wrapperConstructor = wrapperModule.attr( c.ClassName );
	py::object wrapper = wrapperConstructor( pyLayer );

	return wrapper.attr("create_python_object")();
}

//------------------------------------------------------------------------------------------------------------

CPyDnn::CPyDnn( CPyRandomOwner& _randomOwner, CPyMathEngineOwner& _mathEngineOwner ) :
	randomOwner( &_randomOwner ),
	mathEngineOwner( &_mathEngineOwner ),
	initializer( _randomOwner, new CDnnXavierInitializer( _randomOwner.Random() ) ),
	dnn( new CDnn( _randomOwner.Random(), _mathEngineOwner.MathEngine() ) )
{
}

void CPyDnn::Load(const std::string& path)
{
	py::gil_scoped_release release;
	CArchiveFile file( path.c_str(), CArchive::load );
	CArchive archive( &file, CArchive::load );
	dnn->Serialize( archive );
}

void CPyDnn::Store(const std::string& path)
{
	py::gil_scoped_release release;
	CArchiveFile file( path.c_str(), CArchive::store );
	CArchive archive( &file, CArchive::store );
	dnn->Serialize( archive );
}

void CPyDnn::LoadCheckpoint(const std::string& path)
{
	py::gil_scoped_release release;
	CArchiveFile file( path.c_str(), CArchive::load );
	CArchive archive( &file, CArchive::load );
	dnn->SerializeCheckpoint( archive );
}

void CPyDnn::StoreCheckpoint(const std::string& path)
{
	py::gil_scoped_release release;
	CArchiveFile file( path.c_str(), CArchive::store );
	CArchive archive( &file, CArchive::store );
	dnn->SerializeCheckpoint( archive );
}

py::object CPyDnn::GetMathEngine() const
{
	CPyMathEngine mathEngine( *mathEngineOwner );
	py::object m = py::module::import("neoml.MathEngine");
	if( mathEngineOwner->MathEngine().GetType() == MET_Cpu ) {
		py::object constructor = m.attr( "CpuMathEngine" );
		return constructor( mathEngine );
	}
	py::object constructor = m.attr( "GpuMathEngine" );
	return constructor( mathEngine );
}

void CPyDnn::SetSolver( const CPySolver& solver )
{
	dnn->SetSolver( &solver.BaseSolver() );
}

py::object CPyDnn::GetSolver() const
{
	CPySolver pySolver( *dnn->GetSolver(), *mathEngineOwner );
	CString solverName( pySolver.GetClassName() );

	CString wrapperModuleName = "neoml.PythonWrapper";
	CString moduleName = "neoml.Dnn";

	py::object wrapperModule = py::module::import( wrapperModuleName );
	py::object module = py::module::import( moduleName );

	py::object wrapperConstructor = wrapperModule.attr( solverName );
	py::object wrapper = wrapperConstructor( pySolver );

	py::object constructor = module.attr( solverName );
	return constructor( wrapper );
}

void CPyDnn::SetInitializer( const CPyInitializer& _initializer )
{
	initializer = _initializer;
	dnn->SetInitializer( initializer.Initializer<CDnnInitializer>() );
}

py::object CPyDnn::GetInitializer() const
{
	CString initializerName( initializer.GetClassName() );

	CString wrapperModuleName = "neoml.PythonWrapper";
	CString moduleName = "neoml.Dnn";

	py::object wrapperModule = py::module::import( wrapperModuleName );
	py::object module = py::module::import( moduleName );

	py::object wrapperConstructor = wrapperModule.attr( initializerName );
	py::object wrapper = wrapperConstructor( initializer );

	py::object constructor = module.attr( initializerName );
	return constructor( wrapper );
}

py::dict CPyDnn::GetInputs() const
{
	CArray<const char*> layerNames;
	dnn->GetLayerList( layerNames );

	auto result = py::dict();
	for( int i = 0; i < layerNames.Size(); i++ ) {
		CPtr<CBaseLayer> layer = dnn->GetLayer( layerNames[i] );
		if( dynamic_cast<CSourceLayer*>( layer.Ptr() ) != 0 ) {
			result[layerNames[i]] = createLayer( *layer, *mathEngineOwner );
		}
	}
	return result;
}

py::dict CPyDnn::GetOutputs() const
{
	CArray<const char*> layerNames;
	dnn->GetLayerList( layerNames );

	auto result = py::dict();
	for( int i = 0; i < layerNames.Size(); i++ ) {
		CPtr<CBaseLayer> layer = dnn->GetLayer( layerNames[i] );
		if( dynamic_cast<CSinkLayer*>( layer.Ptr() ) != 0 ) {
			result[layerNames[i]] = createLayer( *layer, *mathEngineOwner );
		}
	}
	return result;
}

bool CPyDnn::HasLayer( const char* layerName ) const
{
	return dnn->HasLayer( layerName );
}

void CPyDnn::AddLayer( CPyLayer& pyLayer )
{
	dnn->AddLayer( pyLayer.BaseLayer() );
}

void CPyDnn::DeleteLayer( const char* layerName )
{
	dnn->DeleteLayer( layerName );
}

py::dict CPyDnn::GetLayers() const
{
	CArray<const char*> layerNames;
	dnn->GetLayerList( layerNames );

	auto result = py::dict();
	for( int i = 0; i < layerNames.Size(); i++ ) {
		CPtr<CBaseLayer> layer = dnn->GetLayer( layerNames[i] );
		result[layerNames[i]] = createLayer( *layer, *mathEngineOwner );
	}
	return result;
}

py::dict CPyDnn::Run( py::list inputs )
{
	CArray<const char*> layerNames;
	dnn->GetLayerList( layerNames );

	int index = 0;
	for( int layerIndex = 0; layerIndex < layerNames.Size(); ++layerIndex ) {
		CPtr<CSourceLayer> layer = dynamic_cast<CSourceLayer*>( dnn->GetLayer( layerNames[layerIndex] ).Ptr() );
		if( layer != 0 ) {
			CPyBlob input = inputs[index].cast<CPyBlob>();
			layer->SetBlob( input.Blob() );
			index++;
		}
	}
	{
		py::gil_scoped_release release;
		dnn->RunOnce();
	}

	auto result = py::dict();
	for( int layerIndex = 0; layerIndex < layerNames.Size(); ++layerIndex ) {
		CPtr<CSinkLayer> layer = dynamic_cast<CSinkLayer*>( dnn->GetLayer( layerNames[layerIndex] ).Ptr() );
		if( layer != 0 ) {
			result[layerNames[layerIndex]] = CPyBlob( *mathEngineOwner, layer->GetBlob() );
		}
	}

	return result;
}

py::dict CPyDnn::RunAndBackward( py::list inputs )
{
	CArray<const char*> layerNames;
	dnn->GetLayerList( layerNames );
			
	int index = 0;
	for( int layerIndex = 0; layerIndex < layerNames.Size(); ++layerIndex ) {
		CPtr<CSourceLayer> layer = dynamic_cast<CSourceLayer*>( dnn->GetLayer( layerNames[layerIndex] ).Ptr() );
		if( layer != 0 ) {
			CPyBlob input = inputs[index].cast<CPyBlob>();
			layer->SetBlob( input.Blob() );
			index++;
		}
	}

	{
		py::gil_scoped_release release;
		dnn->RunAndBackwardOnce();
	}

	auto result = py::dict();
	for( int layerIndex = 0; layerIndex < layerNames.Size(); ++layerIndex ) {
		CPtr<CSinkLayer> layer = dynamic_cast<CSinkLayer*>( dnn->GetLayer( layerNames[layerIndex] ).Ptr() );
		if( layer != 0 ) {
			result[layerNames[layerIndex]] = CPyBlob( *mathEngineOwner, layer->GetBlob() );
		}
	}

	return result;
}

py::dict CPyDnn::Learn( py::list inputs )
{
	CArray<const char*> layerNames;
	dnn->GetLayerList( layerNames );
			
	int index = 0;
	for( int layerIndex = 0; layerIndex < layerNames.Size(); ++layerIndex ) {
		CPtr<CSourceLayer> layer = dynamic_cast<CSourceLayer*>( dnn->GetLayer( layerNames[layerIndex] ).Ptr() );
		if( layer != 0 ) {
			CPyBlob input = inputs[index].cast<CPyBlob>();
			layer->SetBlob( input.Blob() );
			index++;
		}
	}

	{
		py::gil_scoped_release release;
		dnn->RunAndLearnOnce();
	}

	auto result = py::dict();
	for( int layerIndex = 0; layerIndex < layerNames.Size(); ++layerIndex ) {
		CPtr<CSinkLayer> layer = dynamic_cast<CSinkLayer*>( dnn->GetLayer( layerNames[layerIndex] ).Ptr() );
		if( layer != 0 ) {
			result[layerNames[layerIndex]] = CPyBlob( *mathEngineOwner, layer->GetBlob() );
		}
	}

	return result;
}

//------------------------------------------------------------------------------------------------------------

void InitializeDnn(py::module& m)
{
	py::class_<CPyDnn>(m, "Dnn")
		.def( py::init([]( const CPyRandom& random, const CPyMathEngine& mathEngine )
		{
			return new CPyDnn( random.RandomOwner(), mathEngine.MathEngineOwner() );
		}) )
		.def( "get_math_engine", &CPyDnn::GetMathEngine, py::return_value_policy::reference )
		.def( "set_solver", &CPyDnn::SetSolver, py::return_value_policy::reference )
		.def( "get_solver", &CPyDnn::GetSolver, py::return_value_policy::reference )
		.def( "set_initializer", &CPyDnn::SetInitializer, py::return_value_policy::reference )
		.def( "get_initializer", &CPyDnn::GetInitializer, py::return_value_policy::reference )
		.def( "get_inputs", &CPyDnn::GetInputs, py::return_value_policy::reference )
		.def( "get_outputs", &CPyDnn::GetOutputs, py::return_value_policy::reference )
		.def( "get_layers", &CPyDnn::GetLayers, py::return_value_policy::reference )
		.def( "has_layer", &CPyDnn::HasLayer, py::return_value_policy::reference )
		.def( "_add_layer", &CPyDnn::AddLayer, py::return_value_policy::reference )
		.def( "_delete_layer", &CPyDnn::DeleteLayer, py::return_value_policy::reference )
		.def( "_run", &CPyDnn::Run, py::return_value_policy::reference )
		.def( "_run_and_backward", &CPyDnn::RunAndBackward, py::return_value_policy::reference )
		.def( "_learn", &CPyDnn::Learn, py::return_value_policy::reference )
		.def( "_load", &CPyDnn::Load, py::return_value_policy::reference )
		.def( "_load_checkpoint", &CPyDnn::LoadCheckpoint, py::return_value_policy::reference )
		.def( "_store", &CPyDnn::Store, py::return_value_policy::reference )
		.def( "_store_checkpoint", &CPyDnn::StoreCheckpoint, py::return_value_policy::reference )
		.def(py::pickle(
			[](const CPyDnn& pyDnn) {
				CPyMemoryFile file;
				CArchive archive( &file, CArchive::store );
				pyDnn.Dnn().SerializeCheckpoint( archive );
				archive.Close();
				file.Close();
				return py::make_tuple( file.GetBuffer() );
			},
			[](py::tuple t) {
				if( t.size() != 1 ) {
					throw std::runtime_error("Invalid state!");
				}

				auto t0_array = t[0].cast<py::array>();
				CPyMemoryFile file( t0_array );
				CArchive archive( &file, CArchive::load );
				CPtr<CPyMathEngineOwner> mathEngineOwner( new CPyMathEngineOwner() );
				CPtr<CPyRandomOwner> randomOwner( new CPyRandomOwner() );
				CPyDnn pyDnn( *randomOwner, *mathEngineOwner );
				pyDnn.Dnn().SerializeCheckpoint( archive );
				return pyDnn;
			}
		))

	;
}
