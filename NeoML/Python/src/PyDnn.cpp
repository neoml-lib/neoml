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
REGISTER_NEOML_PYLAYER( "ConcatChannels", "FmlCnnConcatChannelsLayer" )
REGISTER_NEOML_PYLAYER( "ConcatDepth", "FmlCnnConcatDepthLayer" )
REGISTER_NEOML_PYLAYER( "ConcatWidth", "FmlCnnConcatWidthLayer" )
REGISTER_NEOML_PYLAYER( "ConcatHeight", "FmlCnnConcatHeightLayer" )
REGISTER_NEOML_PYLAYER( "ConcatBatchWidth", "FmlCnnConcatBatchWidthLayer" )
REGISTER_NEOML_PYLAYER( "ConcatObject", "FmlCnnConcatObjectLayer" )
REGISTER_NEOML_PYLAYER( "SplitChannels", "FmlCnnSplitChannelsLayer" )
REGISTER_NEOML_PYLAYER( "SplitDepth", "FmlCnnSplitDepthLayer" )
REGISTER_NEOML_PYLAYER( "SplitWidth", "FmlCnnSplitWidthLayer" )
REGISTER_NEOML_PYLAYER( "SplitHeight", "FmlCnnSplitHeightLayer" )
REGISTER_NEOML_PYLAYER( "SplitBatchWidth", "FmlCnnSplitBatchWidthLayer" )
REGISTER_NEOML_PYLAYER_EX( "Eltwise", "EltwiseSum", "FmlCnnEltwiseSumLayer" )
REGISTER_NEOML_PYLAYER_EX( "Eltwise", "EltwiseMul", "FmlCnnEltwiseMulLayer" )
REGISTER_NEOML_PYLAYER_EX( "Eltwise", "EltwiseNegMul", "FmlCnnEltwiseNegMulLayer" )
REGISTER_NEOML_PYLAYER_EX( "Eltwise", "EltwiseMax", "FmlCnnEltwiseMaxLayer" )
REGISTER_NEOML_PYLAYER( "ELU", "FmlCnnELULayer" )
REGISTER_NEOML_PYLAYER( "ReLU", "FmlCnnReLULayer" )
REGISTER_NEOML_PYLAYER( "LeakyReLU", "FmlCnnLeakyReLULayer" )
REGISTER_NEOML_PYLAYER( "Abs", "FmlCnnAbsLayer" )
REGISTER_NEOML_PYLAYER( "Sigmoid", "FmlCnnSigmoidLayer" )
REGISTER_NEOML_PYLAYER( "Tanh", "FmlCnnTanhLayer" )
REGISTER_NEOML_PYLAYER( "HardTanh", "FmlCnnHardTanhLayer" )
REGISTER_NEOML_PYLAYER( "HardSigmoid", "FmlCnnSigmoidTanhLayer" )
REGISTER_NEOML_PYLAYER( "HSwish", "FmlCnnHSwishLayer" )
REGISTER_NEOML_PYLAYER( "Power", "FmlCnnPowerLayer" )
REGISTER_NEOML_PYLAYER( "Conv", "FmlCnnConvLayer" )
REGISTER_NEOML_PYLAYER( "RleConv", "FmlCnnRleConvLayer" )
REGISTER_NEOML_PYLAYER_EX( "Pooling", "MaxPooling", "FmlCnnMaxPoolingLayer" )
REGISTER_NEOML_PYLAYER_EX( "Pooling", "MeanPooling", "FmlCnnMeanPoolingLayer" )
REGISTER_NEOML_PYLAYER( "FullyConnected", "FmlCnnFullyConnectedLayer" )
REGISTER_NEOML_PYLAYER( "FullyConnectedSource", "FmlCnnFullyConnectedSourceLayer" )
REGISTER_NEOML_PYLAYER_EX( "Loss", "CrossEntropyLoss", "FmlCnnCrossEntropyLossLayer" )
REGISTER_NEOML_PYLAYER_EX( "Loss", "BinaryCrossEntropyLoss", "FmlCnnBinaryCrossEntropyLossLayer" )
REGISTER_NEOML_PYLAYER_EX( "Loss", "EuclideanLoss", "FmlCnnEuclideanLossLayer" )
REGISTER_NEOML_PYLAYER_EX( "Loss", "HingeLoss", "FmlCnnHingeLossLayer" )
REGISTER_NEOML_PYLAYER_EX( "Loss", "SquaredHingeLoss", "FmlCnnSquaredHingeLossLayer" )
REGISTER_NEOML_PYLAYER( "ProblemSource", "FmlCnnProblemSourceLayer" )
REGISTER_NEOML_PYLAYER( "BatchNormalization", "FmlCnnBatchNormalizationLayer" )
REGISTER_NEOML_PYLAYER( "ObjectNormalization", "NeoMLDnnObjectNormalizationLayer" )
REGISTER_NEOML_PYLAYER( "Linear", "FmlCnnLinearLayer" )
REGISTER_NEOML_PYLAYER( "Dropout", "FmlCnnDropoutLayer" )
REGISTER_NEOML_PYLAYER( "ImageResize", "FmlCnnImageResizeLayer" )
REGISTER_NEOML_PYLAYER( "MultichannelLookup", "FmlCnnMultychannelLookupLayer" )
REGISTER_NEOML_PYLAYER( "Composite", "FmlCnnCompositeLayer" )
REGISTER_NEOML_PYLAYER( "Recurrent", "FmlCnnRecurrentLayer" )
REGISTER_NEOML_PYLAYER( "SubSequence", "FmlCnnSubSequenceLayer" )
REGISTER_NEOML_PYLAYER( "BackLink", "FmlCnnBackLink" )
REGISTER_NEOML_PYLAYER( "CaptureSink", "FmlCnnCaptureSink" )
REGISTER_NEOML_PYLAYER( "EnumBinarization", "FmlCnnEnumBinarizationLayer" )
REGISTER_NEOML_PYLAYER( "BitSetVectorization", "FmlCnnBitSetVectorizationLayerClassName" )
REGISTER_NEOML_PYLAYER( "Softmax", "FmlCnnSoftmaxLayer" )
REGISTER_NEOML_PYLAYER( "GlobalMeanPooling", "FmlCnnGlobalAveragePoolingLayer" )
REGISTER_NEOML_PYLAYER( "GlobalMaxPooling", "FmlCnnGlobalMaxPoolingLayer" )
REGISTER_NEOML_PYLAYER( "Lstm", "FmlCnnLstmLayer" )
REGISTER_NEOML_PYLAYER( "Gru", "FmlCnnGruLayer" )
REGISTER_NEOML_PYLAYER( "MaxOverTimePooling", "FmlCnnMaxOverTimePoolingLayer" )
REGISTER_NEOML_PYLAYER( "TimeConv", "FmlCnnTimeConvLayer" )
REGISTER_NEOML_PYLAYER( "3dConv", "FmlCnn3dConvLayer" )
REGISTER_NEOML_PYLAYER( "3dMaxPooling", "FmlCnn3dMaxPoolingLayer" )
REGISTER_NEOML_PYLAYER( "3dMeanPooling", "FmlCnn3dMeanPoolingLayer" )
REGISTER_NEOML_PYLAYER( "TransposedConv", "FmlCnnTransposedConvLayer" )
REGISTER_NEOML_PYLAYER( "3dTransposedConv", "FmlCnn3dTransposedConvLayer" )
REGISTER_NEOML_PYLAYER( "Crf", "FmlCnnCrfLayer" )
REGISTER_NEOML_PYLAYER( "CrfCalculation", "FmlCnnCrfCalculationLayer" )
REGISTER_NEOML_PYLAYER( "CrfLoss", "FmlCnnCrfLossLayer" )
REGISTER_NEOML_PYLAYER( "CrfInternalLoss", "FmlCnnCrfInternalLossLayer" )
REGISTER_NEOML_PYLAYER( "SequenceSum", "FmlCnnSequenceSumLayer" )
REGISTER_NEOML_PYLAYER( "BestSequence", "FmlCnnBestSequenceLayer" )
REGISTER_NEOML_PYLAYER( "CtcLoss", "FmlCnnCtcLossLayer" )
REGISTER_NEOML_PYLAYER( "CtcDecoding", "FmlCnnCtcDecodingLayer" )
REGISTER_NEOML_PYLAYER_EX( "Loss", "MultiHingeLoss", "FmlCnnMultyHingeLossLayer" )
REGISTER_NEOML_PYLAYER_EX( "Loss", "MultiSquaredHingeLoss", "FmlCnnMultySquaredHingeLossLayer" )
REGISTER_NEOML_PYLAYER( "Upsampling2D", "FmlCnnUpsampling2DLayer" )
REGISTER_NEOML_PYLAYER( "ChannelwiseConv", "FmlCnnChannelwiseConvLayer" )
REGISTER_NEOML_PYLAYER( "AccumulativeLookup", "FmlCnnAccumulativeLookupLayer" )
REGISTER_NEOML_PYLAYER( "Accuracy", "FmlCnnAccuracyLayer" )
REGISTER_NEOML_PYLAYER( "ConfusionMatrix", "FmlCnnConfusionMatrixLayer" )
REGISTER_NEOML_PYLAYER( "PrecisionRecall", "FmlCnnPrecisionRecallLayer" )
REGISTER_NEOML_PYLAYER_EX( "Loss", "CenterLoss", "FmlCnnCenterLossLayer" )
REGISTER_NEOML_PYLAYER_EX( "Loss", "FocalLoss", "FmlCnnFocalLossLayer" )
REGISTER_NEOML_PYLAYER_EX( "Loss", "BinaryFocalLoss", "FmlCnnBinaryFocalLossLayer" )
REGISTER_NEOML_PYLAYER( "ImageToPixel", "FmlCnnImageToPixelLayerClass" )
REGISTER_NEOML_PYLAYER( "PixelToImage", "FmlCnnPixelToImageLayerClass" )
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
REGISTER_NEOML_PYLAYER( "GELU", "NeoMLDnnGELULayer" )
REGISTER_NEOML_PYLAYER( "ProjectionPooling", "FmlCnnProjectionPoolingLayerClass" )

}

py::object createLayer( CBaseLayer& layer, CPyMathEngineOwner& mathEngineOwner )
{
	CPyLayer pyLayer( layer, mathEngineOwner );

	CMap<CString, CPyClass, CDefaultHash<CString>, RuntimeHeap>& layers = getRegisteredPyLayers();
	CString layerName( GetLayerName( layer ) );

	CPyClass c = layers.Get( layerName );

	CString wrapperModuleName = "neoml.PythonWrapper";
	CString moduleName = "neoml." + c.ModuleName;

	py::object wrapperModule = py::module::import( wrapperModuleName );
	py::object module = py::module::import( moduleName );

	py::object wrapperConstructor = wrapperModule.attr( c.ClassName );
	py::object wrapper = wrapperConstructor( pyLayer );

	py::object constructor = module.attr( c.ClassName );
	return constructor( wrapper );
}

//------------------------------------------------------------------------------------------------------------

CPyDnn::CPyDnn( CPyRandomOwner& _randomOwner, CPyMathEngineOwner& _mathEngineOwner ) :
	randomOwner( &_randomOwner ),
	mathEngineOwner( &_mathEngineOwner ),
	initializer( _randomOwner, new CDnnXavierInitializer( _randomOwner.Random() ) ),
	dnn( new CDnn( _randomOwner.Random(), _mathEngineOwner.MathEngine() ) )
{
}

CPyDnn::CPyDnn( CPyRandomOwner& _randomOwner, CPyMathEngineOwner& _mathEngineOwner, const std::string& path ) :
	randomOwner( &_randomOwner ),
	mathEngineOwner( &_mathEngineOwner ),
	initializer( _randomOwner, new CDnnXavierInitializer( _randomOwner.Random() ) ),
	dnn( new CDnn( _randomOwner.Random(), _mathEngineOwner.MathEngine() ) )
{
	CArchiveFile file( path.c_str(), CArchive::load );
	CArchive archive( &file, CArchive::load );
	dnn->Serialize( archive );
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
	CString moduleName = "neoml.Solver";

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
	CString moduleName = "neoml.Initializer";

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

py::list CPyDnn::Run( py::list inputs )
{
	CArray<const char*> layerNames;
	dnn->GetLayerList( layerNames );
			
	int index = 0;
	for( int layerIndex = 0; layerIndex < layerNames.Size(); ++layerIndex ) {
		CPtr<CSourceLayer> layer = dynamic_cast<CSourceLayer*>( dnn->GetLayer( layerNames[layerIndex] ).Ptr() );
		if( layer != 0 ) {
			CPyDnnBlob input = inputs[index].cast<CPyDnnBlob>();
			layer->SetBlob( &input.Blob() );
			index++;
		}
	}

	dnn->RunOnce();

	py::list result;
	for( int layerIndex = 0; layerIndex < layerNames.Size(); ++layerIndex ) {
		CPtr<CSinkLayer> layer = dynamic_cast<CSinkLayer*>( dnn->GetLayer( layerNames[layerIndex] ).Ptr() );
		if( layer != 0 ) {
			result.append( CreateArray( *layer->GetBlob() ) );
		}
	}

	return result;
}

void CPyDnn::RunAndBackward( py::list inputs )
{
	CArray<const char*> layerNames;
	dnn->GetLayerList( layerNames );
			
	int index = 0;
	for( int layerIndex = 0; layerIndex < layerNames.Size(); ++layerIndex ) {
		CPtr<CSourceLayer> layer = dynamic_cast<CSourceLayer*>( dnn->GetLayer( layerNames[layerIndex] ).Ptr() );
		if( layer != 0 ) {
			CPyDnnBlob input = inputs[index].cast<CPyDnnBlob>();
			layer->SetBlob( &input.Blob() );
			index++;
		}
	}

	dnn->RunAndBackwardOnce();
}

void CPyDnn::Learn( py::list inputs )
{
	CArray<const char*> layerNames;
	dnn->GetLayerList( layerNames );
			
	int index = 0;
	for( int layerIndex = 0; layerIndex < layerNames.Size(); ++layerIndex ) {
		CPtr<CSourceLayer> layer = dynamic_cast<CSourceLayer*>( dnn->GetLayer( layerNames[layerIndex] ).Ptr() );
		if( layer != 0 ) {
			CPyDnnBlob input = inputs[index].cast<CPyDnnBlob>();
			layer->SetBlob( &input.Blob() );
			index++;
		}
	}

	dnn->RunAndLearnOnce();
}

//------------------------------------------------------------------------------------------------------------

void InitializeDnn(py::module& m)
{
	py::class_<CPyDnn>(m, "Dnn")
		.def( py::init([]( const CPyRandom& random, const CPyMathEngine& mathEngine )
		{
			return new CPyDnn( random.RandomOwner(), mathEngine.MathEngineOwner() );
		}) )
		.def( py::init([]( const CPyRandom& random, const CPyMathEngine& mathEngine, const std::string& path )
		{
			return new CPyDnn( random.RandomOwner(), mathEngine.MathEngineOwner(), path );
		}) )
		.def( "get_math_engine", &CPyDnn::GetMathEngine, py::return_value_policy::reference )
		.def( "set_solver", &CPyDnn::SetSolver, py::return_value_policy::reference )
		.def( "get_solver", &CPyDnn::GetSolver, py::return_value_policy::reference )
		.def( "set_initializer", &CPyDnn::SetInitializer, py::return_value_policy::reference )
		.def( "get_initializer", &CPyDnn::GetInitializer, py::return_value_policy::reference )
		.def( "get_inputs", &CPyDnn::GetInputs, py::return_value_policy::reference )
		.def( "get_outputs", &CPyDnn::GetOutputs, py::return_value_policy::reference )
		.def( "get_layers", &CPyDnn::GetLayers, py::return_value_policy::reference )
		.def( "_run", &CPyDnn::Run, py::return_value_policy::reference )
		.def( "_run_and_backward", &CPyDnn::RunAndBackward, py::return_value_policy::reference )
		.def( "_learn", &CPyDnn::Learn, py::return_value_policy::reference )
	;
}
