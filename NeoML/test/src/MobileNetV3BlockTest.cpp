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

#include <TestFixture.h>

#include <NeoML/Dnn/Layers/Onnx/OnnxEltwiseLayer.h>

namespace NeoMLTest {

static CBaseLayer* addMNv3Activation( const char* name, CBaseLayer& input, const CActivationDesc& desc )
{
	CDnn& dnn = *input.GetDnn();
	IMathEngine& mathEngine = dnn.GetMathEngine();

	CPtr<CBaseLayer> result = CreateActivationLayer( mathEngine, desc );
	result->SetName( name );
	result->Connect( input );
	dnn.AddLayer( *result );
	return result.Ptr();
}

struct CPreSEParams {
	CPtr<CDnnBlob> ExpandFilter;
	CPtr<CDnnBlob> ExpandFreeTerm;
	CActivationDesc ExpandActivation = AF_HSwish;
	CPtr<CDnnBlob> ChannelwiseFilter;
	CPtr<CDnnBlob> ChannelwiseFreeTerm;
	int ChannelwiseStride;
	CActivationDesc ChannelwiseActivation = AF_HSwish;
};

static CBaseLayer* addMNv3PreSE( const CPreSEParams& params, CBaseLayer& input )
{
	CDnn& dnn = *input.GetDnn();
	IMathEngine& mathEngine = dnn.GetMathEngine();

	CPtr<CConvLayer> expandConv = new CConvLayer( mathEngine );
	expandConv->SetName( "ExpandConv" );
	expandConv->SetFilterCount( params.ExpandFilter->GetObjectCount() );
	expandConv->SetFilterData( params.ExpandFilter );
	if( params.ExpandFreeTerm == nullptr ) {
		expandConv->SetZeroFreeTerm( true );
	} else {
		expandConv->SetZeroFreeTerm( false );
		expandConv->SetFreeTermData( params.ExpandFreeTerm );
	}
	expandConv->Connect( input );
	dnn.AddLayer( *expandConv );

	CBaseLayer* expandActivationLayer = addMNv3Activation( "expandActivation", *expandConv,
		params.ExpandActivation );

	CPtr<CChannelwiseConvLayer> channelwise = new CChannelwiseConvLayer( mathEngine );
	channelwise->SetFilterCount( params.ChannelwiseFilter->GetChannelsCount() );
	channelwise->SetFilterHeight( 3 );
	channelwise->SetFilterWidth( 3 );
	channelwise->SetFilterData( params.ChannelwiseFilter );
	channelwise->SetPaddingHeight( 1 );
	channelwise->SetPaddingWidth( 1 );
	channelwise->SetStrideHeight( params.ChannelwiseStride );
	channelwise->SetStrideWidth( params.ChannelwiseStride );
	if( params.ChannelwiseFreeTerm == nullptr ) {
		channelwise->SetZeroFreeTerm( true );
	} else {
		channelwise->SetZeroFreeTerm( false );
		channelwise->SetFreeTermData( params.ChannelwiseFreeTerm );
	}
	channelwise->Connect( *expandActivationLayer );
	dnn.AddLayer( *channelwise );

	CBaseLayer* channelwiseActivationLayer = addMNv3Activation( "channelwiseActivation", *channelwise,
		params.ChannelwiseActivation );
	return channelwiseActivationLayer;
}

struct CSEParams {
	CPtr<CDnnBlob> FirstWeight;
	CPtr<CDnnBlob> FirstFreeTerm;
	CActivationDesc FirstActivation = AF_HSwish;
	CPtr<CDnnBlob> SecondWeight;
	CPtr<CDnnBlob> SecondFreeTerm;
	CActivationDesc SecondActivation = AF_HSwish;
};

static CBaseLayer* addMNv3SE( const CSEParams& params, CBaseLayer& input )
{
	CDnn& dnn = *input.GetDnn();
	IMathEngine& mathEngine = dnn.GetMathEngine();

	CPtr<CGlobalMeanPoolingLayer> pooling = new CGlobalMeanPoolingLayer( mathEngine );
	pooling->SetName( "SEPooling" );
	pooling->Connect( input );
	dnn.AddLayer( *pooling );

	CPtr<CFullyConnectedLayer> firstFc = new CFullyConnectedLayer( mathEngine, "SEFirstFc" );
	firstFc->SetNumberOfElements( params.FirstWeight->GetObjectCount() );
	firstFc->SetWeightsData( params.FirstWeight.Ptr() );
	firstFc->SetZeroFreeTerm( params.FirstFreeTerm == nullptr );
	firstFc->SetFreeTermData( params.FirstFreeTerm.Ptr() );
	firstFc->Connect( *pooling );
	dnn.AddLayer( *firstFc );

	CBaseLayer* firstActivation = addMNv3Activation( "SEFirstActivation", *firstFc,
		params.FirstActivation );

	CPtr<CFullyConnectedLayer> secondFc = new CFullyConnectedLayer( mathEngine, "SESecondFc" );
	secondFc->SetNumberOfElements( params.SecondWeight->GetObjectCount() );
	secondFc->SetWeightsData( params.SecondWeight.Ptr() );
	secondFc->SetZeroFreeTerm( params.SecondFreeTerm == nullptr );
	secondFc->SetFreeTermData( params.SecondFreeTerm.Ptr() );
	secondFc->Connect( *firstActivation );
	dnn.AddLayer( *secondFc );

	return addMNv3Activation( "SESecondActivation", *secondFc,
		params.SecondActivation );
}

struct CPostSEParams {
	CActivationDesc PostSEActivation = AF_HSwish;
	CPtr<CDnnBlob> DownFilter;
	CPtr<CDnnBlob> DownFreeTerm;
	bool Residual;
};

static CBaseLayer* addMNv3PostSE( const CPostSEParams& params, CBaseLayer& input,
	CBaseLayer& preSe, CBaseLayer& se )
{
	CDnn& dnn = *input.GetDnn();
	IMathEngine& mathEngine = dnn.GetMathEngine();

	CPtr<COnnxEltwiseLayer> mul = new COnnxEltwiseLayer( mathEngine );
	mul->SetName( "SEMul" );
	mul->SetOperation( COnnxEltwiseLayer::TOperation::Mul );
	mul->Connect( se );
	mul->Connect( 1, preSe );
	dnn.AddLayer( *mul );

	CBaseLayer* activation = addMNv3Activation( "PostSEActivation", *mul,
		params.PostSEActivation );

	CPtr<CConvLayer> downConv = new CConvLayer( mathEngine );
	downConv->SetName( "DownConv" );
	downConv->SetFilterCount( params.DownFilter->GetObjectCount() );
	downConv->SetFilterData( params.DownFilter );
	if( params.DownFreeTerm == nullptr ) {
		downConv->SetZeroFreeTerm( true );
	} else {
		downConv->SetZeroFreeTerm( false );
		downConv->SetFreeTermData( params.DownFreeTerm );
	}
	downConv->Connect( *activation );
	dnn.AddLayer( *downConv );

	if( !params.Residual ) {
		return downConv.Ptr();
	}

	CPtr<CEltwiseSumLayer> sum = new CEltwiseSumLayer( mathEngine );
	sum->SetName( "ResidualSum" );
	sum->Connect( 0, input );
	sum->Connect(  1, *downConv );
	dnn.AddLayer( *sum );

	return sum.Ptr();
}

} // namespace NeoMLTest

using namespace NeoML;
using namespace NeoMLTest;

static void mobileNetV3BlockTestImpl( unsigned int seed, int freeTermMask, const CActivationDesc& expandActivation,
	const CActivationDesc& channelwiseActivation, const CActivationDesc& firstSEActivation,
	const CActivationDesc& secondSEActivation, const CActivationDesc& postSEActivation,
	int stride, bool residual )
{
	auto createBlob = [] ( const std::initializer_list<int>& dims, CRandom& random ) -> CPtr<CDnnBlob> {
		CPtr<CDnnBlob> blob = CDnnBlob::CreateTensor( MathEngine(), CT_Float, dims );
		CREATE_FILL_FLOAT_ARRAY( data, -1, 1, blob->GetDataSize(), random );
		blob->CopyFrom( data.GetPtr() );
		return blob;
	};

	NeoAssert( stride == 1 || stride == 2 );
	NeoAssert( freeTermMask >= 0 && freeTermMask < 32 );
	NeoAssert( !residual || stride == 1 );

	CRandom random( seed );

	const int expandFreeTermBit = 1 << 0;
	const int channelwiseFreeTermBit = 1 << 1;
	const int firstFcFreeTermBit = 1 << 2;
	const int secondFcFreeTermBit = 1 << 3;
	const int downFreeTermBit = 1 << 4;

	const int batch = 3;
	const int inputChannels = 4;
	const int outputChannels = residual ? inputChannels : 6;
	const int seChannels = 9;
	const int expandedChannels = 8;
	const int imageHeight = 13;
	const int imageWidth = 15;

	CPreSEParams preSEParams;
	preSEParams.ExpandFilter = createBlob( { 1, expandedChannels, 1, 1, 1, 1, inputChannels }, random );
	if( ( freeTermMask & expandFreeTermBit ) != 0 ) {
		preSEParams.ExpandFreeTerm = createBlob( { expandedChannels }, random );
	}
	preSEParams.ExpandActivation = expandActivation;

	preSEParams.ChannelwiseFilter = createBlob( { 1, 1, 1, 3, 3, 1, expandedChannels }, random );
	if( ( freeTermMask & channelwiseFreeTermBit ) != 0 ) {
		preSEParams.ChannelwiseFreeTerm = createBlob( { expandedChannels }, random );
	}
	preSEParams.ChannelwiseStride = stride;
	preSEParams.ChannelwiseActivation = channelwiseActivation;

	CSEParams seParams;
	seParams.FirstWeight = createBlob( { 1, seChannels, 1, 1, 1, 1, expandedChannels }, random );
	if( ( freeTermMask & firstFcFreeTermBit ) != 0 ) {
		seParams.FirstFreeTerm = createBlob( { seChannels }, random );
	}
	seParams.FirstActivation = firstSEActivation;

	seParams.SecondWeight = createBlob( { 1, expandedChannels, 1, 1, 1, 1, seChannels }, random );
	if( ( freeTermMask & secondFcFreeTermBit ) != 0 ) {
		seParams.SecondFreeTerm = createBlob( { expandedChannels }, random );
	}
	seParams.SecondActivation = secondSEActivation;

	CPostSEParams postSEParams;
	postSEParams.DownFilter = createBlob( { 1, outputChannels, 1, 1, 1, 1, expandedChannels }, random );
	if( ( freeTermMask & downFreeTermBit ) != 0 ) {
		postSEParams.DownFreeTerm = createBlob( { outputChannels }, random );
	}
	postSEParams.Residual = residual;

	CDnn expectedDnn( random, MathEngine() );
	CPtr<CSourceLayer> expectedData = AddLayer<CSourceLayer>( "Data", expectedDnn );
	CBaseLayer* expectedPreSE = addMNv3PreSE( preSEParams, *expectedData );
	CBaseLayer* expectedSE = addMNv3SE( seParams, *expectedPreSE );
	CBaseLayer* expandPostSE = addMNv3PostSE( postSEParams, *expectedData, *expectedPreSE, *expectedSE );
	CPtr<CSinkLayer> expectedSink = AddLayer<CSinkLayer>( "expectedSink", { expandPostSE } );

	CDnn actualDnn( random, MathEngine() );
	CPtr<CSourceLayer> actualData = AddLayer<CSourceLayer>( "Data", actualDnn );
	CPtr<CMobileNetV3PreSEBlockLayer> actualPreSE = AddLayer<CMobileNetV3PreSEBlockLayer>(
		new CMobileNetV3PreSEBlockLayer( MathEngine(), preSEParams.ExpandFilter, preSEParams.ExpandFreeTerm,
			preSEParams.ExpandActivation, preSEParams.ChannelwiseStride, preSEParams.ChannelwiseFilter,
			preSEParams.ChannelwiseFreeTerm, preSEParams.ChannelwiseActivation ),
		"actualPreSE", { actualData } );
	CBaseLayer* actualSE = addMNv3SE( seParams, *actualPreSE );
	CPtr<CMobileNetV3PostSEBlockLayer> actualPostSE = AddLayer<CMobileNetV3PostSEBlockLayer>(
		new CMobileNetV3PostSEBlockLayer( MathEngine(), postSEParams.PostSEActivation, postSEParams.DownFilter,
			postSEParams.DownFreeTerm ),
		"actualPostSE", { actualPreSE, actualSE } );
	if( postSEParams.Residual ) {
		actualPostSE->Connect( 2, *actualData );
	}
	CPtr<CSinkLayer> actualSink = AddLayer<CSinkLayer>( "actualSink", { actualPostSE } );

	actualData->SetBlob( createBlob( { 1, batch, 1, imageHeight, imageWidth, 1, inputChannels }, random ) );
	expectedData->SetBlob( actualData->GetBlob() );
	
	actualDnn.RunOnce();
	expectedDnn.RunOnce();

	CPtr<CDnnBlob> expectedBlob = expectedSink->GetBlob();
	CPtr<CDnnBlob> actualBlob = actualSink->GetBlob();

	CDnnBlobBuffer<float> expected( *expectedBlob, TDnnBlobBufferAccess::Read );
	CDnnBlobBuffer<float> actual( *actualBlob, TDnnBlobBufferAccess::Read );

	ASSERT_EQ( expected.Size(), actual.Size() ) << "output size mismatch";
	for( int i = 0; i < expected.Size(); ++i ) {
		ASSERT_NEAR( expected[i], actual[i], 1e-3 ) << "at index " << i;
	}
}

TEST( MobileNetV3BlockLayerTest, Run )
{
	CRandom seedRandom( 0x654 );

	std::vector<CActivationDesc> blockActivations = {
		CActivationDesc( AF_ReLU, CReLULayer::CParam{ -1.f } ),
		CActivationDesc( AF_ReLU, CReLULayer::CParam{ 6.f } ),
		CActivationDesc( AF_HSwish ),
		CActivationDesc( AF_Linear, CLinearLayer::CParam{ 1.f, 0.f } ) };
	std::vector<CActivationDesc> seActivations = {
		CActivationDesc( AF_ReLU, CReLULayer::CParam{ -1.f } ),
		CActivationDesc( AF_ReLU, CReLULayer::CParam{ 6.f } ),
		CActivationDesc( AF_HardSigmoid, CHardSigmoidLayer::CParam{ 0.5f, 0.5f } ) };

	for( int ftMask = 0; ftMask < 32; ++ftMask ) {
		for( const CActivationDesc& expandActivation : blockActivations ) {
			for( const CActivationDesc& channelwiseActivation : blockActivations ) {
				for( const CActivationDesc& firstSEActivation : seActivations ) {
					for( const CActivationDesc& secondSEActivation : seActivations ) {
						for( const CActivationDesc& postSEActivation : blockActivations ) {
							mobileNetV3BlockTestImpl( seedRandom.Next(), ftMask, expandActivation, channelwiseActivation,
								firstSEActivation, secondSEActivation, postSEActivation, 1, true );
							mobileNetV3BlockTestImpl( seedRandom.Next(), ftMask, expandActivation, channelwiseActivation,
								firstSEActivation, secondSEActivation, postSEActivation, 1, false );
							mobileNetV3BlockTestImpl( seedRandom.Next(), ftMask, expandActivation, channelwiseActivation,
								firstSEActivation, secondSEActivation, postSEActivation, 2, false );
						}
					}
				}
			}
		}
	}
}
/*
TEST( MobileNetv2OptimizerTest, SimpleNonResidual )
{
	CRandom random( 0x654 );
	CDnn dnn( random, MathEngine() );
	CSourceLayer* data = Source( dnn, "data" );
	CConvLayer* expandConv = Conv( 32, CConvAxisParams( 1 ), CConvAxisParams( 1 ) )( "expandConv", data );
	CReLULayer* expandReLU = Relu()( "expandReLU", expandConv );
	CChannelwiseConvLayer* channelwiseConv = ChannelwiseConv( 32, CConvAxisParams( 3, 1, 2 ), CConvAxisParams( 3, 1, 2 ) )
		( "channewlseConv", expandReLU );
	CReLULayer* channelwiseReLU = Relu( 6.f )( "channelwiseReLU", channelwiseConv );
	CConvLayer* downConv = Conv( 8, CConvAxisParams( 1 ), CConvAxisParams( 1 ) )( "downConv", channelwiseReLU );
	Sink( downConv, "sink" );
	CDnnOptimizationReport report = OptimizeDnn( dnn );
	ASSERT_EQ( 1, report.MobileNetV2NonResidualBlocks );
	ASSERT_EQ( 0, report.MobileNetV2ResidualBlocks );
	ASSERT_EQ( 3, dnn.GetLayerCount() );
}

TEST( MobileNetv2OptimizerTest, SimpleResidual )
{
	CRandom random( 0x654 );
	CDnn dnn( random, MathEngine() );
	CSourceLayer* data = Source( dnn, "data" );
	CConvLayer* expandConv = Conv( 32, CConvAxisParams( 1 ), CConvAxisParams( 1 ) )( "expandConv", data );
	CReLULayer* expandReLU = Relu()( "expandReLU", expandConv );
	CChannelwiseConvLayer* channelwiseConv = ChannelwiseConv( 32, CConvAxisParams( 3, 1, 1 ), CConvAxisParams( 3, 1, 1 ) )
		( "channewlseConv", expandReLU );
	CReLULayer* channelwiseReLU = Relu( 6.f )( "channelwiseReLU", channelwiseConv );
	CConvLayer* downConv = Conv( 8, CConvAxisParams( 1 ), CConvAxisParams( 1 ) )( "downConv", channelwiseReLU );
	CEltwiseSumLayer* residual = Sum()( "residual", data, downConv );
	Sink( residual, "sink" );
	CDnnOptimizationReport report = OptimizeDnn( dnn );
	ASSERT_EQ( 0, report.MobileNetV2NonResidualBlocks );
	ASSERT_EQ( 1, report.MobileNetV2ResidualBlocks );
	ASSERT_EQ( 3, dnn.GetLayerCount() );
}

TEST( MobileNetv2OptimizerTest, ResidualResidual )
{
	CRandom random( 0x654 );
	CDnn dnn( random, MathEngine() );
	CSourceLayer* data = Source( dnn, "data" );
	CConvLayer* expandConv = Conv( 32, CConvAxisParams( 1 ), CConvAxisParams( 1 ) )( "expandConv", data );
	CReLULayer* expandReLU = Relu()( "expandReLU", expandConv );
	CChannelwiseConvLayer* channelwiseConv = ChannelwiseConv( 32, CConvAxisParams( 3, 1, 1 ), CConvAxisParams( 3, 1, 1 ) )
		( "channewlseConv", expandReLU );
	CReLULayer* channelwiseReLU = Relu( 6.f )( "channelwiseReLU", channelwiseConv );
	CConvLayer* downConv = Conv( 8, CConvAxisParams( 1 ), CConvAxisParams( 1 ) )( "downConv", channelwiseReLU );
	CEltwiseSumLayer* residual = Sum()( "residual", data, downConv );
	CEltwiseSumLayer* doubleResidual = Sum()( "doubleResidual", data, residual );
	Sink( doubleResidual, "sink" );
	CDnnOptimizationReport report = OptimizeDnn( dnn );
	ASSERT_EQ( 0, report.MobileNetV2NonResidualBlocks );
	ASSERT_EQ( 1, report.MobileNetV2ResidualBlocks );
	ASSERT_EQ( 4, dnn.GetLayerCount() );
}

TEST( MobileNetv2OptimizerTest, NeighboringResiduals )
{
	CRandom random( 0x654 );
	CDnn dnn( random, MathEngine() );
	CSourceLayer* data = Source( dnn, "data" );
	CConvLayer* expandConv = Conv( 32, CConvAxisParams( 1 ), CConvAxisParams( 1 ) )( "expandConv", data );
	CReLULayer* expandReLU = Relu()( "expandReLU", expandConv );
	CChannelwiseConvLayer* channelwiseConv = ChannelwiseConv( 32, CConvAxisParams( 3, 1, 1 ), CConvAxisParams( 3, 1, 1 ) )
		( "channewlseConv", expandReLU );
	CReLULayer* channelwiseReLU = Relu( 6.f )( "channelwiseReLU", channelwiseConv );
	CConvLayer* downConv = Conv( 8, CConvAxisParams( 1 ), CConvAxisParams( 1 ) )( "downConv", channelwiseReLU );
	CEltwiseSumLayer* residual = Sum()( "residual", data, downConv );
	Sink( residual, "sink" );
	CEltwiseSumLayer* secondResidual = Sum()( "secondResidual", data, downConv );
	Sink( secondResidual, "secondSink" );
	CDnnOptimizationReport report = OptimizeDnn( dnn );
	ASSERT_EQ( 1, report.MobileNetV2NonResidualBlocks );
	ASSERT_EQ( 0, report.MobileNetV2ResidualBlocks );
	ASSERT_EQ( 6, dnn.GetLayerCount() );
}

TEST( MobileNetv2OptimizerTest, SinkFromTheMiddle )
{
	CRandom random( 0x654 );
	CDnn dnn( random, MathEngine() );
	CSourceLayer* data = Source( dnn, "data" );
	CConvLayer* expandConv = Conv( 32, CConvAxisParams( 1 ), CConvAxisParams( 1 ) )( "expandConv", data );
	CReLULayer* expandReLU = Relu()( "expandReLU", expandConv );
	CChannelwiseConvLayer* channelwiseConv = ChannelwiseConv( 32, CConvAxisParams( 3, 1, 1 ), CConvAxisParams( 3, 1, 1 ) )
		( "channewlseConv", expandReLU );
	Sink( channelwiseConv, "channelwiseSink" );
	CReLULayer* channelwiseReLU = Relu( 6.f )( "channelwiseReLU", channelwiseConv );
	CConvLayer* downConv = Conv( 8, CConvAxisParams( 1 ), CConvAxisParams( 1 ) )( "downConv", channelwiseReLU );
	CEltwiseSumLayer* residual = Sum()( "residual", data, downConv );
	Sink( residual, "sink" );
	CDnnOptimizationReport report = OptimizeDnn( dnn );
	ASSERT_EQ( 0, report.MobileNetV2NonResidualBlocks );
	ASSERT_EQ( 0, report.MobileNetV2ResidualBlocks );
	ASSERT_EQ( 9, dnn.GetLayerCount() );
}

TEST( MobileNetv2OptimizerTest, SinkDisablesResidual )
{
	CRandom random( 0x654 );
	CDnn dnn( random, MathEngine() );
	CSourceLayer* data = Source( dnn, "data" );
	CConvLayer* expandConv = Conv( 32, CConvAxisParams( 1 ), CConvAxisParams( 1 ) )( "expandConv", data );
	CReLULayer* expandReLU = Relu()( "expandReLU", expandConv );
	CChannelwiseConvLayer* channelwiseConv = ChannelwiseConv( 32, CConvAxisParams( 3, 1, 1 ), CConvAxisParams( 3, 1, 1 ) )
		( "channewlseConv", expandReLU );
	CReLULayer* channelwiseReLU = Relu( 6.f )( "channelwiseReLU", channelwiseConv );
	CConvLayer* downConv = Conv( 8, CConvAxisParams( 1 ), CConvAxisParams( 1 ) )( "downConv", channelwiseReLU );
	Sink( downConv, "downConvSink" );
	CEltwiseSumLayer* residual = Sum()( "residual", data, downConv );
	Sink( residual, "sink" );
	CDnnOptimizationReport report = OptimizeDnn( dnn );
	ASSERT_EQ( 1, report.MobileNetV2NonResidualBlocks );
	ASSERT_EQ( 0, report.MobileNetV2ResidualBlocks );
	ASSERT_EQ( 5, dnn.GetLayerCount() );
}
*/
