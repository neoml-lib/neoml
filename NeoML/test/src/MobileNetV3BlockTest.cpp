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

#include <initializer_list>

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
	CActivationDesc FirstActivation = AF_HardSigmoid;
	CPtr<CDnnBlob> SecondWeight;
	CPtr<CDnnBlob> SecondFreeTerm;
	CActivationDesc SecondActivation = AF_HardSigmoid;
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
	postSEParams.PostSEActivation = postSEActivation;
	postSEParams.DownFilter = createBlob( { 1, outputChannels, 1, 1, 1, 1, expandedChannels }, random );
	if( ( freeTermMask & downFreeTermBit ) != 0 ) {
		postSEParams.DownFreeTerm = createBlob( { outputChannels }, random );
	}
	postSEParams.Residual = residual;

	CDnn expectedDnn( random, MathEngine() );
	CSourceLayer* expectedData = Source( expectedDnn, "Data" );
	CBaseLayer* expectedPreSE = addMNv3PreSE( preSEParams, *expectedData );
	CBaseLayer* expectedSE = addMNv3SE( seParams, *expectedPreSE );
	CBaseLayer* expandPostSE = addMNv3PostSE( postSEParams, *expectedData, *expectedPreSE, *expectedSE );
	CSinkLayer* expectedSink = Sink( expandPostSE, "expectedSink" );

	CDnn actualDnn( random, MathEngine() );
	CSourceLayer* actualData = Source( actualDnn, "Data" );
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
	CSinkLayer* actualSink = Sink( actualPostSE.Ptr(), "actualSink" );

	actualData->SetBlob( createBlob( { 1, batch, 1, imageHeight, imageWidth, 1, inputChannels }, random ) );
	expectedData->SetBlob( actualData->GetBlob() );
	
	actualDnn.RunOnce();
	expectedDnn.RunOnce();

	CPtr<CDnnBlob> expectedBlob = expectedSink->GetBlob();
	CPtr<CDnnBlob> actualBlob = actualSink->GetBlob();

	CDnnBlobBuffer<float> expected( *expectedBlob, TDnnBlobBufferAccess::Read );
	CDnnBlobBuffer<float> actual( *actualBlob, TDnnBlobBufferAccess::Read );

	EXPECT_EQ( expected.Size(), actual.Size() ) << "output size mismatch";
	for( int i = 0; i < expected.Size(); ++i ) {
		EXPECT_NEAR( expected[i], actual[i], 1e-3 ) << "at index " << i;
	}
}

static std::initializer_list<CActivationDesc> mnv3BlockActivations = {
	CActivationDesc( AF_ReLU, CReLULayer::CParam{ -1.f } ),
	CActivationDesc( AF_ReLU, CReLULayer::CParam{ 6.f } ),
	CActivationDesc( AF_HSwish ),
	CActivationDesc( AF_Linear, CLinearLayer::CParam{ 1.f, 0.f } ) };
static std::initializer_list<CActivationDesc> mnv3SeActivations = {
	CActivationDesc( AF_ReLU, CReLULayer::CParam{ -1.f } ),
	CActivationDesc( AF_ReLU, CReLULayer::CParam{ 6.f } ),
	CActivationDesc( AF_HardSigmoid, CHardSigmoidLayer::CParam{ 0.5f, 0.5f } ) };

TEST( MobileNetV3BlockLayerTest, Run )
{
	const auto met = MathEngine().GetType();
	if(met != MET_Cpu && met != MET_Cuda) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		// MobileNetV3PreSEBlock
		return;
	}

	// This test is allowed on GPU because of backward compatibility
	CRandom seedRandom( 0x654 );

	for( int ftMask = 0; ftMask < 32; ++ftMask ) {
		for( const CActivationDesc& expandActivation : mnv3BlockActivations ) {
			for( const CActivationDesc& channelwiseActivation : mnv3BlockActivations ) {
				for( const CActivationDesc& firstSEActivation : mnv3SeActivations ) {
					for( const CActivationDesc& secondSEActivation : mnv3SeActivations ) {
						for( const CActivationDesc& postSEActivation : mnv3BlockActivations ) {
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

TEST( MobileNetV3OptimizerTest, SimpleNonResidual )
{
	for( const CActivationDesc& expandActivation : mnv3BlockActivations ) {
		for( const CActivationDesc& channelwiseActivation : mnv3BlockActivations ) {
			for( const CActivationDesc& firstSEActivation : mnv3SeActivations ) {
				for( const CActivationDesc& secondSEActivation : mnv3SeActivations ) {
					for( const CActivationDesc& postSEActivation : mnv3BlockActivations ) {
						CRandom random( 0x654 );
						CDnn dnn( random, MathEngine() );
						CSourceLayer* data = Source( dnn, "data" );
						CPreSEParams preSEParams;
						preSEParams.ExpandFilter = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1, 12, 1, 1, 8 );
						preSEParams.ExpandActivation = expandActivation;
						preSEParams.ChannelwiseFilter = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1, 1, 5, 5, 12 );
						preSEParams.ChannelwiseStride = 2;
						preSEParams.ChannelwiseActivation = channelwiseActivation;
						CBaseLayer* preSE = addMNv3PreSE( preSEParams, *data );
						CSEParams seParams;
						seParams.FirstWeight = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 16, 12 );
						seParams.FirstActivation = firstSEActivation;
						seParams.SecondWeight = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 12, 16 );
						seParams.SecondActivation = secondSEActivation;
						CBaseLayer* se = addMNv3SE( seParams, *preSE );
						CPostSEParams postSEParams;
						postSEParams.PostSEActivation = postSEActivation;
						postSEParams.DownFilter = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1, 10, 1, 1, 12 );
						postSEParams.Residual = false;
						CBaseLayer* postSE = addMNv3PostSE( postSEParams, *data, *preSE, *se );
						Sink( postSE, "sink" );
						CDnnOptimizationReport report = OptimizeDnn( dnn );
						EXPECT_EQ( 1, report.MobileNetV3NonResidualBlocks );
						EXPECT_EQ( 0, report.MobileNetV3ResidualBlocks );
						EXPECT_EQ( 9, dnn.GetLayerCount() );
					}
				}
			}
		}
	}
}

TEST( MobileNetV3OptimizerTest, SimpleResidual )
{
	CRandom random( 0x654 );
	CDnn dnn( random, MathEngine() );
	CSourceLayer* data = Source( dnn, "data" );
	CPreSEParams preSEParams;
	preSEParams.ExpandFilter = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1, 12, 1, 1, 8 );
	preSEParams.ChannelwiseFilter = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1, 1, 5, 5, 12 );
	preSEParams.ChannelwiseStride = 2;
	CBaseLayer* preSE = addMNv3PreSE( preSEParams, *data );
	CSEParams seParams;
	seParams.FirstWeight = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 16, 12 );
	seParams.SecondWeight = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 12, 16 );\
		CBaseLayer* se = addMNv3SE( seParams, *preSE );
	CPostSEParams postSEParams;
	postSEParams.DownFilter = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1, 10, 1, 1, 12 );
	postSEParams.Residual = true;
	CBaseLayer* postSE = addMNv3PostSE( postSEParams, *data, *preSE, *se );
	Sink( postSE, "sink" );
	CDnnOptimizationReport report = OptimizeDnn( dnn );
	EXPECT_EQ( 0, report.MobileNetV3NonResidualBlocks );
	EXPECT_EQ( 1, report.MobileNetV3ResidualBlocks );
	EXPECT_EQ( 9, dnn.GetLayerCount() );
}

TEST( MobileNetV3OptimizerTest, ResidualResidual )
{
	CRandom random( 0x654 );
	CDnn dnn( random, MathEngine() );
	CSourceLayer* data = Source( dnn, "data" );
	CPreSEParams preSEParams;
	preSEParams.ExpandFilter = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1, 12, 1, 1, 8 );
	preSEParams.ChannelwiseFilter = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1, 1, 5, 5, 12 );
	preSEParams.ChannelwiseStride = 2;
	CBaseLayer* preSE = addMNv3PreSE( preSEParams, *data );
	CSEParams seParams;
	seParams.FirstWeight = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 16, 12 );
	seParams.SecondWeight = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 12, 16 );\
		CBaseLayer* se = addMNv3SE( seParams, *preSE );
	CPostSEParams postSEParams;
	postSEParams.DownFilter = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1, 10, 1, 1, 12 );
	postSEParams.Residual = true;
	CBaseLayer* postSE = addMNv3PostSE( postSEParams, *data, *preSE, *se );
	Sink( postSE, "sink" );
	CEltwiseSumLayer* secondResidual = Sum()( "secondResidual", data, postSE );
	Sink( secondResidual, "secondSink" );
	CDnnOptimizationReport report = OptimizeDnn( dnn );
	EXPECT_EQ( 0, report.MobileNetV3NonResidualBlocks );
	EXPECT_EQ( 1, report.MobileNetV3ResidualBlocks );
	EXPECT_EQ( 11, dnn.GetLayerCount() );
}

TEST( MobileNetV3OptimizerTest, NeighboringResiduals )
{
	CRandom random( 0x654 );
	CDnn dnn( random, MathEngine() );
	CSourceLayer* data = Source( dnn, "data" );
	CPreSEParams preSEParams;
	preSEParams.ExpandFilter = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1, 12, 1, 1, 8 );
	preSEParams.ChannelwiseFilter = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1, 1, 5, 5, 12 );
	preSEParams.ChannelwiseStride = 2;
	CBaseLayer* preSE = addMNv3PreSE( preSEParams, *data );
	CSEParams seParams;
	seParams.FirstWeight = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 16, 12 );
	seParams.SecondWeight = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 12, 16 );\
		CBaseLayer* se = addMNv3SE( seParams, *preSE );
	CPostSEParams postSEParams;
	postSEParams.DownFilter = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1, 10, 1, 1, 12 );
	postSEParams.Residual = true;
	CBaseLayer* postSE = addMNv3PostSE( postSEParams, *data, *preSE, *se );
	Sink( postSE, "sink" );
	CEltwiseSumLayer* secondResidual = Sum()( "secondResidual", dnn.GetLayer( "DownConv" ).Ptr(), postSE );
	Sink( secondResidual, "secondSink" );
	CDnnOptimizationReport report = OptimizeDnn( dnn );
	EXPECT_EQ( 1, report.MobileNetV3NonResidualBlocks );
	EXPECT_EQ( 0, report.MobileNetV3ResidualBlocks );
	EXPECT_EQ( 12, dnn.GetLayerCount() );
}

TEST( MobileNetV3OptimizerTest, SinkFromTheMiddle )
{
	for( int testIndex = 0; testIndex < 4; ++testIndex ) {
		CRandom random( 0x654 );
		CDnn dnn( random, MathEngine() );
		CSourceLayer* data = Source( dnn, "data" );
		CPreSEParams preSEParams;
		preSEParams.ExpandFilter = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1, 12, 1, 1, 8 );
		preSEParams.ChannelwiseFilter = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1, 1, 5, 5, 12 );
		preSEParams.ChannelwiseStride = 2;
		CBaseLayer* preSE = addMNv3PreSE( preSEParams, *data );
		CSEParams seParams;
		seParams.FirstWeight = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 16, 12 );
		seParams.SecondWeight = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 12, 16 );\
			CBaseLayer* se = addMNv3SE( seParams, *preSE );
		CPostSEParams postSEParams;
		postSEParams.DownFilter = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1, 10, 1, 1, 12 );
		postSEParams.Residual = true;
		CBaseLayer* postSE = addMNv3PostSE( postSEParams, *data, *preSE, *se );
		Sink( postSE, "sink" );
		switch( testIndex ) {
			case 0:
				Sink( dnn.GetLayer( "ExpandConv" ).Ptr(), "breakingSink" );
				break;
			case 1:
				Sink( dnn.GetLayer( "CCnnChannelwiseConvLayer" ).Ptr(), "breakingSink" );
				break;
			case 2:
				Sink( dnn.GetLayer( "SEMul" ).Ptr(), "breakingSink" );
				break;
			case 3:
				Sink( dnn.GetLayer( "PostSEActivation" ).Ptr(), "breakingSink" );
				break;
			default:
				FAIL();
		}
		CDnnOptimizationReport report = OptimizeDnn( dnn );
		EXPECT_EQ( 0, report.MobileNetV3NonResidualBlocks );
		EXPECT_EQ( 0, report.MobileNetV3ResidualBlocks );
	}
}

TEST( MobileNetV3OptimizerTest, SinkDisablesResidual )
{
	CRandom random( 0x654 );
	CDnn dnn( random, MathEngine() );
	CSourceLayer* data = Source( dnn, "data" );
	CPreSEParams preSEParams;
	preSEParams.ExpandFilter = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1, 12, 1, 1, 8 );
	preSEParams.ChannelwiseFilter = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1, 1, 5, 5, 12 );
	preSEParams.ChannelwiseStride = 2;
	CBaseLayer* preSE = addMNv3PreSE( preSEParams, *data );
	CSEParams seParams;
	seParams.FirstWeight = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 16, 12 );
	seParams.SecondWeight = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 12, 16 );\
		CBaseLayer* se = addMNv3SE( seParams, *preSE );
	CPostSEParams postSEParams;
	postSEParams.DownFilter = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1, 10, 1, 1, 12 );
	postSEParams.Residual = true;
	CBaseLayer* postSE = addMNv3PostSE( postSEParams, *data, *preSE, *se );
	Sink( postSE, "sink" );
	Sink( dnn.GetLayer( "DownConv" ).Ptr(), "secondSink" );
	CDnnOptimizationReport report = OptimizeDnn( dnn );
	EXPECT_EQ( 1, report.MobileNetV3NonResidualBlocks );
	EXPECT_EQ( 0, report.MobileNetV3ResidualBlocks );
	EXPECT_EQ( 11, dnn.GetLayerCount() );
}

