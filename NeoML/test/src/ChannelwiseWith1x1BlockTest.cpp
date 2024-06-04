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

#include <TestFixture.h>

using namespace NeoML;
using namespace NeoMLTest;

namespace NeoMLTest {

class CChannelwiseWith1x1Composite : public CCompositeLayer {
public:
	CChannelwiseWith1x1Composite( IMathEngine& mathEngine, int stride, const CPtr<CDnnBlob>& channelwiseFilter,
		const CPtr<CDnnBlob>& channelwiseFreeTerm, TActivationFunction activation, float reluParam,
		const CPtr<CDnnBlob>& convFilter, const CPtr<CDnnBlob>& convFreeTerm, bool residuall );

	CPtr<CChannelwiseConvLayer> Channelwise;
	CPtr<CConvLayer> Conv;
};

CChannelwiseWith1x1Composite::CChannelwiseWith1x1Composite( IMathEngine& mathEngine, int stride,
		const CPtr<CDnnBlob>& channelwiseFilter, const CPtr<CDnnBlob>& channelwiseFreeTerm,
		TActivationFunction activation, float reluParam, const CPtr<CDnnBlob>& convFilter,
		const CPtr<CDnnBlob>& convFreeTerm, bool residual ) :
	CCompositeLayer( mathEngine, "ChannelwiseWith1x1Composite" )
{
	Channelwise = new CChannelwiseConvLayer( mathEngine );
	Channelwise->SetFilterCount( channelwiseFilter->GetChannelsCount() );
	Channelwise->SetFilterHeight( 3 );
	Channelwise->SetFilterWidth( 3 );
	Channelwise->SetFilterData( channelwiseFilter );
	Channelwise->SetPaddingHeight( 1 );
	Channelwise->SetPaddingWidth( 1 );
	Channelwise->SetStrideHeight( stride );
	Channelwise->SetStrideWidth( stride );
	if( channelwiseFreeTerm == nullptr ) {
		Channelwise->SetZeroFreeTerm( true );
	} else {
		Channelwise->SetZeroFreeTerm( false );
		Channelwise->SetFreeTermData( channelwiseFreeTerm );
	}
	SetInputMapping( *Channelwise );
	AddLayer( *Channelwise );

	CPtr<CBaseLayer> activationLayer;
	if( activation == AF_ReLU ) {
		CPtr<CReLULayer> relu = new CReLULayer( mathEngine );
		relu->SetUpperThreshold( reluParam );
		activationLayer = relu;
	} else {
		activationLayer = new CHSwishLayer( mathEngine );
	}
	activationLayer->Connect( *Channelwise );
	AddLayer( *activationLayer );

	Conv = new CConvLayer( mathEngine );
	Conv->SetFilterCount( convFilter->GetObjectCount() );
	Conv->SetFilterData( convFilter );
	if( convFreeTerm == nullptr ) {
		Conv->SetZeroFreeTerm( true );
	} else {
		Conv->SetZeroFreeTerm( false );
		Conv->SetFreeTermData( convFreeTerm );
	}
	Conv->Connect( *activationLayer );
	AddLayer( *Conv );

	if( !residual ) {
		SetOutputMapping( *Conv );
	} else {
		CPtr<CEltwiseSumLayer> sum = new CEltwiseSumLayer( mathEngine );
		AddLayer( *sum );
		SetInputMapping( *sum );
		sum->Connect( 1, *Conv );
		SetOutputMapping( *sum );
	}
}

static void channelwiseWith1x1TestImpl( unsigned int seed, int freeTermMask, TActivationFunction activation,
	float reluParam, int stride, bool residual, const std::initializer_list<int>& inputDims )
{
	auto createBlob = [] ( const std::initializer_list<int>& dims, CRandom& random ) -> CPtr<CDnnBlob> {
		CPtr<CDnnBlob> blob = CDnnBlob::CreateTensor( MathEngine(), CT_Float, dims );
		CREATE_FILL_FLOAT_ARRAY( data, -1, 1, blob->GetDataSize(), random );
		blob->CopyFrom( data.GetPtr() );
		return blob;
	};

	NeoAssert( stride == 1 || stride == 2 );
	NeoAssert( freeTermMask >= 0 && freeTermMask < 4 );
	NeoAssert( !residual || stride == 1 );

	CRandom random( seed );

	const int channelwiseFreeTermBit = 1 << 0;
	const int convFreeTermBit = 1 << 1;

	const int inputChannels = *( inputDims.begin() + 6 );
	const int outputChannels = residual ? inputChannels : 12;

	CPtr<CDnnBlob> channelwiseFilter = createBlob( { 1, 1, 1, 3, 3, 1, inputChannels }, random );
	CPtr<CDnnBlob> channelwiseFreeTerm;
	if( ( freeTermMask & channelwiseFreeTermBit ) != 0 ) {
		channelwiseFreeTerm = createBlob( { inputChannels }, random );
	}

	CPtr<CDnnBlob> convFilter = createBlob( { 1, outputChannels, 1, 1, 1, 1, inputChannels }, random );
	CPtr<CDnnBlob> convFreeTerm;
	if( ( freeTermMask & convFreeTermBit ) != 0 ) {
		convFreeTerm = createBlob( { outputChannels }, random );
	}

	CDnn dnn( random, MathEngine() );
	CPtr<CSourceLayer> data = AddLayer<CSourceLayer>( "Data", dnn );

	CPtr<CChannelwiseWith1x1Composite> expectedBlock = AddLayer<CChannelwiseWith1x1Composite>(
		new CChannelwiseWith1x1Composite( MathEngine(), stride, channelwiseFilter, channelwiseFreeTerm,
			activation, reluParam, convFilter, convFreeTerm, residual ), "expectedBlock", { data } );
	CPtr<CSinkLayer> expectedSink = AddLayer<CSinkLayer>( "expectedSink", { expectedBlock } );

	NeoAssert( activation == AF_ReLU || activation == AF_HSwish );
	CActivationDesc activationDesc( activation );
	if( activation == AF_ReLU ) {
		CReLULayer::CParam param;
		param.UpperThreshold = reluParam;
		activationDesc.SetParam( param );
	}
	CPtr<CChannelwiseWith1x1Layer> actualBlock = new CChannelwiseWith1x1Layer( MathEngine(), stride, channelwiseFilter,
		channelwiseFreeTerm, activationDesc, convFilter, convFreeTerm, residual );
	AddLayer( actualBlock, "actualBlock", { data } );
	CPtr<CSinkLayer> actualSink = AddLayer<CSinkLayer>( "actualSink", { actualBlock } );

	data->SetBlob( createBlob( inputDims, random ) );

	dnn.RunOnce();

	CPtr<CDnnBlob> expectedBlob = expectedSink->GetBlob();
	CPtr<CDnnBlob> actualBlob = actualSink->GetBlob();

	CDnnBlobBuffer<float> expected( *expectedBlob, TDnnBlobBufferAccess::Read );
	CDnnBlobBuffer<float> actual( *actualBlob, TDnnBlobBufferAccess::Read );

	EXPECT_EQ( expected.Size(), actual.Size() ) << "output size mismatch";
	for( int i = 0; i < expected.Size(); ++i ) {
		EXPECT_NEAR( expected[i], actual[i], 1e-3 ) << "at index " << i;
	}
}

} // namespace NeoMLTest

TEST( ChannelwiseWith1x1LayerTest, Run )
{
	const auto met = MathEngine().GetType();
	if(met != MET_Cpu && met != MET_Cuda) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		// InitRowwiseChWith1x1
		return;
	}

	// This test is allowed on GPU because of backward compatibility
	std::initializer_list<int> inputDims = { 1, 3, 1, 26, 31, 1, 8 };
	CRandom seedRandom( 0x654 );
	for( int ftMask = 0; ftMask < 4; ++ftMask ) {
		for( int stride = 1; stride < 3; ++stride ) {
			channelwiseWith1x1TestImpl( seedRandom.Next(), ftMask, AF_ReLU, 0, stride, false, inputDims );
			channelwiseWith1x1TestImpl( seedRandom.Next(), ftMask, AF_ReLU, 6.0f, stride, false, inputDims );
			channelwiseWith1x1TestImpl( seedRandom.Next(), ftMask, AF_HSwish, 0, stride, false, inputDims );
		}
		channelwiseWith1x1TestImpl( seedRandom.Next(), ftMask, AF_ReLU, 0, 1, true, inputDims );
		channelwiseWith1x1TestImpl( seedRandom.Next(), ftMask, AF_ReLU, 6.0f, 1, true, inputDims );
		channelwiseWith1x1TestImpl( seedRandom.Next(), ftMask, AF_HSwish, 0, 1, true, inputDims );
	}
}

TEST( ChannelwiseWith1x1LayerTest, CornerCases )
{
	const auto met = MathEngine().GetType();
	if(met != MET_Cpu && met != MET_Cuda) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		// InitRowwiseChWith1x1
		return;
	}

	// This test is allowed on GPU because of backward compatibility
	CRandom seedRandom( 0x654 );
	channelwiseWith1x1TestImpl( seedRandom.Next(), 0, AF_ReLU, 0, 1, true, { 1, 7, 1, 1, 3, 1, 3 } );
	channelwiseWith1x1TestImpl( seedRandom.Next(), 1, AF_HSwish, 1, 2, false, { 1, 7, 1, 2, 3, 1, 3 } );
	channelwiseWith1x1TestImpl( seedRandom.Next(), 2, AF_ReLU, 2, 2, false, { 1, 7, 1, 2, 1, 1, 3 } );
	channelwiseWith1x1TestImpl( seedRandom.Next(), 3, AF_HSwish, 3, 2, false, { 1, 7, 1, 3, 1, 1, 3 } );
	channelwiseWith1x1TestImpl( seedRandom.Next(), 0, AF_ReLU, 0, 1, false, { 1, 7, 1, 3, 1, 1, 3 } );
	channelwiseWith1x1TestImpl( seedRandom.Next(), 1, AF_HSwish, 1, 1, true, { 1, 7, 1, 3, 1, 1, 3 } );
	channelwiseWith1x1TestImpl( seedRandom.Next(), 2, AF_ReLU, 2, 1, true, { 1, 21, 1, 1, 3, 1, 1022 } );
	channelwiseWith1x1TestImpl( seedRandom.Next(), 3, AF_HSwish, 3, 2, false, { 1, 22, 1, 2, 3, 1, 1023 } );
	channelwiseWith1x1TestImpl( seedRandom.Next(), 0, AF_ReLU, 0, 2, false, { 1, 23, 1, 2, 1, 1, 1024 } );
	channelwiseWith1x1TestImpl( seedRandom.Next(), 1, AF_HSwish, 1, 2, false, { 1, 24, 1, 3, 1, 1, 1025 } );
	channelwiseWith1x1TestImpl( seedRandom.Next(), 2, AF_ReLU, 2, 1, false, { 1, 25, 1, 3, 1, 1, 1026 } );
	channelwiseWith1x1TestImpl( seedRandom.Next(), 3, AF_HSwish, 3, 1, true, { 1, 26, 1, 3, 1, 1, 1027 } );
}

TEST( ChannelwiseWith1x1OptimizerTest, SimpleNonResidual )
{
	CRandom random( 0x654 );
	CDnn dnn( random, MathEngine() );
	CSourceLayer* data = Source( dnn, "data" );
	CChannelwiseConvLayer* channelwiseConv = ChannelwiseConv( 32, CConvAxisParams( 3, 1, 2 ), CConvAxisParams( 3, 1, 2 ) )
		( "channewlseConv", data );
	CReLULayer* channelwiseReLU = Relu( 6.f )( "channelwiseReLU", channelwiseConv );
	CConvLayer* conv = Conv( 8, CConvAxisParams( 1 ), CConvAxisParams( 1 ) )( "conv", channelwiseReLU );
	Sink( conv, "sink" );
	CDnnOptimizationReport report = OptimizeDnn( dnn );
	EXPECT_EQ( 1, report.ChannelwiseWith1x1NonResidual );
	EXPECT_EQ( 0, report.ChannelwiseWith1x1Residual );
	EXPECT_EQ( 3, dnn.GetLayerCount() );
}

TEST( ChannelwiseWith1x1OptimizerTest, SimpleResidual )
{
	CRandom random( 0x654 );
	CDnn dnn( random, MathEngine() );
	CSourceLayer* data = Source( dnn, "data" );
	CChannelwiseConvLayer* channelwiseConv = ChannelwiseConv( 32, CConvAxisParams( 3, 1, 2 ), CConvAxisParams( 3, 1, 2 ) )
		( "channewlseConv", data );
	CHSwishLayer* channelwiseHSwish = HSwish()( "channelwiseHSwish", channelwiseConv );
	CConvLayer* conv = Conv( 8, CConvAxisParams( 1 ), CConvAxisParams( 1 ) )( "conv", channelwiseHSwish );
	CEltwiseSumLayer* residual = Sum()( "residual", data, conv );
	Sink( residual, "sink" );
	CDnnOptimizationReport report = OptimizeDnn( dnn );
	EXPECT_EQ( 0, report.ChannelwiseWith1x1NonResidual );
	EXPECT_EQ( 1, report.ChannelwiseWith1x1Residual );
	EXPECT_EQ( 3, dnn.GetLayerCount() );
}

TEST( ChannelwiseWith1x1OptimizerTest, ResidualResidual )
{
	CRandom random( 0x654 );
	CDnn dnn( random, MathEngine() );
	CSourceLayer* data = Source( dnn, "data" );
	CChannelwiseConvLayer* channelwiseConv = ChannelwiseConv( 32, CConvAxisParams( 3, 1, 2 ), CConvAxisParams( 3, 1, 2 ) )
		( "channewlseConv", data );
	CReLULayer* channelwiseReLU = Relu( 6.f )( "channelwiseReLU", channelwiseConv );
	CConvLayer* conv = Conv( 8, CConvAxisParams( 1 ), CConvAxisParams( 1 ) )( "conv", channelwiseReLU );
	CEltwiseSumLayer* residual = Sum()( "residual", data, conv );
	CEltwiseSumLayer* doubleResidual = Sum()( "doubleResidual", data, residual );
	Sink( doubleResidual, "sink" );
	CDnnOptimizationReport report = OptimizeDnn( dnn );
	EXPECT_EQ( 0, report.ChannelwiseWith1x1NonResidual );
	EXPECT_EQ( 1, report.ChannelwiseWith1x1Residual );
	EXPECT_EQ( 4, dnn.GetLayerCount() );
}

TEST( ChannelwiseWith1x1OptimizerTest, NeighboringResiduals )
{
	CRandom random( 0x654 );
	CDnn dnn( random, MathEngine() );
	CSourceLayer* data = Source( dnn, "data" );
	CChannelwiseConvLayer* channelwiseConv = ChannelwiseConv( 32, CConvAxisParams( 3, 1, 2 ), CConvAxisParams( 3, 1, 2 ) )
		( "channewlseConv", data );
	CReLULayer* channelwiseReLU = Relu( 6.f )( "channelwiseReLU", channelwiseConv );
	CConvLayer* conv = Conv( 8, CConvAxisParams( 1 ), CConvAxisParams( 1 ) )( "conv", channelwiseReLU );
	CEltwiseSumLayer* residual = Sum()( "residual", data, conv );
	Sink( residual, "sink" );
	CEltwiseSumLayer* secondResidual = Sum()( "secondResidual", data, conv );
	Sink( secondResidual, "secondSink" );
	CDnnOptimizationReport report = OptimizeDnn( dnn );
	EXPECT_EQ( 1, report.ChannelwiseWith1x1NonResidual );
	EXPECT_EQ( 0, report.ChannelwiseWith1x1Residual );
	EXPECT_EQ( 6, dnn.GetLayerCount() );
}

TEST( ChannelwiseWith1x1OptimizerTest, SinkFromTheMiddle )
{
	CRandom random( 0x654 );
	CDnn dnn( random, MathEngine() );
	CSourceLayer* data = Source( dnn, "data" );
	CChannelwiseConvLayer* channelwiseConv = ChannelwiseConv( 32, CConvAxisParams( 3, 1, 2 ), CConvAxisParams( 3, 1, 2 ) )
		( "channewlseConv", data );
	CHSwishLayer* channelwiseHSwish = HSwish()( "channelwiseHSwish", channelwiseConv );
	Sink( channelwiseConv, "activationSink" );
	CConvLayer* conv = Conv( 8, CConvAxisParams( 1 ), CConvAxisParams( 1 ) )( "conv", channelwiseHSwish );
	CEltwiseSumLayer* residual = Sum()( "residual", data, conv );
	Sink( residual, "sink" );
	CDnnOptimizationReport report = OptimizeDnn( dnn );
	EXPECT_EQ( 0, report.ChannelwiseWith1x1NonResidual );
	EXPECT_EQ( 0, report.ChannelwiseWith1x1Residual );
}

TEST( ChannelwiseWith1x1OptimizerTest, SinkDisablesResidual )
{
	CRandom random( 0x654 );
	CDnn dnn( random, MathEngine() );
	CSourceLayer* data = Source( dnn, "data" );
	CChannelwiseConvLayer* channelwiseConv = ChannelwiseConv( 32, CConvAxisParams( 3, 1, 2 ), CConvAxisParams( 3, 1, 2 ) )
		( "channewlseConv", data );
	CHSwishLayer* channelwiseHSwish = HSwish()( "channelwiseHSwish", channelwiseConv );
	CConvLayer* conv = Conv( 8, CConvAxisParams( 1 ), CConvAxisParams( 1 ) )( "conv", channelwiseHSwish );
	Sink( conv, "convSink" );
	CEltwiseSumLayer* residual = Sum()( "residual", data, conv );
	Sink( residual, "sink" );
	CDnnOptimizationReport report = OptimizeDnn( dnn );
	EXPECT_EQ( 1, report.ChannelwiseWith1x1NonResidual );
	EXPECT_EQ( 0, report.ChannelwiseWith1x1Residual );
	EXPECT_EQ( 5, dnn.GetLayerCount() );
}
