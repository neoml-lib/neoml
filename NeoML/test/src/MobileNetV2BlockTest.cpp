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

namespace NeoMLTest {

class CMobileNetV2Composite : public CCompositeLayer {
public:
	CMobileNetV2Composite( IMathEngine& mathEngine, CPtr<CDnnBlob> expandFilter, CPtr<CDnnBlob> expandFreeTerm,
		float expandReLUThreshold, CPtr<CDnnBlob> channelwiseFilter, CPtr<CDnnBlob> channelwiseFreeTerm,
		float channelwiseReLUThreshold, CPtr<CDnnBlob> downFilter, CPtr<CDnnBlob> downFreeTerm,
		int stride, bool residual );

	CPtr<CConvLayer> ExpandConv;
	CPtr<CChannelwiseConvLayer> Channelwise;
	CPtr<CConvLayer> DownConv;
};

CMobileNetV2Composite::CMobileNetV2Composite( IMathEngine& mathEngine, CPtr<CDnnBlob> expandFilter,
		CPtr<CDnnBlob> expandFreeTerm, float expandReLUThreshold, CPtr<CDnnBlob> channelwiseFilter,
		CPtr<CDnnBlob> channelwiseFreeTerm, float channelwiseReLUThreshold, CPtr<CDnnBlob> downFilter,
		CPtr<CDnnBlob> downFreeTerm, int stride, bool residual ) :
	CCompositeLayer( mathEngine, "MobileNetV2Composite" )
{
	ExpandConv = new CConvLayer( mathEngine );
	ExpandConv->SetName( "ExpandConv" );
	ExpandConv->SetFilterCount( expandFilter->GetObjectCount() );
	ExpandConv->SetFilterData( expandFilter );
	if( expandFreeTerm == nullptr ) {
		ExpandConv->SetZeroFreeTerm( true );
	} else {
		ExpandConv->SetZeroFreeTerm( false );
		ExpandConv->SetFreeTermData( expandFreeTerm );
	}
	AddLayer( *ExpandConv );
	SetInputMapping( *ExpandConv );

	CPtr<CReLULayer> expandReLU = new CReLULayer( mathEngine );
	expandReLU->SetName( "expandReLU" );
	expandReLU->SetUpperThreshold( expandReLUThreshold );
	expandReLU->Connect( *ExpandConv );
	AddLayer( *expandReLU );

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
	Channelwise->Connect( *expandReLU );
	AddLayer( *Channelwise );

	CPtr<CReLULayer> channelwiseReLU = new CReLULayer( mathEngine );
	channelwiseReLU->SetName( "channelwiseReLU" );
	channelwiseReLU->SetUpperThreshold( channelwiseReLUThreshold );
	channelwiseReLU->Connect( *Channelwise );
	AddLayer( *channelwiseReLU );

	DownConv = new CConvLayer( mathEngine );
	DownConv->SetName( "DownConv" );
	DownConv->SetFilterCount( downFilter->GetObjectCount() );
	DownConv->SetFilterData( downFilter );
	if( downFreeTerm == nullptr ) {
		DownConv->SetZeroFreeTerm( true );
	} else {
		DownConv->SetZeroFreeTerm( false );
		DownConv->SetFreeTermData( downFreeTerm );
	}
	DownConv->Connect( *channelwiseReLU );
	AddLayer( *DownConv );

	if( !residual ) {
		SetOutputMapping( *DownConv );
	} else {
		CPtr<CEltwiseSumLayer> sum = new CEltwiseSumLayer( mathEngine );
		AddLayer( *sum );
		SetInputMapping( *sum );
		sum->Connect( 1, *DownConv );
		SetOutputMapping( *sum );
	}
}

} // namespace NeoMLTest

using namespace NeoML;
using namespace NeoMLTest;

static void mobileNetV2BlockTestImpl( unsigned int seed, int freeTermMask, float expandReLUThreshold,
	float channelwiseReLUThreshold, int stride, bool residual )
{
	auto createBlob = [] ( const std::initializer_list<int>& dims, CRandom& random ) -> CPtr<CDnnBlob> {
		CPtr<CDnnBlob> blob = CDnnBlob::CreateTensor( MathEngine(), CT_Float, dims );
		CREATE_FILL_FLOAT_ARRAY( data, -1, 1, blob->GetDataSize(), random );
		blob->CopyFrom( data.GetPtr() );
		return blob;
	};

	NeoAssert( stride == 1 || stride == 2 );
	NeoAssert( freeTermMask >= 0 && freeTermMask < 8 );
	NeoAssert( !residual || stride == 1 );

	CRandom random( seed );

	const int expandFreeTermBit = 1 << 0;
	const int channelwiseFreeTermBit = 1 << 1;
	const int downFreeTermBit = 1 << 2;

	const int batch = 3;
	const int inputChannels = 8;
	const int outputChannels = residual ? inputChannels : 12;
	const int expandedChannels = 16;
	const int imageHeight = 26;
	const int imageWidth = 31;

	CPtr<CDnnBlob> expandFilter = createBlob( { 1, expandedChannels, 1, 1, 1, 1, inputChannels }, random );
	CPtr<CDnnBlob> expandFreeTerm;
	if( ( freeTermMask & expandFreeTermBit ) != 0 ) {
		expandFreeTerm = createBlob( { expandedChannels }, random );
	}

	CPtr<CDnnBlob> channelwiseFilter = createBlob( { 1, 1, 1, 3, 3, 1, expandedChannels }, random );
	CPtr<CDnnBlob> channelwiseFreeTerm;
	if( ( freeTermMask & channelwiseFreeTermBit ) != 0 ) {
		channelwiseFreeTerm = createBlob( { expandedChannels }, random );
	}

	CPtr<CDnnBlob> downFilter = createBlob( { 1, outputChannels, 1, 1, 1, 1, expandedChannels }, random );
	CPtr<CDnnBlob> downFreeTerm;
	if( ( freeTermMask & downFreeTermBit ) != 0 ) {
		downFreeTerm = createBlob( { outputChannels }, random );
	}

	CDnn dnn( random, MathEngine() );
	CPtr<CSourceLayer> data = AddLayer<CSourceLayer>( "Data", dnn );

	CPtr<CMobileNetV2Composite> expectedBlock = AddLayer<CMobileNetV2Composite>( new CMobileNetV2Composite( MathEngine(),
		expandFilter, expandFreeTerm, expandReLUThreshold, channelwiseFilter, channelwiseFreeTerm,
		channelwiseReLUThreshold, downFilter, downFreeTerm, stride, residual ), "expectedBlock", { data } );
	CPtr<CSinkLayer> expectedSink = AddLayer<CSinkLayer>( "expectedSink", { expectedBlock } );

	CPtr<CMobileNetV2BlockLayer> actualBlock = new CMobileNetV2BlockLayer( MathEngine(), expandFilter, expandFreeTerm,
		CActivationDesc( AF_ReLU, CReLULayer::CParam{ expandReLUThreshold } ), stride, channelwiseFilter,
		channelwiseFreeTerm, CActivationDesc( AF_ReLU, CReLULayer::CParam{ channelwiseReLUThreshold } ),
		downFilter, downFreeTerm, residual );
	AddLayer( actualBlock, "actualBlock", { data } );
	CPtr<CSinkLayer> actualSink = AddLayer<CSinkLayer>( "actualSink", { actualBlock } );

	data->SetBlob( createBlob( { 1, batch, 1, imageHeight, imageWidth, 1, inputChannels }, random ) );
	
	dnn.RunOnce();

	CPtr<CDnnBlob> expectedBlob = expectedSink->GetBlob();
	CPtr<CDnnBlob> actualBlob = actualSink->GetBlob();

	CDnnBlobBuffer<float> expected( *expectedBlob, TDnnBlobBufferAccess::Read );
	CDnnBlobBuffer<float> actual( *actualBlob, TDnnBlobBufferAccess::Read );

	ASSERT_EQ( expected.Size(), actual.Size() ) << "output size mismatch";
	for( int i = 0; i < expected.Size(); ++i ) {
		ASSERT_NEAR( expected[i], actual[i], 1e-3 ) << "at index " << i;
	}
}

TEST( MobileNetV2BlockLayerTest, Run )
{
	CRandom seedRandom( 0x654 );
	for( int ftMask = 0; ftMask < 8; ++ftMask ) {
		for( float expandReLU : { 0.f, 6.f } ) {
			for( float channelwiseReLU : { 0.f, 1.f } ) {
				mobileNetV2BlockTestImpl( seedRandom.Next(), ftMask, expandReLU, channelwiseReLU, 1, false );
				mobileNetV2BlockTestImpl( seedRandom.Next(), ftMask, expandReLU, channelwiseReLU, 2, false );
				mobileNetV2BlockTestImpl( seedRandom.Next(), ftMask, expandReLU, channelwiseReLU, 1, true );
			}
		}
	}
}

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
