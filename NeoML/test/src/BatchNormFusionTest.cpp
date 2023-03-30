/* Copyright © 2017-2023 ABBYY Production LLC

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

CPtr<CDnnBlob> bnFusionData( CRandom& random )
{
	const int batch = 2;
	const int height = 15;
	const int width = 17;
	const int depth = 5;
	const int channels = 3;

	CREATE_FILL_FLOAT_ARRAY( dataArr, 0.f, 1.f, batch * height * width * depth * channels, random );
	CPtr<CDnnBlob> dataBlob = CDnnBlob::Create3DImageBlob( MathEngine(), CT_Float, 1, batch,
		height, width, depth, channels );
	dataBlob->CopyFrom( dataArr.GetPtr() );
	return dataBlob;
}

void checkBnFusion( CDnn& dnn, CSinkLayer* sink, int expectedFusions )
{
	dnn.RunOnce();
	CPtr<CDnnBlob> expected = sink->GetBlob()->GetCopy();

	CDnnOptimizationReport report = OptimizeDnn( dnn );
	EXPECT_EQ( expectedFusions, report.FusedBatchNormalizations );
	dnn.RunOnce();
	CPtr<CDnnBlob> actual = sink->GetBlob()->GetCopy();
	EXPECT_TRUE( CompareBlobs( *expected, *actual ) );
}

TEST( BatchNormFusionTest, SimpleFusion )
{
	CRandom random( 0x654 );
	CDnn dnn( random, MathEngine() );
	CSourceLayer* data = Source( dnn, "source" );
	CBaseLayer* lastLayer = Conv3d( 16, CConvAxisParams(), CConvAxisParams(), CConvAxisParams() )( "conv3d", data );
	lastLayer = BatchNormalization( true )( "conv3dBn", lastLayer );
	lastLayer = Conv( 2, CConvAxisParams(), CConvAxisParams() )( "conv", lastLayer );
	lastLayer = BatchNormalization( true )( "convBn", lastLayer );
	lastLayer = TransposedConv( 4, CConvAxisParams(), CConvAxisParams() )( "tConv", lastLayer );
	lastLayer = BatchNormalization( true )( "tConvBn", lastLayer );
	lastLayer = ChannelwiseConv( 6, CConvAxisParams(), CConvAxisParams() )( "chConv", lastLayer );
	lastLayer = BatchNormalization( true )( "chConvBn", lastLayer );
	lastLayer = FullyConnected( 8 )( "fc", lastLayer );
	lastLayer = BatchNormalization( true )( "fcBn", lastLayer );
	CSinkLayer* sink = Sink( lastLayer, "sink" );

	data->SetBlob( bnFusionData( random ) );
	checkBnFusion( dnn, sink, 5 );
}

TEST( BatchNormFusionTest, DoubleFusion )
{
	CRandom random( 0x654 );
	CDnn dnn( random, MathEngine() );
	CSourceLayer* data = Source( dnn, "source" );

	CBaseLayer* lastLayer = FullyConnected( 8 )( "fc", data );
	lastLayer = BatchNormalization( true )( "firstBn", lastLayer );
	lastLayer = BatchNormalization( true )( "secondBn", lastLayer );
	CSinkLayer* sink = Sink( lastLayer, "sink" );

	data->SetBlob( bnFusionData( random ) );
	checkBnFusion( dnn, sink, 2 );
}

TEST( BatchNormFusionTest, ImpossibleFusion )
{
	CRandom random( 0x654 );
	CDnn dnn( random, MathEngine() );
	CSourceLayer* data = Source( dnn, "source" );
	CFullyConnectedLayer* fc = FullyConnected( 8 )( "fc", data );
	
	// Connect 2 bns to fully connected
	// None of them should be fused into fc (otherwise dnns won't be equivalent)

	CBaseLayer* lastLayer = BatchNormalization( true )( "firstBn", fc );
	CSinkLayer* sink = Sink( lastLayer, "sink" );

	lastLayer = BatchNormalization( true )( "secondBn", fc );
	( void ) Sink( lastLayer, "secondSink" );

	data->SetBlob( bnFusionData( random ) );
	checkBnFusion( dnn, sink, 0 );
}
