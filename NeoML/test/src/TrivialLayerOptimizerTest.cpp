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

using namespace NeoML;
using namespace NeoMLTest;

TEST( TrivialLayerOptimizerTest, Linear )
{
	CRandom random( 0x9845 );
	CDnn dnn( random, MathEngine() );
	CSourceLayer* source = Source( dnn, "source" );
	CLinearLayer* linearToRemove0 = Linear( 1.f, 0.f )( "linearToRemove0", source );
	CLinearLayer* linearToSave0 = Linear( 2.f, 0.f )( "linearToSave0", linearToRemove0 );
	CLinearLayer* linearToRemove1 = Linear( 1.f, 0.f )( "linearToRemove1", linearToSave0 );
	CLinearLayer* linearToSave1 = Linear( 1.f, 1.f )( "linearToSave1", linearToRemove1 );
	CLinearLayer* linearToRemove2 = Linear( 1.0f, 0 )( "linearToRemove2", linearToSave1 );
	CLinearLayer* linearToRemove3 = Linear( 1.0f, 0 )( "linearToRemove3", linearToRemove2 );
	( void ) Sink( linearToRemove3, "sink" );

	CDnnOptimizationReport report = OptimizeDnn( dnn );

	EXPECT_EQ( 4, report.RemovedTrivialLayers );

	EXPECT_EQ( 4, dnn.GetLayerCount() );
	EXPECT_TRUE( dnn.HasLayer( "source" ) );
	EXPECT_TRUE( dnn.HasLayer( "linearToSave0" ) );
	EXPECT_TRUE( dnn.HasLayer( "linearToSave1" ) );
	EXPECT_TRUE( dnn.HasLayer( "sink" ) );

	EXPECT_FALSE( dnn.HasLayer( "linearToRemove0" ) );
	EXPECT_FALSE( dnn.HasLayer( "linearToRemove1" ) );
	EXPECT_FALSE( dnn.HasLayer( "linearToRemove2" ) );
	EXPECT_FALSE( dnn.HasLayer( "linearToRemove3" ) );
}

TEST( TrivialLayerOptimizerTest, Dropout )
{
	CRandom random( 0x654 );
	CDnn dnn( random, MathEngine() );
	CSourceLayer* source = Source( dnn, "source" );
	CDropoutLayer* dropout0 = Dropout( 0.1f )( "dropout0", source );
	CFullyConnectedLayer* fc0 = FullyConnected( 10 )( "fc0", dropout0 );
	CDropoutLayer* dropout1 = Dropout( 0.2f )( "dropout1", fc0 );
	CDropoutLayer* dropout2 = Dropout( 0.5f )( "dropout2", dropout1 );
	CDropoutLayer* dropout3 = Dropout( 0.001f )( "dropout3", dropout2 );
	( void ) Sink( dropout3, "sink" );

	CDnnOptimizationReport report = OptimizeDnn( dnn );

	EXPECT_EQ( 4, report.RemovedTrivialLayers );

	EXPECT_EQ( 3, dnn.GetLayerCount() );
	EXPECT_TRUE( dnn.HasLayer( "source" ) );
	EXPECT_TRUE( dnn.HasLayer( "fc0" ) );
	EXPECT_TRUE( dnn.HasLayer( "sink" ) );

	EXPECT_FALSE( dnn.HasLayer( "dropout0" ) );
	EXPECT_FALSE( dnn.HasLayer( "dropout1" ) );
	EXPECT_FALSE( dnn.HasLayer( "dropout2" ) );
	EXPECT_FALSE( dnn.HasLayer( "dropout3" ) );
}

TEST( TrivialLayerOptimizerTest, None )
{
	CRandom random( 0x5346 );
	CDnn dnn( random, MathEngine() );
	CSourceLayer* source = Source( dnn, "source" );
	CFullyConnectedLayer* fc = FullyConnected( 10 )( "fc", source );
	CLinearLayer* linear0 = Linear( 2.f, 0.f )( "linear0", fc );
	CConvLayer* conv = Conv( 12, CConvAxisParams( 3 ), CConvAxisParams( 4 ), true )( "conv", linear0 );
	CLinearLayer* linear1 = Linear( 1.f, 1.f )( "linear1", conv );
	( void ) Sink( linear1, "sink" );

	EXPECT_EQ( 6, dnn.GetLayerCount() );

	CDnnOptimizationReport report = OptimizeDnn( dnn );

	EXPECT_EQ( 0, report.RemovedTrivialLayers );
	EXPECT_EQ( 6, dnn.GetLayerCount() );
}

TEST( TrivialLayerOptimizerTest, All )
{
	CRandom random( 0x654 );
	CDnn dnn( random, MathEngine() );
	CSourceLayer* source = Source( dnn, "source" );
	CDropoutLayer* dropout0 = Dropout( 0.1f )( "dropout0", source );
	CLinearLayer* linear0 = Linear( 1.f, 0.f )( "linear0", dropout0 );
	CDropoutLayer* dropout1 = Dropout( 0.2f )( "dropout1", linear0 );
	CDropoutLayer* dropout2 = Dropout( 0.5f )( "dropout2", dropout1 );
	CDropoutLayer* dropout3 = Dropout( 0.001f )( "dropout3", dropout2 );
	CLinearLayer* linear1 = Linear( 1.f, 0.f )( "linear1", dropout3 );
	CLinearLayer* linear2 = Linear( 1.f, 0.f )( "linear2", linear1 );
	( void ) Sink( linear2, "sink" );

	CDnnOptimizationReport report = OptimizeDnn( dnn );

	EXPECT_EQ( 7, report.RemovedTrivialLayers );
	EXPECT_EQ( 2, dnn.GetLayerCount() );
	EXPECT_TRUE( dnn.HasLayer( "source" ) );
	EXPECT_TRUE( dnn.HasLayer( "sink" ) );
}
