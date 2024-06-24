/* Copyright @ 2024 ABBYY

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

struct CReferenceDnnTestParam {
	CDnn* net;
};

static void runDnn( int, void* params )
{
	CReferenceDnnTestParam* taskParams = static_cast<CReferenceDnnTestParam*>( params );
	taskParams->net->RunOnce();
}

static CDnn* createDnn( CRandom& random, float dropoutRate = 0.1f )
{
	CDnn* net = new CDnn( random, MathEngine() );

	CBaseLayer* layer = Source( *net, "in" );
	layer = FullyConnected( 50 )( "fc1", layer );
	layer = Dropout( dropoutRate )( "dp1", layer );
	layer = FullyConnected( 200 )( "fc2", layer );
	layer = Dropout( dropoutRate )( "dp2", layer );
	layer = FullyConnected( 10 )( "fc3", layer );
	layer = Sink( layer, "sink" );

	return net;
}

static void initializeBlob( CDnnBlob& blob, CRandom& random, double min, double max )
{
	for( int j = 0; j < blob.GetDataSize(); ++j ) {
		blob.GetData().SetValueAt( j, static_cast<float>( random.Uniform( min, max ) ) );
	}
}

static void getTestDnns( CPointerArray<CDnn>& dnns, CArray<CRandom>& randoms, bool useReference, const int& numOfThreads )
{
	CObjectArray<CSourceLayer> sourceLayers;
	sourceLayers.Add( nullptr, numOfThreads );

	dnns.SetBufferSize( numOfThreads );
	for( int i = 0; i < numOfThreads; ++i ) {
		if( i == 0 || !useReference ) {
			dnns.Add( createDnn( randoms[i] ) );
		} else {
			dnns.Add( dnns[i - 1]->CreateReferenceDnn() );
		}

		sourceLayers[i] = CheckCast<CSourceLayer>( dnns[i]->GetLayer( "in" ).Ptr() );
		CPtr<CDnnBlob> blob = CDnnBlob::CreateTensor( MathEngine(), CT_Float, { 1, 1, 1, 8, 20, 30, 10 } );
		sourceLayers[i]->SetBlob( blob );

		dnns[i]->RunOnce(); // reshaped
	}

	for( int i = 0; i < numOfThreads; ++i ) {
		initializeBlob( *sourceLayers[i]->GetBlob(), dnns[i]->Random(), /*min*/0., /*max*/1. );
	}
}

static void runMultithreadInference( CPointerArray<CDnn>& dnns, const int numOfThreads )
{
	CArray<CReferenceDnnTestParam> taskParams;
	for( int i = 0; i < numOfThreads; ++i ) {
		taskParams.Add( { dnns[i] } );
	}

	IThreadPool* pool = CreateThreadPool( numOfThreads );
	for( int i = 0; i < numOfThreads; ++i ) {
		pool->AddTask( i, runDnn, &( taskParams[i] ) );
	}

	pool->WaitAllTask();
	delete pool;
}

static void perfomanceTest( bool useReference, const int numOfThreads = 4 )
{
	CArray<CRandom> randoms;
	const int numOfOriginalDnns = useReference ? 1 : numOfThreads;
	for( int i = 0; i < numOfOriginalDnns; ++i ) {
		randoms.Add( CRandom( 0 ) );
	}

	CDnn* originDnn = nullptr;
	{
		CPointerArray<CDnn> dnns;
		getTestDnns( dnns, randoms, useReference, numOfThreads );

		IPerformanceCounters* counters( MathEngine().CreatePerformanceCounters() );

		counters->Synchronise();
		runMultithreadInference( dnns, numOfThreads );
		counters->Synchronise();

		GTEST_LOG_( INFO )
			<< '\n' << "Time: " << ( double( ( *counters )[0].Value ) / 1000000 ) << " ms."
			<< '\t' << "Peak.Mem: " << ( double( MathEngine().GetPeakMemoryUsage() ) / 1024 / 1024 ) << " MB \n";

		delete counters;
		originDnn = dnns.DetachAndReplaceAt( nullptr, 0 );
	} // delete references first
	delete originDnn;
}

} // namespace NeoMLTest

//------------------------------------------------------------------------------------------------

TEST( ReferenceDnnTest, ReferenceDnnInferenceTest )
{
	const auto met = MathEngine().GetType();
	if( met != MET_Cpu ) {
		GTEST_LOG_( INFO ) << "Skipped rest of test for MathEngine type=" << int( met ) << " because no implementation.\n";
		return;
	}

	CDnn* originDnn = nullptr;
	{
		CPointerArray<CDnn> dnns;
		CArray<CRandom> randoms = { CRandom( 0x123 ) };

		const int numOfThreads = 4;
		getTestDnns( dnns, randoms, /*useReference*/true, numOfThreads );
		runMultithreadInference( dnns, numOfThreads );

		originDnn = dnns.DetachAndReplaceAt( nullptr, 0 );
		CPtr<CDnnBlob> sourceBlob = static_cast<CSourceLayer*>( originDnn->GetLayer( "in" ).Ptr() )->GetBlob();
		CPtr<CDnnBlob> sinkBlob = static_cast<CSourceLayer*>( originDnn->GetLayer( "sink" ).Ptr() )->GetBlob();

		for( int i = 1; i < numOfThreads; ++i ) {
			EXPECT_TRUE( CompareBlobs( *sourceBlob,
				*( static_cast<CSourceLayer*>( dnns[i]->GetLayer( "in" ).Ptr() )->GetBlob() ) )
			);

			EXPECT_TRUE( CompareBlobs( *sinkBlob,
				*( static_cast<CSinkLayer*>( dnns[i]->GetLayer( "sink" ).Ptr() )->GetBlob() ) )
			);
		}
	} // delete references first
	delete originDnn;
}

TEST( ReferenceDnnTest, CDnnReferenceRegisterTest )
{
	const auto met = MathEngine().GetType();
	if( met != MET_Cpu ) {
		GTEST_LOG_( INFO ) << "Skipped rest of test for MathEngine type=" << int( met ) << " because no implementation.\n";
		return;
	}

	// Implement scenario - learn dnn, use multihtread inference, learn again
	const int numOfThreads = 4;
	const int interations = 10;

	// 1.Create and learn dnn
	CRandom random( 0x123 );
	CDnn* originDnn = createDnn( random );

	CPtr<CDnnBlob> sourceBlob = CDnnBlob::CreateTensor( MathEngine(), CT_Float, { 1, 1, 1, 8, 20, 30, 10 } );
	initializeBlob( *sourceBlob, random, /*min*/0., /*max*/1. );
	CheckCast<CSourceLayer>( originDnn->GetLayer( "in" ).Ptr() )->SetBlob( sourceBlob );

	CPtr<CDnnBlob> labelBlob = CDnnBlob::CreateTensor( MathEngine(), CT_Float, { 1, 1, 1, 1, 1, 1, 10 } );
	initializeBlob( *labelBlob, random, /*min*/0., /*max*/1. );

	CPtr<CSourceLayer> labels = Source( *originDnn, "labels" );
	labels->SetBlob( labelBlob );

	CPtr<CL1LossLayer> loss = L1Loss()( "loss", originDnn->GetLayer( "fc3" ).Ptr(), labels.Ptr() );

	CPtr<CDnnSimpleGradientSolver> solver = new CDnnSimpleGradientSolver( MathEngine() );
	originDnn->SetSolver( solver );

	for( int i = 0; i < interations; ++i ) {
		originDnn->RunAndLearnOnce();
	}

	// 2. Run mulithread inference
	originDnn->DeleteLayer( "labels" );
	originDnn->DeleteLayer( "loss" );

	{
		CPointerArray<CDnn> dnns;
		dnns.SetBufferSize( numOfThreads );

		dnns.Add( originDnn );
		for( int i = 1; i < numOfThreads; ++i ) {
			dnns.Add( originDnn->CreateReferenceDnn() );

			CPtr<CDnnBlob> blob = CDnnBlob::CreateTensor( MathEngine(), CT_Float, { 1, 1, 1, 8, 20, 30, 10 } );
			initializeBlob( *blob, random, /*min*/0., /*max*/1. );
			CheckCast<CSourceLayer>( dnns[i]->GetLayer( "in" ).Ptr() )->SetBlob( blob );
		}

		EXPECT_TRUE( !dnns[0]->IsLearningEnabled() );
		runMultithreadInference( dnns, numOfThreads );

		dnns.DetachAndReplaceAt( nullptr, 0 );
	} // delete references first

	// 3. Learn again
	originDnn->AddLayer( *labels );
	originDnn->AddLayer( *loss );

	EXPECT_TRUE( originDnn->IsLearningEnabled() );
	for( int i = 0; i < interations; ++i ) {
		originDnn->RunAndLearnOnce();
	}
	delete originDnn;
}

TEST( ReferenceDnnTest, DISABLED_PerfomanceReferenceDnnsThreads )
{
	const auto met = MathEngine().GetType();
	if( met != MET_Cpu ) {
		GTEST_LOG_( INFO ) << "Skipped rest of test for MathEngine type=" << int( met ) << " because no implementation.\n";
		return;
	}

	perfomanceTest( true );
}

TEST( ReferenceDnnTest, DISABLED_PerfomanceDnnsThreads )
{
	const auto met = MathEngine().GetType();
	if( met != MET_Cpu ) {
		GTEST_LOG_( INFO ) << "Skipped rest of test for MathEngine type=" << int( met ) << " because no implementation.\n";
		return;
	}

	perfomanceTest( false );
}
