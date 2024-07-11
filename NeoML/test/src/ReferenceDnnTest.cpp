/* Copyright © 2024 ABBYY

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

#include <functional>
#include <TestFixture.h>

using namespace NeoML;
using namespace NeoMLTest;

namespace NeoMLTest {

static constexpr int iterationsRunOnce = 10;

struct CReferenceDnnTestParam final {
	CReferenceDnnTestParam( CDnnReferenceRegister& reg, CDnnBlob& in, CDnnBlob& out,
			bool useReference, bool checkOutput = true ) :
		ReferenceDnnRegister( reg ),
		Input( in ),
		Expected( out ),
		UseReference( useReference ),
		CheckOutput( checkOutput )
	{}

	CDnnReferenceRegister& ReferenceDnnRegister;
	CDnnBlob& Input;
	CDnnBlob& Expected;
	const bool UseReference;
	const bool CheckOutput;
};

static void runDnn( int, void* params )
{
	CDnn& dnn = *static_cast<CDnn*>( params );

	for( int i = 0; i < iterationsRunOnce; ++i ) {
		dnn.RunOnce();
	}

	CPtr<CDnnBlob> input = CheckCast<CSourceLayer>( dnn.GetLayer( "in" ).Ptr() )->GetBlob();
	// for reference dnns only
	if( !dnn.IsLearningEnabled() ) {
		const CDnn& dnnConst = dnn;
		EXPECT_NO_THROW( dnn.RequestReshape() );
		EXPECT_NO_THROW( dnn.RequestReshape( true ); );
		EXPECT_NO_THROW( dnn.RunAndBackwardOnce(); );
		// learning
		EXPECT_NO_THROW( dnn.DisableLearning(); );
		NEOML_EXPECT_THROW( dnn.EnableLearning() );
		bool isLearningEnabled;
		EXPECT_NO_THROW( isLearningEnabled = dnn.IsLearningEnabled(); );
		EXPECT_EQ( isLearningEnabled, false );
		EXPECT_NO_THROW( dnn.RunAndLearnOnce() );
		// other
		EXPECT_NO_THROW( dnn.GetLog() );
		EXPECT_NO_THROW( dnn.SetLog( nullptr ); );
		EXPECT_NO_THROW( dnn.GetLogFrequency(); );
		EXPECT_NO_THROW( dnn.SetLogFrequency( 0 ); );
		EXPECT_NO_THROW( dnn.IsLogging(); );
		EXPECT_NO_THROW( dnn.GetLayerCount(); );
		EXPECT_NO_THROW( dnn.HasLayer( "fc2" ); );
		CArray<const char*> layerList;
		EXPECT_NO_THROW( dnn.GetLayerList( layerList ); );
		CPtr<CBaseLayer> layer;
		CPtr<const CBaseLayer> layerConst;
		NEOML_EXPECT_THROW( layer = dnn.GetLayer( "fc2" ) );
		EXPECT_EQ( layer, nullptr );
		EXPECT_NO_THROW( layerConst = dnnConst.GetLayer( "fc2" ); );
		EXPECT_NE( layerConst, nullptr );
		CArray<CString> path{ "fc2" };
		NEOML_EXPECT_THROW( layer = dnn.GetLayer( path ) );
		EXPECT_EQ( layer, nullptr );
		EXPECT_NO_THROW( layerConst = dnnConst.GetLayer( path ); );
		EXPECT_NE( layerConst, nullptr );
		NEOML_EXPECT_THROW( dnn.DeleteLayer( const_cast<CBaseLayer&>( *layerConst ) ) );
		NEOML_EXPECT_THROW( dnn.DeleteLayer( "fc2" ) );
		NEOML_EXPECT_THROW( dnn.DeleteAllLayers() );
		NEOML_EXPECT_THROW( dnn.AddLayer( const_cast<CBaseLayer&>( *layerConst ) ) );
		EXPECT_NO_THROW( dnn.CleanUp(); );
		EXPECT_NO_THROW( dnn.CleanUp( true ); );
		EXPECT_NO_THROW( dnn.GetMaxSequenceLength(); );
		EXPECT_NO_THROW( dnn.GetCurrentSequencePos(); );
		EXPECT_NO_THROW( dnn.IsReverseSequense(); );
		EXPECT_NO_THROW( dnn.IsFirstSequencePos(); );
		EXPECT_NO_THROW( dnn.IsLastSequencePos(); );
		EXPECT_NO_THROW( dnn.IsRecurrentMode(); );
		EXPECT_NO_THROW( dnn.IsBackwardPerformed(); );
		EXPECT_NO_THROW( dnn.RestartSequence(); );
		EXPECT_NO_THROW( dnn.EnableProfile( false ); );
		bool autoRestartMode;
		EXPECT_NO_THROW( autoRestartMode = dnn.GetAutoRestartMode(); );
		EXPECT_NO_THROW( dnn.SetAutoRestartMode( autoRestartMode ); );
		EXPECT_NO_THROW( dnn.ForceRebuild(); );
		EXPECT_NO_THROW( dnn.IsRebuildRequested(); );
		EXPECT_NO_THROW( dnn.Random(); );
		EXPECT_NO_THROW( dnn.GetMathEngine(); );
		const CDnnSolver* solverConst;
		EXPECT_NO_THROW( solverConst = dnnConst.GetSolver(); );
		EXPECT_NE( solverConst, nullptr );
		CDnnSolver* solver;
		EXPECT_NO_THROW( solver = dnn.GetSolver(); );
		EXPECT_NE( solver, nullptr );
		EXPECT_NO_THROW( dnn.SetSolver( solver ); );
		CPtr<CDnnInitializer> init;
		EXPECT_NO_THROW( init = dnn.GetInitializer(); );
		EXPECT_NE( init, nullptr );
		EXPECT_NO_THROW( dnn.SetInitializer( init ); );
		NEOML_EXPECT_THROW( dnn.FilterLayersParams( 0.1 ) );
		NEOML_EXPECT_THROW( dnn.FilterLayersParams( layerList, 0.1 ) );
		CMemoryFile file;
		CArchive archive( &file, CArchive::store );
		NEOML_EXPECT_THROW( dnn.Serialize( archive ) );
		NEOML_EXPECT_THROW( dnn.SerializeCheckpoint( archive ) );
	}
	// keep the result
	CheckCast<CSourceLayer>( dnn.GetLayer( "in" ).Ptr() )->SetBlob( input );
	dnn.RunOnce();
}

static void createDnn( CDnn& dnn, bool learn = false, float dropoutRate = 0.1f )
{
	CBaseLayer* layer = Source( dnn, "in" );
	layer = FullyConnected( 50 )( "fc1", layer );
	layer = Dropout( dropoutRate )( "dp1", layer );
	layer = FullyConnected( 200 )( "fc2", layer );
	layer = Dropout( dropoutRate )( "dp2", layer );
	layer = FullyConnected( 10 )( "fc3", layer );
	( void ) Sink( layer, "sink" );

	if( learn ) {
		CBaseLayer* labels = Source( dnn, "labels" );
		layer = L1Loss()( "loss", layer, labels );

		CPtr<CDnnSimpleGradientSolver> solver = new CDnnSimpleGradientSolver( dnn.GetMathEngine() );
		dnn.SetSolver( solver );
	}
}

static CPtr<CDnnBlob> getInitedBlob( IMathEngine& mathEngine, CRandom& rand, std::initializer_list<int> desc,
	double min = 0., double max = 1. )
{
	CPtr<CDnnBlob> blob = CDnnBlob::CreateTensor( mathEngine, CT_Float, desc );
	for( int j = 0; j < blob->GetDataSize(); ++j ) {
		blob->GetData().SetValueAt( j, static_cast<float>( rand.Uniform( min, max ) ) );
	}
	return blob;
}

static void setInputDnn( CDnn& dnn, CDnnBlob& blob, CDnnBlob* labelBlob = nullptr, bool reshape = false )
{
	CheckCast<CSourceLayer>( dnn.GetLayer( "in" ).Ptr() )->SetBlob( &blob );

	if( labelBlob ) {
		CheckCast<CSourceLayer>( dnn.GetLayer( "labels" ).Ptr() )->SetBlob( labelBlob );
	}

	if( reshape ) {
		dnn.RunOnce(); // reshaped
	}
}

static void learnDnn( CDnn& dnn, int interations = 10 )
{
	for( int i = 0; i < interations; ++i ) {
		dnn.RunAndLearnOnce();
	}

	dnn.DeleteLayer( "labels" );
	dnn.DeleteLayer( "loss" );
}

static CDnn* copyDnn( const CDnn& oldDnn, CRandom& random )
{
	CDnn* dnn = new CDnn( random, oldDnn.GetMathEngine() );

	CArray<const char*> layersList;
	oldDnn.GetLayerList( layersList );

	CMemoryFile file;
	for( const char* layerName : layersList ) {
		file.SeekToBegin();
		{
			CArchive archive( &file, CArchive::store );
			const CPtr<const CBaseLayer> layerConst = oldDnn.GetLayer( layerName );
			// This hack is need, because Serialize() is not const, but actually doesn't change anything while storing
			CPtr<CBaseLayer> layer = const_cast<CBaseLayer*>( layerConst.Ptr() );
			SerializeLayer( archive, oldDnn.GetMathEngine(), layer );
		}
		file.SeekToBegin();
		CPtr<CBaseLayer> copyLayer;
		{
			CArchive archive( &file, CArchive::load );
			SerializeLayer( archive, oldDnn.GetMathEngine(), copyLayer );
		}
		dnn->AddLayer( *copyLayer );
	}
	return dnn;
}

static void runDnnCreation( int thread, void* arg )
{
	CReferenceDnnTestParam& params = *static_cast<CReferenceDnnTestParam*>( arg );

	for( int i = 0; i < iterationsRunOnce; ++i ) {
		CPtrOwner<CRandom> random;
		CPtrOwner<CDnn> dnn;

		if( params.UseReference ) {
			dnn = params.ReferenceDnnRegister.CreateReferenceDnn();
		} else {
			CDnn& oldDnn = *reinterpret_cast<CDnn*>( &params.ReferenceDnnRegister );
			random = new CRandom( oldDnn.Random() );
			dnn = copyDnn( oldDnn, *random );
		}
		setInputDnn( *dnn, params.Input );

		for( int i = 0; i < 2; ++i ) {
			dnn->RunOnce();
		}
	}
}

static CDnnReferenceRegister* getTestDnns( IMathEngine& mathEngine, CPointerArray<CDnn>& dnns, CArray<CRandom>& randoms,
	bool useReference, int numOfThreads )
{
	CRandom random( 0 );
	CPtr<CDnnBlob> blob = getInitedBlob( mathEngine, random, { 1, 1, 1, 8, 20, 30, 100 } );
	CDnnReferenceRegister* referenceDnnRegister = nullptr;

	dnns.SetBufferSize( numOfThreads );
	randoms.SetBufferSize( numOfThreads );

	for( int i = 0; i < numOfThreads; ++i ) {
		if( !useReference ) {
			randoms.Add( random );
			dnns.Add( new CDnn( randoms.Last(), mathEngine ) );
			createDnn( *dnns.Last() );
		} else if( i == 0 ) {
			CRandom rand( random );
			CDnn dnn( rand, mathEngine );
			createDnn( dnn );
			referenceDnnRegister = new CDnnReferenceRegister( mathEngine, dnn );
			// Like in class CDistributedInference
			// Here either a one more reference dnn can be used
			// Or also the original dnn, because no one can create a new reference dnn, while the inference
			dnns.Add( reinterpret_cast<CDnn*>( referenceDnnRegister ) );
		} else {
			dnns.Add( referenceDnnRegister->CreateReferenceDnn() );
		}
		setInputDnn( *dnns.Last(), *blob, nullptr, /*reshape*/( i == 0 || !useReference ) );
	}
	EXPECT_TRUE( !useReference || referenceDnnRegister != nullptr );
	return referenceDnnRegister;
}

static void runMultiThreadInference( std::function<void*( int )> getter, void( *run )( int, void* ), int numOfThreads )
{
	CPtrOwner<IThreadPool> threadPool( CreateThreadPool( numOfThreads ) );
	for( int i = 0; i < threadPool->Size(); ++i ) {
		threadPool->AddTask( i, run, getter( i ) );
	}
	threadPool->WaitAllTask();
}

// Scenario: learn dnn, then use multi-threaded inference
static void perfomanceTest( IMathEngine& mathEngine, bool useReference, int numOfThreads = 4 )
{
	CArray<CRandom> randoms;
	CPointerArray<CDnn> dnns;
	CDnnReferenceRegister* referenceDnnRegister = getTestDnns( mathEngine, dnns, randoms, useReference, numOfThreads );

	CPtrOwner<IPerformanceCounters> counters( mathEngine.CreatePerformanceCounters() );
	mathEngine.ResetPeakMemoryUsage();

	counters->Synchronise();
	runMultiThreadInference( [&]( int thread ) { return dnns[thread]; }, runDnn, numOfThreads );
	counters->Synchronise();

	GTEST_LOG_( INFO ) << "Run once multi-threaded " << ( useReference ? "(ref)" : "(cpy)" )
		<< "\nTime: " << ( double( ( *counters )[0].Value ) / 1000000 ) << " ms. "
		<< "\tPeak.Mem: " << ( double( mathEngine.GetPeakMemoryUsage() ) / 1024 / 1024 ) << " MB \n";

	CPtr<CDnnBlob> sourceBlob = CheckCast<CSourceLayer>( dnns[0]->GetLayer( "in" ).Ptr() )->GetBlob();
	CPtr<CDnnBlob> sinkBlob = CheckCast<CSinkLayer>( dnns[0]->GetLayer( "sink" ).Ptr() )->GetBlob();

	for( int i = 1; i < numOfThreads; ++i ) {
		EXPECT_TRUE( CompareBlobs( *sourceBlob,
			*( CheckCast<CSourceLayer>( dnns[i]->GetLayer( "in" ).Ptr() )->GetBlob() ) )
		) << " i = " << i;

		EXPECT_TRUE( CompareBlobs( *sinkBlob,
			*( CheckCast<CSinkLayer>( dnns[i]->GetLayer( "sink" ).Ptr() )->GetBlob() ) )
		) << " i = " << i;
	}

	if( referenceDnnRegister != nullptr ) {
		( void ) dnns.DetachAndReplaceAt( nullptr, 0 );
		dnns.DeleteAll(); // delete reference dnns first
		delete referenceDnnRegister;
	}
}

// Scenario: learn dnn, then use multi-threaded inference, each thread creates reference dnn by itself
static void implTest( IMathEngine& mathEngine, bool useReference, bool learn = true, int numOfThreads = 4 )
{
	// 1. Create and learn dnn
	CRandom random( 0x123 );
	CPtr<CDnnBlob> blob = getInitedBlob( mathEngine, random, { 1, 1, 1, 8, 20, 30, 100 } );
	CPtr<CDnnBlob> labelBlob = getInitedBlob( mathEngine, random, { 1, 1, 1, 1, 1, 1, 10 } );

	CDnn dnn( random, mathEngine );
	createDnn( dnn, learn );
	setInputDnn( dnn, *blob, ( learn ? labelBlob.Ptr() : nullptr ), /*reshape*/true );
	if( learn ) {
		learnDnn( dnn );
	}

	dnn.RunOnce();
	CPtr<CDnnBlob> sinkBlob = CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ).Ptr() )->GetBlob();

	CDnnReferenceRegister referenceDnnRegister( mathEngine, dnn );
	CReferenceDnnTestParam param( referenceDnnRegister, *blob, *sinkBlob, useReference );

	CPtrOwner<IPerformanceCounters> counters( mathEngine.CreatePerformanceCounters() );
	mathEngine.ResetPeakMemoryUsage();

	// 2. Run multi-threaded inference
	counters->Synchronise();
	runMultiThreadInference( [&param]( int ) { return &param; }, runDnnCreation, numOfThreads );
	counters->Synchronise();

	GTEST_LOG_( INFO ) << "Run multi-threaded inference and creation "
		<< "\t" << ( useReference ? "(ref)" : "(cpy)" )
		<< "\t" << ( learn ? "(learn)" : "" )
		<< "\nTime: " << ( double( ( *counters )[0].Value ) / 1000000 ) << " ms. "
		<< "\tPeak.Mem: " << ( double( mathEngine.GetPeakMemoryUsage() ) / 1024 / 1024 ) << " MB \n";
}

} // namespace NeoMLTest

//------------------------------------------------------------------------------------------------

TEST( ReferenceDnnTest, CDnnReferenceRegisterTest )
{
	// As mathEngine is owned, there are no buffers in pools left for any thread
	CPtrOwner<IMathEngine> mathEngine( CreateCpuMathEngine( /*memoryLimit*/0u ) );

	implTest( *mathEngine, /*useReference*/true );

	implTest( *mathEngine, /*useReference*/false );
}

TEST( ReferenceDnnTest, InferenceReferenceDnns )
{
	// As mathEngine is owned, there are no buffers in pools left for any thread
	CPtrOwner<IMathEngine> mathEngine( CreateCpuMathEngine( /*memoryLimit*/0u ) );

	perfomanceTest( *mathEngine, /*useReference*/true );

	perfomanceTest( *mathEngine, /*useReference*/false );
}
