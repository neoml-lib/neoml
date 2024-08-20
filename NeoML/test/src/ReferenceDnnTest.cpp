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

constexpr int iterationsRunOnce = 10;

struct CDnnReferenceTest : public CDnnReference {
	CDnnReferenceTest( CRandom& random, IMathEngine& mathEngine ) :
		CDnnReference( random, mathEngine )  {}
};

struct CReferenceDnnTestParam final {
	CReferenceDnnTestParam( CReferenceDnnFactory& ref, CDnnBlob& in, CDnnBlob& out ) :
		CReferenceDnnTestParam( in, out, /*useReference*/true, /*useDnn*/false )
	{ ReferenceDnnFactory = &ref; }

	CReferenceDnnTestParam( CDnnReference& dnnRef, CDnnBlob& in, CDnnBlob& out, bool useReference ) :
		CReferenceDnnTestParam( in, out, useReference, /*useDnn*/true )
	{ DnnRef = &dnnRef; }

	CDnnBlob& Input;
	CDnnBlob& Expected;
	const bool UseReference;
	const bool CheckOutput;
	const bool UseDnn;
	CReferenceDnnFactory* ReferenceDnnFactory = nullptr;
	CDnnReference* DnnRef = nullptr;

private:
	CReferenceDnnTestParam( CDnnBlob& in, CDnnBlob& out, bool useReference, bool useDnn, bool checkOutput = true ) :
		Input( in ), Expected( out ), UseReference( useReference ), CheckOutput( checkOutput ), UseDnn( useDnn )
	{}
};

static void runDnn( int thread, void* arg )
{
	CReferenceDnnTestParam& params = static_cast<CReferenceDnnTestParam*>( arg )[thread];
	ASSERT_TRUE( params.UseDnn );
	CDnn& dnn = params.DnnRef->Dnn;

	for( int i = 0; i < iterationsRunOnce; ++i ) {
		dnn.RunOnce();

		if( params.CheckOutput ) {
			CPtr<CDnnBlob> input = CheckCast<CSourceLayer>( dnn.GetLayer( "in" ).Ptr() )->GetBlob();
			EXPECT_TRUE( CompareBlobs( params.Input, *input ) ) << " thread = " << thread;

			CPtr<CDnnBlob> result = CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ).Ptr() )->GetBlob();
			EXPECT_TRUE( CompareBlobs( params.Expected, *result ) ) << " thread = " << thread;
		}
	}

	// for reference dnns only
	if( params.UseReference ) {
		const CDnn& dnnConst = dnn;
		EXPECT_NO_THROW( dnn.RequestReshape() );
		EXPECT_NO_THROW( dnn.RequestReshape( true ); );
		EXPECT_NO_THROW( dnn.RunAndBackwardOnce(); );
		// learning
		EXPECT_NO_THROW( dnn.DisableLearning(); );
		NEOML_EXPECT_THROW( dnn.EnableLearning() );
		bool isLearningEnabled{};
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
		bool autoRestartMode{};
		EXPECT_NO_THROW( autoRestartMode = dnn.GetAutoRestartMode(); );
		EXPECT_NO_THROW( dnn.SetAutoRestartMode( autoRestartMode ); );
		EXPECT_NO_THROW( dnn.ForceRebuild(); );
		EXPECT_NO_THROW( dnn.IsRebuildRequested(); );
		EXPECT_NO_THROW( dnn.Random(); );
		EXPECT_NO_THROW( dnn.GetMathEngine(); );
		const CDnnSolver* solverConst{};
		EXPECT_NO_THROW( solverConst = dnnConst.GetSolver(); );
		EXPECT_NE( solverConst, nullptr );
		CDnnSolver* solver{};
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
}

static void createDnn( CDnn& dnn, bool learn = false, bool composite = false, float dropoutRate = 0.1f )
{
	CBaseLayer* layer = Source( dnn, "in" );
	if( composite ) {
		layer = TransformerEncoder( 2, 8, dropoutRate, 10, TActivationFunction::AF_ReLU )( "te", layer );
	} else {
		layer = FullyConnected( 50, true )( "fc1", layer );
		layer = Dropout( dropoutRate )( "dp1", layer );
		layer = FullyConnected( 200 )( "fc2", layer );
		layer = Dropout( dropoutRate )( "dp2", layer );
		layer = FullyConnected( 10 )( "fc3", layer );
	}
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

static CPtr<CDnnReference> copyDnn( CDnn& oldDnn, CRandom& random )
{
	CPtr<CDnnReference> dnnRef = new CDnnReferenceTest( random, oldDnn.GetMathEngine() );

	CArray<const char*> layersList;
	oldDnn.GetLayerList( layersList );

	CMemoryFile file;
	for( const char* layerName : layersList ) {
		file.SeekToBegin();
		{
			CPtr<CBaseLayer> layer = oldDnn.GetLayer( layerName );
			CArchive archive( &file, CArchive::store );
			SerializeLayer( archive, oldDnn.GetMathEngine(), layer );
		}
		file.SeekToBegin();
		CPtr<CBaseLayer> copyLayer;
		{
			CArchive archive( &file, CArchive::load );
			SerializeLayer( archive, oldDnn.GetMathEngine(), copyLayer );
		}
		dnnRef->Dnn.AddLayer( *copyLayer );
	}
	return dnnRef;
}

static void runDnnCreation( int thread, void* arg )
{
	CReferenceDnnTestParam& params = *static_cast<CReferenceDnnTestParam*>( arg );

	for( int i = 0; i < iterationsRunOnce; ++i ) {
		CPtrOwner<CRandom> random;
		CPtr<CDnnReference> dnnRef;

		if( params.UseReference ) {
			dnnRef = params.ReferenceDnnFactory->CreateReferenceDnn( /*getOriginDnn*/( thread == 0 ) );
		} else {
			ASSERT_TRUE( params.UseDnn );
			random = new CRandom( params.DnnRef->Dnn.Random() );
			dnnRef = copyDnn( params.DnnRef->Dnn, *random );
		}
		CDnn* dnn = &dnnRef->Dnn;
		setInputDnn( *dnn, params.Input );

		for( int i = 0; i < 2; ++i ) {
			dnn->RunOnce();
		}

		if( params.CheckOutput ) {
			CPtr<CDnnBlob> input = CheckCast<CSourceLayer>( dnn->GetLayer( "in" ).Ptr() )->GetBlob();
			EXPECT_TRUE( CompareBlobs( params.Input, *input ) ) << " thread = " << thread;

			CPtr<CDnnBlob> result = CheckCast<CSinkLayer>( dnn->GetLayer( "sink" ).Ptr() )->GetBlob();
			EXPECT_TRUE( CompareBlobs( params.Expected, *result ) ) << " thread = " << thread;
		}

		IMathEngine& mathEngine = dnn->GetMathEngine();
		dnnRef.Release();
		mathEngine.CleanUp();
	}
}

static CPtr<CReferenceDnnFactory> getTestDnns( IMathEngine& mathEngine, CObjectArray<CDnnReference>& dnnRefs,
	CArray<CRandom>& randoms, bool useReference, bool learn, int numOfThreads )
{
	CRandom random( 0 );
	CPtr<CDnnBlob> blob = getInitedBlob( mathEngine, random, { 1, 1, 1, 8, 20, 30, 100 } );
	CPtr<CDnnBlob> labelBlob = getInitedBlob( mathEngine, random, { 1, 1, 1, 1, 1, 1, 10 } );
	CPtr<CReferenceDnnFactory> referenceDnnFactory = nullptr;

	dnnRefs.SetBufferSize( numOfThreads );
	randoms.SetBufferSize( numOfThreads );

	for( int i = 0; i < numOfThreads; ++i ) {
		if( !useReference ) {
			randoms.Add( random );
			dnnRefs.Add( new CDnnReferenceTest( randoms.Last(), mathEngine ) );
			createDnn( dnnRefs.Last()->Dnn );
		} else if( i == 0 ) {
			CRandom rand( random );
			CDnn dnn( rand, mathEngine );
			createDnn( dnn, learn );
			setInputDnn( dnn, *blob, ( learn ? labelBlob.Ptr() : nullptr ), /*reshape*/true );
			if( learn ) {
				learnDnn( dnn );
			}
			referenceDnnFactory = new CReferenceDnnFactory( std::move( dnn ) );
			// Like in class CDistributedInference
			// Here either a one more reference dnn can be used
			// Or also the original dnn, because no one can create a new reference dnn, while the inference
			dnnRefs.Add( referenceDnnFactory->CreateReferenceDnn( /*getOriginDnn*/true ) );
		} else {
			dnnRefs.Add( referenceDnnFactory->CreateReferenceDnn() );
		}
		setInputDnn( dnnRefs.Last()->Dnn, *blob, nullptr, /*reshape*/( i == 0 || !useReference ) );
	}
	EXPECT_TRUE( !useReference || referenceDnnFactory != nullptr );
	return referenceDnnFactory;
}

static void runMultiThreadInference( CReferenceDnnTestParam* params, IThreadPool::TFunction run, int numOfThreads )
{
	CPtrOwner<IThreadPool> threadPool( CreateThreadPool( numOfThreads ) );
	NEOML_NUM_THREADS( *threadPool, params, run )
}

// Scenario: learn dnn, then use multi-threaded inference
static void perfomanceTest( IMathEngine& mathEngine, bool useReference, bool learn = true, int numOfThreads = 4 )
{
	CArray<CRandom> randoms;
	CObjectArray<CDnnReference> dnnRefs;
	CPtr<CReferenceDnnFactory> referenceDnnFactory = getTestDnns( mathEngine, dnnRefs,
		randoms, useReference, learn, numOfThreads );
	referenceDnnFactory.Release(); // try to release the factory, to check dtor deletion order

	CPtr<CDnnBlob> sourceBlob = CheckCast<CSourceLayer>( dnnRefs[0]->Dnn.GetLayer( "in" ).Ptr() )->GetBlob();
	CPtr<CDnnBlob> sinkBlob = CheckCast<CSinkLayer>( dnnRefs[0]->Dnn.GetLayer( "sink" ).Ptr() )->GetBlob();

	CArray<CReferenceDnnTestParam> params;
	params.SetBufferSize( numOfThreads );
	for( int i = 0; i < numOfThreads; ++i ) {
		params.Add( CReferenceDnnTestParam( *( dnnRefs[i] ), *sourceBlob, *sinkBlob, useReference ) );
	}

	CPtrOwner<IPerformanceCounters> counters( mathEngine.CreatePerformanceCounters() );
	mathEngine.ResetPeakMemoryUsage();

	counters->Synchronise();
	runMultiThreadInference( params.GetPtr(), runDnn, numOfThreads );
	counters->Synchronise();

	GTEST_LOG_( INFO ) << "Run once multi-threaded " << ( useReference ? "(ref)" : "(cpy)" )
		<< "\nTime: " << GetTimeScaled( *counters ) << " ms. "
		<< "\tPeak.Mem: " << GetPeakMemScaled( mathEngine ) << " MB \n";
}

// Scenario: learn dnn, then use multi-threaded inference, each thread creates reference dnn by itself
static void implTest( IMathEngine& mathEngine, bool useReference, bool learn = true, int numOfThreads = 4,
	bool composite = false )
{
	// 1. Create and learn dnn
	CRandom random( 0x123 );
	CPtr<CDnnBlob> blob = getInitedBlob( mathEngine, random,
		{ 1, 1, 1, (composite ? 1 : 8), (composite ? 1 : 20), (composite ? 1 : 30), 100 } );
	CPtr<CDnnBlob> labelBlob = getInitedBlob( mathEngine, random, { 1, 1, 1, 1, 1, 1, (composite ? 100 : 10) } );

	CPtr<CDnnReference> dnnRef = new CDnnReferenceTest( random, mathEngine );
	CDnn& dnn = dnnRef->Dnn;
	createDnn( dnn, learn, composite );
	setInputDnn( dnn, *blob, ( learn ? labelBlob.Ptr() : nullptr ), /*reshape*/true );
	if( learn ) {
		learnDnn( dnn );
	}

	dnn.RunOnce();
	CPtr<CDnnBlob> sinkBlob = CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ).Ptr() )->GetBlob();

	CPtr<CReferenceDnnFactory> referenceDnnFactory;
	CPtrOwner<CReferenceDnnTestParam> param;
	if( useReference ) {
		referenceDnnFactory = new CReferenceDnnFactory( mathEngine, dnn );
		dnnRef.Release(); // check for factory do not depends on given dnn by const ref
		param = new CReferenceDnnTestParam( *referenceDnnFactory, *blob, *sinkBlob );
	} else {
		param = new CReferenceDnnTestParam( *dnnRef, *blob, *sinkBlob, useReference );
	}

	CPtrOwner<IPerformanceCounters> counters( mathEngine.CreatePerformanceCounters() );
	mathEngine.ResetPeakMemoryUsage();

	// 2. Run multi-threaded inference
	counters->Synchronise();
	runMultiThreadInference( param, runDnnCreation, numOfThreads );
	counters->Synchronise();

	GTEST_LOG_( INFO ) << "Run multi-threaded inference and creation "
		<< "\t" << ( useReference ? "(ref)" : "(cpy)" )
		<< "\nTime: " << GetTimeScaled( *counters ) << " ms. "
		<< "\tPeak.Mem: " << GetPeakMemScaled( MathEngine() ) << " MB \n";

	if( referenceDnnFactory != nullptr ) {
		CPtr<CDnnReference> originDnn = referenceDnnFactory->CreateReferenceDnn( /*originDnn*/true );
		referenceDnnFactory.Release(); // try to release the factory, to check dtor deletion order

		originDnn->Dnn.RunOnce();
		CPtr<CDnnBlob> result = CheckCast<CSinkLayer>( originDnn->Dnn.GetLayer( "sink" ).Ptr() )->GetBlob();
		EXPECT_TRUE( CompareBlobs( *sinkBlob, *result ) );
	}
}

} // namespace NeoMLTest

//------------------------------------------------------------------------------------------------

TEST( ReferenceDnnTest, CReferenceDnnFactoryTest )
{
	// As mathEngine is owned, there are no buffers in pools left for any thread
	CPtrOwner<IMathEngine> mathEngine( CreateCpuMathEngine( /*memoryLimit*/0u ) );

	implTest( *mathEngine, /*useReference*/true );

	implTest( *mathEngine, /*useReference*/false );
}

TEST( ReferenceDnnTest, CReferenceDnnFactoryCompositeTest )
{
	// As mathEngine is owned, there are no buffers in pools left for any thread
	CPtrOwner<IMathEngine> mathEngine( CreateCpuMathEngine( /*memoryLimit*/0u ) );

	implTest( *mathEngine, /*useReference*/true, /*learn*/true, /*numOfThreads*/5, /*composite*/true );

	implTest( *mathEngine, /*useReference*/false, /*learn*/true, /*numOfThreads*/5, /*composite*/true );
}

TEST( ReferenceDnnTest, InferenceReferenceDnns )
{
	// As mathEngine is owned, there are no buffers in pools left for any thread
	CPtrOwner<IMathEngine> mathEngine( CreateCpuMathEngine( /*memoryLimit*/0u ) );

	perfomanceTest( *mathEngine, /*useReference*/true );

	perfomanceTest( *mathEngine, /*useReference*/false );
}
