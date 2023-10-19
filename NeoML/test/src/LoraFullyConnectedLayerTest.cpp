/* Copyright © 2023 ABBYY

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
#include <NeoML/NeoML.h>

using namespace NeoML;
using namespace NeoMLTest;

namespace NeoMLTest {

static void setDnnInputs( CDnn& dnn, int labelSize, int inputSize = 1000 )
{
	CArray<float> inArr;
	inArr.Add( 0.01f, inputSize );
	CPtr<CDnnBlob> in = CDnnBlob::CreateTensor( dnn.GetMathEngine(), CT_Float, { 1, 1, 1, 1, 1, 1, inputSize } );
	in->CopyFrom( inArr.GetPtr() );

	CArray<float> labelArr;
	labelArr.Add( 1.f, labelSize );
	CPtr<CDnnBlob> labels = CDnnBlob::CreateTensor( dnn.GetMathEngine(), CT_Float, { 1, 1, 1, 1, 1, 1, labelSize } );
	labels->CopyFrom( labelArr.GetPtr() );

	CheckCast<CSourceLayer>( dnn.GetLayer( "in" ) )->SetBlob( in );
	CheckCast<CSourceLayer>( dnn.GetLayer( "label" ) )->SetBlob( labels );
}

static void buildDnn( CDnn& dnn, int outputSize )
{
	CPtr<CSourceLayer> dataLayer = new CSourceLayer( dnn.GetMathEngine() );
	dataLayer->SetName( "in" );
	dnn.AddLayer( *dataLayer );

	CPtr<CLoraFullyConnectedLayer> full = new CLoraFullyConnectedLayer( dnn.GetMathEngine() );
	full->SetNumberOfElements( outputSize );
	full->SetName( "full" );
	full->SetZeroFreeTerm( false );
	full->Connect( *dataLayer );
	dnn.AddLayer( *full );

	CPtr<CSourceLayer> label = new CSourceLayer( dnn.GetMathEngine() );
	label->SetName( "label" );
	dnn.AddLayer( *label );

	CPtr<CEuclideanLossLayer> loss = new CEuclideanLossLayer( dnn.GetMathEngine() );
	loss->SetName( "loss" );
	loss->Connect( 0, *full );
	loss->Connect( 1, *label );
	dnn.AddLayer( *loss );

	CPtr<CSinkLayer> out = new CSinkLayer( dnn.GetMathEngine() );
	out->SetName( "sink" );
	out->Connect( *full );
	dnn.AddLayer( *out );

	CPtr<CDnnAdaptiveGradientSolver> solver = new CDnnAdaptiveGradientSolver( dnn.GetMathEngine() );
	dnn.SetSolver( solver.Ptr() );
}

static void testArchive( CDnn& dnn, const char* archiveName = "lora.archive", bool setSolver = true )
{
	{
		CArchiveFile file( archiveName, CArchive::store, GetPlatformEnv() );
		CArchive archive( &file, CArchive::store );
		archive.Serialize( dnn );
	}
	dnn.DeleteAllLayers();
	{
		CArchiveFile file( archiveName, CArchive::load, GetPlatformEnv() );
		CArchive archive( &file, CArchive::load );
		archive.Serialize( dnn );
	}

	if( setSolver ) {
		CPtr<CDnnSolver> solver = new CDnnAdaptiveGradientSolver( dnn.GetMathEngine() );
		dnn.SetSolver( solver.Ptr() );
	}
}

static void testLearn( CDnn& dnn, int epochs )
{
	for( int i = 0; i < epochs; ++i ) {
		dnn.RunAndLearnOnce();
	}
	dnn.RunOnce();
}

struct CLoraTestParams final {
	int rank = 2;
	float alpha = 1.f;
	float dropout = 0.f;
	int epochs = 100;

	bool buildLora = false;
	bool archiveAfterBuild = false;
	bool learnBeforeMerge = false;
	bool mergeLora = false;
	bool destroyLora = false;
	bool archiveBeforeMerge = false;
	bool archiveAfterMerge = false;
	bool archiveAfterDestroy = false;
};

static void testImpl( const CLoraTestParams& params )
{
	CRandom random( 42 );

	const int outputSize = 5;
	CDnn dnn( random, MathEngine() );

	buildDnn( dnn, outputSize );
	CheckCast<CLoraFullyConnectedLayer>( dnn.GetLayer( "full" ) )->SetStoreSeparateLoRA( false );
	setDnnInputs( dnn, outputSize );

	dnn.RunAndLearnOnce();

	float outputData[outputSize]{};
	{
		dnn.RunOnce();

		const CPtr<CDnnBlob> sinkBlob = CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ) )->GetBlob();
		EXPECT_EQ( outputSize, sinkBlob->GetDataSize() );
		sinkBlob->CopyTo( outputData );
	}

	if( params.buildLora ) {
		CheckCast<CLoraFullyConnectedLayer>( dnn.GetLayer( "full" ) )->BuildLoRA( params.rank, params.alpha, params.dropout );
		if( params.archiveAfterBuild ) {
			testArchive( dnn );
			setDnnInputs( dnn, outputSize );
		}

		dnn.RunOnce();
		auto sinkData = CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ) )->GetBlob()->GetData();
		for( int i = 0; i < outputSize; ++i ) {
			EXPECT_TRUE( FloatEq( sinkData.GetValueAt( i ), outputData[i], 1e-4f ) );
		}
	}
	if( params.learnBeforeMerge ) {
		testLearn( dnn, params.epochs );
	}
	if( params.archiveBeforeMerge ) {
		testArchive( dnn );
		setDnnInputs( dnn, outputSize );
	}
	if( params.mergeLora ) {
		CheckCast<CLoraFullyConnectedLayer>( dnn.GetLayer( "full" ) )->MergeWeightsLoRA();
	}
	if( params.archiveAfterMerge ) {
		testArchive( dnn );
		setDnnInputs( dnn, outputSize );
	}
	if( params.destroyLora ) {
		CheckCast<CLoraFullyConnectedLayer>( dnn.GetLayer( "full" ) )->DestroyUnmergedLoRA();
	}
	if( params.archiveAfterDestroy ) {
		testArchive( dnn );
		setDnnInputs( dnn, outputSize );
	}

	testLearn( dnn, params.epochs );

	const CPtr<CDnnBlob> sinkBlob = CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ) )->GetBlob();
	EXPECT_EQ( outputSize, sinkBlob->GetDataSize() );

	const float loss = CheckCast<CLossLayer>( dnn.GetLayer( "loss" ) )->GetLastLoss();
	GTEST_LOG_( INFO ) << "loss = " << loss;
	if( params.learnBeforeMerge ) {
		EXPECT_LE( loss, 0.5f );
	}
}

} // namespace NeoMLTest

//---------------------------------------------------------------------------------------------------------------------------------------

TEST( LoraFullyConnectedLayerTest, Simple )
{
	CLoraTestParams params;
	testImpl( params );
}

TEST( LoraFullyConnectedLayerTest, SimpleLora )
{
	CLoraTestParams params;
	params.buildLora = true;
	testImpl( params );
}

TEST( LoraFullyConnectedLayerTest, DestroyLora )
{
	CLoraTestParams params;
	params.buildLora = true;
	params.destroyLora = true;
	testImpl( params );
}

TEST( LoraFullyConnectedLayerTest, DestroyNoBuiltLora )
{
	CLoraTestParams params;
	params.destroyLora = true;
	testImpl( params );
}

TEST( LoraFullyConnectedLayerTest, DestroyLearnedLora )
{
	CLoraTestParams params;
	params.epochs = 50;
	params.buildLora = true;
	params.learnBeforeMerge = true;
	params.destroyLora = true;
	testImpl( params );
}

TEST( LoraFullyConnectedLayerTest, ArchiveAfterBuild )
{
	CLoraTestParams params;
	params.buildLora = true;
	params.archiveAfterBuild = true;
	testImpl( params );
}

TEST( LoraFullyConnectedLayerTest, MergeLora )
{
	CLoraTestParams params;
	params.buildLora = true;
	params.mergeLora = true;
	testImpl( params );
}

TEST( LoraFullyConnectedLayerTest, MergeLearnedLora )
{
	CLoraTestParams params;
	params.epochs = 50;
	params.buildLora = true;
	params.learnBeforeMerge = true;
	params.mergeLora = true;
	testImpl( params );
}

TEST( LoraFullyConnectedLayerTest, ArchiveBefore )
{
	CLoraTestParams params;
	params.buildLora = true;
	params.archiveBeforeMerge = true;
	testImpl( params );
}

TEST( LoraFullyConnectedLayerTest, LearnedArchiveBefore )
{
	CLoraTestParams params;
	params.epochs = 50;
	params.buildLora = true;
	params.learnBeforeMerge = true;
	params.mergeLora = true;
	params.archiveBeforeMerge = true;
	testImpl( params );
}

TEST( LoraFullyConnectedLayerTest, ArchiveAfter )
{
	CLoraTestParams params;
	params.buildLora = true;
	params.mergeLora = true;
	params.archiveAfterMerge = true;
	testImpl( params );
}

TEST( LoraFullyConnectedLayerTest, LearnedArchiveAfter )
{
	CLoraTestParams params;
	params.epochs = 50;
	params.buildLora = true;
	params.learnBeforeMerge = true;
	params.mergeLora = true;
	params.archiveAfterMerge = true;
	testImpl( params );
}

TEST( LoraFullyConnectedLayerTest, LearnedArchiveBeforeAndAfter )
{
	CLoraTestParams params;
	params.epochs = 50;
	params.buildLora = true;
	params.learnBeforeMerge = true;
	params.mergeLora = true;
	params.archiveBeforeMerge = true;
	params.archiveAfterMerge = true;
	testImpl( params );
}

TEST( LoraFullyConnectedLayerTest, ArchiveDestroyed )
{
	CLoraTestParams params;
	params.buildLora = true;
	params.learnBeforeMerge = true;
	params.destroyLora = true;
	params.archiveAfterDestroy = true;
	testImpl( params );
}

TEST( LoraFullyConnectedLayerTest, LoadLoraSaved )
{
	CRandom random( 42 );

	const int outputSize = 5;
	CDnn dnn( random, MathEngine() );

	buildDnn( dnn, outputSize );
	setDnnInputs( dnn, outputSize );
	CheckCast<CLoraFullyConnectedLayer>( dnn.GetLayer( "full" ) )->SetStoreSeparateLoRA( false );

	CheckCast<CLoraFullyConnectedLayer>( dnn.GetLayer( "full" ) )->BuildLoRA( /*rank*/3, /*alpha*/1.f, /*dropout*/0.f );
	{ // store the Lora is built and isn't learned
		CArchiveFile file( "lora.archive", CArchive::store, GetPlatformEnv() );
		CArchive archive( &file, CArchive::store );
		archive.Serialize( dnn );
	}

	dnn.RunAndLearnOnce();

	CheckCast<CLoraFullyConnectedLayer>( dnn.GetLayer( "full" ) )->DestroyUnmergedLoRA();
	{ // load the Lora is built (if Lora is destroyed)
		CArchiveFile file( "lora.archive", CArchive::load, GetPlatformEnv() );
		CArchive archive( &file, CArchive::load );
		archive.Serialize( dnn );
	}
	setDnnInputs( dnn, outputSize );

	testLearn( dnn, 100 );

	const float loss = CheckCast<CLossLayer>( dnn.GetLayer( "loss" ) )->GetLastLoss();
	GTEST_LOG_( INFO ) << "loss = " << loss;
}

TEST( LoraFullyConnectedLayerTest, LoadLoraNotSaved )
{
	CRandom random( 42 );

	const int outputSize = 5;
	CDnn dnn( random, MathEngine() );

	buildDnn( dnn, outputSize );
	setDnnInputs( dnn, outputSize );

	{ // store the Lora isn't built
		CArchiveFile file( "lora.archive", CArchive::store, GetPlatformEnv() );
		CArchive archive( &file, CArchive::store );
		archive.Serialize( dnn );
	}

	dnn.RunAndLearnOnce();

	CheckCast<CLoraFullyConnectedLayer>( dnn.GetLayer( "full" ) )->SetStoreSeparateLoRA( false );
	CheckCast<CLoraFullyConnectedLayer>( dnn.GetLayer( "full" ) )->BuildLoRA( /*rank*/3, /*alpha*/1.f, /*dropout*/0.f );
	{ // load the Lora is destroyed (if Lora is built)
		CArchiveFile file( "lora.archive", CArchive::load, GetPlatformEnv() );
		CArchive archive( &file, CArchive::load );
		archive.Serialize( dnn );
	}
	setDnnInputs( dnn, outputSize );

	testLearn( dnn, 100 );

	const float loss = CheckCast<CLossLayer>( dnn.GetLayer( "loss" ) )->GetLastLoss();
	GTEST_LOG_( INFO ) << "loss = " << loss;
}

//---------------------------------------------------------------------------------------------------------------------------------------

TEST( LoraFullyConnectedLayerTest, DropoutSimpleLora )
{
	CLoraTestParams params;
	params.dropout = 0.5f;
	params.buildLora = true;
	testImpl( params );
}

TEST( LoraFullyConnectedLayerTest, DropoutDestroyLearnedLora )
{
	CLoraTestParams params;
	params.dropout = 0.5f;
	params.buildLora = true;
	params.learnBeforeMerge = true;
	params.destroyLora = true;
	testImpl( params );
}

TEST( LoraFullyConnectedLayerTest, DropoutMergeLearnedLora )
{
	CLoraTestParams params;
	params.dropout = 0.5f;
	params.epochs = 50;
	params.buildLora = true;
	params.learnBeforeMerge = true;
	params.mergeLora = true;
	testImpl( params );
}

TEST( LoraFullyConnectedLayerTest, DropoutLearnedArchiveBefore )
{
	CLoraTestParams params;
	params.dropout = 0.5f;
	params.epochs = 50;
	params.buildLora = true;
	params.learnBeforeMerge = true;
	params.mergeLora = true;
	params.archiveBeforeMerge = true;
	testImpl( params );
}

TEST( LoraFullyConnectedLayerTest, DropoutLearnedArchiveAfter )
{
	CLoraTestParams params;
	params.dropout = 0.5f;
	params.epochs = 50;
	params.buildLora = true;
	params.learnBeforeMerge = true;
	params.mergeLora = true;
	params.archiveAfterMerge = true;
	testImpl( params );
}

TEST( LoraFullyConnectedLayerTest, DropoutLearnedArchiveBeforeAndAfter )
{
	CLoraTestParams params;
	params.dropout = 0.5f;
	params.epochs = 50;
	params.buildLora = true;
	params.learnBeforeMerge = true;
	params.mergeLora = true;
	params.archiveBeforeMerge = true;
	params.archiveAfterMerge = true;
	testImpl( params );
}

TEST( LoraFullyConnectedLayerTest, DropoutArchiveDestroyed )
{
	CLoraTestParams params;
	params.dropout = 0.5f;
	params.buildLora = true;
	params.learnBeforeMerge = true;
	params.destroyLora = true;
	params.archiveAfterDestroy = true;
	testImpl( params );
}

//---------------------------------------------------------------------------------------------------------------------------------------

namespace NeoMLTest {

static void buildTransformer( CDnn& dnn )
{
	const int headCount = 2;

	CPtr<CSourceLayer> widthsSourceLayer = AddLayer<CSourceLayer>( "widths", dnn );
	CPtr<CSourceLayer> qSourceLayer = AddLayer<CSourceLayer>( "Q", dnn );

	CPtr<CTransformerEncoderLayer> transformerLayer = AddLayer<CTransformerEncoderLayer>(
		"transformer", { widthsSourceLayer, qSourceLayer } );
	transformerLayer->SetHeadCount( headCount );
	transformerLayer->SetHiddenSize( 36 );
	transformerLayer->SetDropoutRate( 0.2f );
	transformerLayer->SetFeedForwardSize( 16 );
	transformerLayer->SetMaskType( CMultiheadAttentionLayer::MT_OneObject );
	
	( void ) AddLayer<CSinkLayer>( "out", { transformerLayer } );
}

static void createTransformerInput( CPtr<CDnnBlob>& widthsBlob, CPtr<CDnnBlob>& qBlob,
	const char* DnnNameToDumpInputs = nullptr )
{
	const int batchSize = 1;
	const int ListSize_Q = 1;
	const int ListSize_W = 3;

	// Inputs widths and Q-matrix
	float widthsArray[ListSize_W]{ 3.f, 1.f, 2.f };
	float qArray[ListSize_Q]{ 0.f };

	widthsBlob = CDnnBlob::CreateBlob( MathEngine(), CT_Float, { 1, batchSize, 1, 1, 1, 1, ListSize_W } );
	qBlob = CDnnBlob::CreateBlob( MathEngine(), CT_Float, { 1, batchSize, 1, 1, 1, 1, ListSize_Q } );

	widthsBlob->CopyFrom( widthsArray );
	qBlob->CopyFrom( qArray );

	if( DnnNameToDumpInputs != nullptr ) {
		CArchiveFile wfile( CString( DnnNameToDumpInputs ) + ".widths.input", CArchive::store, GetPlatformEnv() );
		CArchive warchive( &wfile, CArchive::store );
		widthsBlob->Serialize( warchive );

		CArchiveFile qfile( CString( DnnNameToDumpInputs ) + ".Q.input", CArchive::store, GetPlatformEnv() );
		CArchive qarchive( &qfile, CArchive::store );
		qBlob->Serialize( qarchive );
	}
}

static void setTransformerInput( CDnn& dnn, CPtr<CDnnBlob> widthsBlob, CPtr<CDnnBlob> qBlob )
{
	CheckCast<CSourceLayer>( dnn.GetLayer( "widths" ) )->SetBlob( widthsBlob );
	CheckCast<CSourceLayer>( dnn.GetLayer( "Q" ) )->SetBlob( qBlob );
}

static void testTransformerArchive( CDnn& dnn )
{
	const bool haveLora = CheckCast<CTransformerEncoderLayer>( dnn.GetLayer( "transformer" ) )->GetRankLoRA() > 0;
	if( haveLora ) {
		CArchiveFile file( "transformerLora.cnnarch", CArchive::store, GetPlatformEnv() );
		CArchive archive( &file, CArchive::store );
		CheckCast<CTransformerEncoderLayer>( dnn.GetLayer( "transformer" ) )->SerializeWeightsLoRA( archive );
		CheckCast<CTransformerEncoderLayer>( dnn.GetLayer( "transformer" ) )->DestroyUnmergedLoRA();
	}

	testArchive( dnn, "transformerItself.cnnarch", /*setSolver*/false );

	if( haveLora ) {
		CArchiveFile file( "transformerLora.cnnarch", CArchive::load, GetPlatformEnv() );
		CArchive archive( &file, CArchive::load );
		CheckCast<CTransformerEncoderLayer>( dnn.GetLayer( "transformer" ) )->SerializeWeightsLoRA( archive );
	}
}

} // namespace NeoMLTest

//---------------------------------------------------------------------------------------------------------------------------------------

TEST( LoraFullyConnectedLayerTest, Transformer )
{
	CRandom random( 87 );

	const int outputSize = 3;
	const int epochs = 10;

	CPtr<CDnnBlob> widthsBlob, qBlob;
	createTransformerInput( widthsBlob, qBlob );

	// Simple nn
	CDnn dnn( random, MathEngine() );
	buildTransformer( dnn );
	setTransformerInput( dnn, widthsBlob, qBlob );

	testLearn( dnn, epochs );

	testTransformerArchive( dnn );
	setTransformerInput( dnn, widthsBlob, qBlob );

	CheckCast<CTransformerEncoderLayer>( dnn.GetLayer( "transformer" ) )->BuildLoRA( /*rank*/2, /*alpha*/2.f, /*dropout*/0.2f );

	testLearn( dnn, epochs );

	testTransformerArchive( dnn );
	setTransformerInput( dnn, widthsBlob, qBlob );

	CheckCast<CTransformerEncoderLayer>( dnn.GetLayer( "transformer" ) )->MergeWeightsLoRA();

	testLearn( dnn, epochs );

	testTransformerArchive( dnn );
	setTransformerInput( dnn, widthsBlob, qBlob );

	dnn.RunOnce();
	for( int i = 0; i < epochs; ++i ) {
		dnn.RunOnce();
	}

	// Expected output
	const float expectedOutput[outputSize]{ 0.753938, 0.659215, -1.413153 };

	CPtr<CDnnBlob> sinkBLob = CheckCast<CSinkLayer>( dnn.GetLayer( "out" ) )->GetBlob();
	EXPECT_EQ( outputSize, sinkBLob->GetDataSize() );

	float outputBuff[outputSize]{};
	sinkBLob->CopyTo( outputBuff );

	if( /*dumpOutputs*/false ) {
		CArchiveFile qfile( "transformerLora.out.output", CArchive::store, GetPlatformEnv() );
		qfile.Write( outputBuff, outputSize * sizeof( float ) );
	}

	// Checking for equality of all elements
	for( int i = 0; i < outputSize; ++i ) {
		EXPECT_TRUE( FloatEq( expectedOutput[i], outputBuff[i], 1e-4f ) );
	}
}

TEST( LoraFullyConnectedLayerTest, TransformerOldLoad )
{
	CRandom random( 87 );

	const CString filepath = GetTestDataFilePath( "data", "transformerLoraOld" );

	CArray<float> expectedOutput{};
	{
		CArchiveFile file( filepath + ".out.output", CArchive::load, GetPlatformEnv() );
		expectedOutput.SetSize( static_cast<int>( file.GetLength() / sizeof( float ) ) );
		file.Read( expectedOutput.GetPtr(), expectedOutput.Size() * sizeof( float ) );
	}

	CPtr<CDnnBlob> widthsBlob = new CDnnBlob( MathEngine() );
	{
		CArchiveFile file( filepath + ".widths.input", CArchive::load, GetPlatformEnv() );
		CArchive archive( &file, CArchive::load );
		widthsBlob->Serialize( archive );
	}
	CPtr<CDnnBlob> qBlob = new CDnnBlob( MathEngine() );
	{
		CArchiveFile file( filepath + ".Q.input", CArchive::load, GetPlatformEnv() );
		CArchive archive( &file, CArchive::load );
		qBlob->Serialize( archive );
	}

	CDnn dnn( random, MathEngine() );
	{
		CArchiveFile file( filepath + ".cnnarch", CArchive::load, GetPlatformEnv() );
		CArchive archive( &file, CArchive::load );
		archive.Serialize( dnn );
	}
	setTransformerInput( dnn, widthsBlob, qBlob );

	const int epochs = 10;

	testLearn( dnn, epochs );

	CheckCast<CTransformerEncoderLayer>( dnn.GetLayer( "transformer" ) )->BuildLoRA( /*rank*/2, /*alpha*/2.f, /*dropout*/0.2f );

	testLearn( dnn, epochs );

	CheckCast<CTransformerEncoderLayer>( dnn.GetLayer( "transformer" ) )->MergeWeightsLoRA();

	testLearn( dnn, epochs );

	dnn.RunOnce();
	for( int i = 0; i < epochs; ++i ) {
		dnn.RunOnce();
	}

	CPtr<CDnnBlob> sinkBLob = CheckCast<CSinkLayer>( dnn.GetLayer( "out" ) )->GetBlob();
	EXPECT_EQ( expectedOutput.Size(), sinkBLob->GetDataSize() );
	// Checking for equality of all elements
	for( int i = 0; i < expectedOutput.Size(); ++i ) {
		EXPECT_TRUE( FloatEq( expectedOutput[i], sinkBLob->GetData().GetValueAt( i ), 1e-4f ) );
	}
}

//---------------------------------------------------------------------------------------------------------------------------------------

namespace NeoMLTest {

static void testPrintTime( IPerformanceCounters* counters, const char* title )
{
	for( const auto& counter : *counters ) {
		GTEST_LOG_( INFO ) << title << counter.Name << ": " << counter.Value
			<< "  (" << ( double( counter.Value ) / 1000000 ) << " ms.)";
	}
	GTEST_LOG_( INFO ) << title << "Peak memory usage: " << ( double( MathEngine().GetPeakMemoryUsage() ) / 1024 / 1024 ) << " MB.";
}

static void testBehchmarkImpl( float dropout, int epochs, bool lora, CArchive::TDirection arhive = CArchive::load )
{
	CRandom rand( 42 );

	const int inputSize = 100000;
	const int outputSize = 1000;
	const char* filename = "loraBenchmark.archive";

	CDnn dnn( rand, MathEngine() );
	if( arhive == CArchive::load ) {
		CArchiveFile file( filename, CArchive::load, GetPlatformEnv() );
		CArchive archive( &file, CArchive::load );
		archive.Serialize( dnn );
	} else {
		buildDnn( dnn, outputSize );
	}

	setDnnInputs( dnn, outputSize, inputSize );
	GTEST_LOG_( INFO ) << "inputSize = " << inputSize << "  outputSize = " << outputSize;
	// Speed measures:
	auto counters = dnn.GetMathEngine().CreatePerformanceCounters();

	if( arhive == CArchive::store ) {
		dnn.RunAndLearnOnce();
		counters->Synchronise();

		testLearn( dnn, epochs );
		counters->Synchronise();
		testPrintTime( counters, "RunAndLearnOnce " );

		CArchiveFile file( filename, CArchive::store, GetPlatformEnv() );
		CArchive archive( &file, CArchive::store );
		archive.Serialize( dnn );
		GTEST_LOG_( INFO ) << "Serialized";

	} else {

		if( lora ) {
			CheckCast<CLoraFullyConnectedLayer>( dnn.GetLayer( "full" ) )->BuildLoRA( /*rank*/2, /*alpha*/1.f, dropout );
		}

		dnn.RunAndLearnOnce();
		counters->Synchronise();

		testLearn( dnn, epochs );

		counters->Synchronise();
		testPrintTime( counters, "RunAndLearnOnce " );

		if( lora ) {
			counters->Synchronise();

			CheckCast<CLoraFullyConnectedLayer>( dnn.GetLayer( "full" ) )->MergeWeightsLoRA();

			counters->Synchronise();
			testPrintTime( counters, "Merge LoRA " );
		}

		dnn.RunOnce();
		counters->Synchronise();

		for( int i = 0; i < epochs; ++i ) {
			dnn.RunOnce();
		}

		counters->Synchronise();
		testPrintTime( counters, "RunOnce " );
	}
	delete counters;

	const float loss = CheckCast<CLossLayer>( dnn.GetLayer( "loss" ) )->GetLastLoss();
	GTEST_LOG_( INFO ) << "loss = " << loss;
}

} // namespace NeoMLTest

//---------------------------------------------------------------------------------------------------------------------------------------

TEST( LoraFullyConnectedLayerTest, DISABLED_BenchmarkCreate )
{
	testBehchmarkImpl( /*dropout*/0.f, /*epochs*/100, /*lora*/false, CArchive::store );
}

TEST( LoraFullyConnectedLayerTest, DISABLED_BenchmarkLora )
{
	testBehchmarkImpl( /*dropout*/0.f, /*epochs*/1000, /*lora*/true );
}

TEST( LoraFullyConnectedLayerTest, DISABLED_BenchmarkLoraDropout )
{
	testBehchmarkImpl( /*dropout*/0.2f, /*epochs*/1000, /*lora*/true );
}

TEST( LoraFullyConnectedLayerTest, DISABLED_BenchmarkNoLora )
{
	testBehchmarkImpl( /*dropout*/0.f, /*epochs*/1000, /*lora*/false );
}

//---------------------------------------------------------------------------------------------------------------------------------------

namespace NeoMLTest {

static void setDnnSpecialInputs( CDnn& dnn, int labelSize, int inputSize )
{
	CArray<float> inArr;
	for( int i = 0; i < inputSize; ++i ) {
		inArr.Add( 0.01f * i );
	}
	CPtr<CDnnBlob> in = CDnnBlob::CreateTensor( dnn.GetMathEngine(), CT_Float, { 1, 1, 1, 1, 1, 1, inputSize } );
	in->CopyFrom( inArr.GetPtr() );

	CArray<float> labelArr;
	for( int i = 0; i < labelSize; ++i ) {
		labelArr.Add( 0.7f * ( labelSize - i ) );
	}
	CPtr<CDnnBlob> labels = CDnnBlob::CreateTensor( dnn.GetMathEngine(), CT_Float, { 1, 1, 1, 1, 1, 1, labelSize } );
	labels->CopyFrom( labelArr.GetPtr() );

	CheckCast<CSourceLayer>( dnn.GetLayer( "in" ) )->SetBlob( in );
	CheckCast<CSourceLayer>( dnn.GetLayer( "label" ) )->SetBlob( labels );
}

static void trainAndTest( CDnn& dnn, int epochs, bool verbose )
{
	const float eps = 0.1;
	CPtr<CLossLayer> lossLayer = CheckCast<CLossLayer>( dnn.GetLayer( "loss" ) );
	float prevTrainLoss = lossLayer->GetLastLoss();
	float prevTestLoss = prevTrainLoss;

	// In several eposhs perform train + test (let train == test), where for each epoch calculate separate train-loss and test-loss.
	// Losses should decrease constantly, because we have the only 1 BLOB, exact equality is not garanteed, it depends on LoRA configs.
	for( int i = 0; i < epochs; ++i ) {
		dnn.RunAndLearnOnce();

		const float trainLoss = lossLayer->GetLastLoss();

		dnn.RunOnce();

		const float testLoss = lossLayer->GetLastLoss();

		if( verbose ) {
			printf( "trainLoss = %10f\ttestLoss = %10f\n", trainLoss, testLoss );
		}
		EXPECT_TRUE( ( trainLoss - eps ) <= ( prevTrainLoss + eps ) ); // losses should decrease (mostly)
		prevTrainLoss = trainLoss;

		EXPECT_TRUE( ( testLoss - eps ) <= ( prevTestLoss + eps ) ); // losses should decrease (mostly)
		prevTestLoss = testLoss;
	}
}

static void printStatus( float loss, CPtr<CDnnBlob> sinkBlob )
{
	printf( "\nprintStatus loss = %f \n", loss );
	printf( "printStatus output = { " );
	for( int i = 0; i < sinkBlob->GetDesc().BlobSize(); ++i ) {
		printf( "%f ", sinkBlob->GetData().GetValueAt( i ) );
	}
	printf( "}\n\n" );
}

static void checkStatus( float originLoss, float loss, CPtr<CDnnBlob> originSinkBlob, CPtr<CDnnBlob> sinkBlob )
{
	EXPECT_TRUE( FloatEq( originLoss, loss ) );
	for( int i = 0; i < originSinkBlob->GetDesc().BlobSize(); ++i ) {
		EXPECT_TRUE( FloatEq(
			sinkBlob->GetData().GetValueAt( i ),
			originSinkBlob->GetData().GetValueAt( i ) ) );
	}
}

static void testConsistenceImpl( int epochs, int inputSize, int outputSize,
	int rank, float alpha, float dropout, bool verbose )
{
	CRandom rand( 42 );

	// Train the sipmlest task: the only 1 BLOB for test and train and the only 1 "right labels" for it.
	// Try to make the LoRA outputs right answers for this BLOB.
	CDnn dnn( rand, MathEngine() );

	// Create the mininal net, consists: input data, loraFullyConnected, «right labels», and any loss.
	buildDnn( dnn, outputSize );
	// Tune inputs that way, so no nan/infinity values would be appeared.
	setDnnSpecialInputs( dnn, outputSize, inputSize );

	CPtr<CLoraFullyConnectedLayer> fullLayer = CheckCast<CLoraFullyConnectedLayer>( dnn.GetLayer( "full" ) );
	CPtr<CLossLayer> lossLayer = CheckCast<CLossLayer>( dnn.GetLayer( "loss" ) );
	CPtr<CSinkLayer> sinkLayer = CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ) );

	// Make forward pass.
	dnn.RunOnce();

	// Several epochs of train + test
	trainAndTest( dnn, epochs, /*verbose*/false );

	// Make another forward pass.
	dnn.RunOnce();

	// Copy the weights and freeTerm of the baseFc and the additianl sink is created to see the values returned by loraFc.
	const CPtr<CDnnBlob> baseFcWightsBlob = fullLayer->Weights()->GetCopy();
	const CPtr<CDnnBlob> baseFcFreeTermBlob = fullLayer->FreeTerms()->GetCopy();
	{
		// Measure the forward pass's loss and output.
		const CPtr<CDnnBlob> sinkOriginBlob = sinkLayer->GetBlob()->GetCopy();
		const float originTestLoss = lossLayer->GetLastLoss();

		// Build the LoRA.
		fullLayer->BuildLoRA( rank, alpha, dropout );
		// Make another forward pass.
		dnn.RunOnce();

		const float loraTestLoss = lossLayer->GetLastLoss();
		if( verbose ) {
			printf( "\ntestLoss = %10f\tloraBuildTestLoss = %10f\n", originTestLoss, loraTestLoss );
			printStatus( originTestLoss, sinkOriginBlob );
		}
		// Check, that loss and output are exactly the same.
		checkStatus( originTestLoss, loraTestLoss, sinkOriginBlob, sinkLayer->GetBlob() );
	}

	// Several epochs of train + test
	trainAndTest( dnn, epochs, verbose );

	{ // Check, that weights and freeterm of the baseFc are not changed while the training (it has been stored earlier).
		auto loraBaseFcWeights = fullLayer->Weights()->GetData();
		for( int i = 0; i < baseFcWightsBlob->GetDesc().BlobSize(); ++i ) {
			EXPECT_TRUE( FloatEq( baseFcWightsBlob->GetData().GetValueAt( i ), loraBaseFcWeights.GetValueAt( i ), 1e-5f ) );
		}
		auto loraBaseFcFreeTerm = fullLayer->FreeTerms()->GetData();
		for( int i = 0; i < baseFcFreeTermBlob->GetDesc().BlobSize(); ++i ) {
			EXPECT_TRUE( FloatEq( baseFcFreeTermBlob->GetData().GetValueAt( i ), loraBaseFcFreeTerm.GetValueAt( i ), 1e-5f ) );
		}
	}

	{ // After training measure test-loss and save the output
		dnn.RunOnce();

		const float currentTestLoss = lossLayer->GetLastLoss();
		const auto currentSinkBlob = sinkLayer->GetBlob()->GetCopy();

		// Merge the LoRA.
		fullLayer->MergeWeightsLoRA();
		// Make another forward pass.
		dnn.RunOnce();

		const float loraMergeTestLoss = lossLayer->GetLastLoss();
		if( verbose ) {
			printf( "\ntestLoss = %10f\tloraMergeTestLoss = %10f\n", currentTestLoss, loraMergeTestLoss );
			printStatus( loraMergeTestLoss, sinkLayer->GetBlob() );
		}
		// Check, that test-loss and output are not changed after the LoRA merging.
		checkStatus( currentTestLoss, loraMergeTestLoss, currentSinkBlob, sinkLayer->GetBlob() );
	}
}

} // namespace NeoMLTest

//---------------------------------------------------------------------------------------------------------------------------------------

TEST( LoraFullyConnectedLayerTest, ConsistenceCheck )
{
	( void ) MathEngine(); // Creates it, if not exist

	const bool verbose = false;
	const int inputSize = 100;
	const int outputSize = 10;
	const int epochs = 4;

	// Check all cases of different scales and dropouts
	for( float dropout = 0; dropout < 1; dropout += 0.11 ) {
		for( int rank = 1; rank <= outputSize / 2; ++rank ) {
			for( float alpha = 1.f; alpha <= rank; ++alpha ) {

				if( verbose ) {
					printf( "\n===============================================================\n"
						"rank=%d, alpha=%.2f, dropout=%.2f, inputSize=%d, outputSize=%d, epochs=%d\n",
						rank, alpha, dropout, inputSize, outputSize, epochs );
				}

				testConsistenceImpl( epochs, inputSize, outputSize, rank, alpha, dropout, verbose );
			}
		}
	}
}

//---------------------------------------------------------------------------------------------------------------------------------------

namespace NeoMLTest {

static CPtr<CDnnBlob> generateLoraFcBLob( int objectCount, int objectSize )
{
	CPtr<CDnnBlob> blob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, objectCount, objectSize );
	CRandom random( ( blob->GetDataSize() << 6 ) + objectCount * 17 - objectSize * 31 );
	CREATE_FILL_FLOAT_ARRAY( rawData, -1, 1, blob->GetDataSize(), random );
	blob->CopyFrom( rawData.GetPtr() );
	return blob;
}

static void setLoraFcInputs( CDnn& dnn, int inputSize, int outputSize )
{
	CArray<float> inArr;
	inArr.Add( 0.01f, inputSize );
	CPtr<CDnnBlob> in = CDnnBlob::CreateTensor( dnn.GetMathEngine(), CT_Float, { 1, 1, 1, 1, 1, 1, inputSize } );
	in->CopyFrom( inArr.GetPtr() );

	CArray<float> labelArr;
	labelArr.Add( 1.f, outputSize );
	CPtr<CDnnBlob> labels = CDnnBlob::CreateTensor( dnn.GetMathEngine(), CT_Float, { 1, 1, 1, 1, 1, 1, outputSize } );
	labels->CopyFrom( labelArr.GetPtr() );

	CheckCast<CSourceLayer>( dnn.GetLayer( "in" ) )->SetBlob( in );
	CheckCast<CSourceLayer>( dnn.GetLayer( "label" ) )->SetBlob( labels );
}

static void buildLoraTestDnn( CBaseLayer& coreLayer )
{
	CDnn& dnn = *coreLayer.GetDnn();
	IMathEngine& mathEngine = dnn.GetMathEngine();
	CPtr<CSourceLayer> dataLayer = new CSourceLayer( mathEngine );
	dataLayer->SetName( "in" );
	dataLayer->StoreBlob( true );
	dnn.AddLayer( *dataLayer );

	coreLayer.Connect( *dataLayer );

	CPtr<CSourceLayer> label = new CSourceLayer( mathEngine );
	label->SetName( "label" );
	label->StoreBlob( true );
	dnn.AddLayer( *label );

	CPtr<CEuclideanLossLayer> loss = new CEuclideanLossLayer( mathEngine );
	loss->SetName( "loss" );
	loss->Connect( 0, coreLayer );
	loss->Connect( 1, *label );
	dnn.AddLayer( *loss );

	CPtr<CSinkLayer> out = new CSinkLayer( mathEngine );
	out->SetName( "sink" );
	out->Connect( coreLayer );
	dnn.AddLayer( *out );

	CPtr<CDnnSolver> solver = new CDnnAdaptiveGradientSolver( mathEngine );
	dnn.SetSolver( solver.Ptr() );
}

static void buildLoraFcDnn( CDnn& dnn, int inputSize, int outputSize, float dropout )
{
	CPtr<CDnnBlob> baseFcWeights = generateLoraFcBLob( outputSize, inputSize );
	CPtr<CDnnBlob> baseFcFreeTerms = generateLoraFcBLob( 1, outputSize );

	CPtr<CLoraFullyConnectedLayer> full = new CLoraFullyConnectedLayer( MathEngine(), "full" );
	full->SetNumberOfElements( baseFcWeights->GetObjectCount() );
	full->Weights() = baseFcWeights; // No copy
	full->FreeTerms() = baseFcFreeTerms; // No copy
	full->BuildLoRA( 4, 8.f, dropout ); // dropout must be off, otherwise math won't match
	dnn.AddLayer( *full );

	buildLoraTestDnn( *full );
	setLoraFcInputs( dnn, inputSize, outputSize );
}

} // namespace NeoMLTest

TEST( LoraFullyConnectedLayerTest, Initialization )
{
	const int inputSize = 6;
	const int outputSize = 8;

	CPtr<CDnnBlob> baseFcWeights = generateLoraFcBLob( outputSize, inputSize );
	CPtr<CLoraFullyConnectedLayer> loraFc = new CLoraFullyConnectedLayer( MathEngine(), "full" );
	loraFc->SetNumberOfElements( baseFcWeights->GetObjectCount() );
	loraFc->Weights() = baseFcWeights; // No copy
	loraFc->BuildLoRA( 4, 8.f, 0.1f );

	// Check that uninitialized A and B are nullptrs
	EXPECT_EQ( nullptr, loraFc->AWeightsLoRA().Ptr() );
	EXPECT_EQ( nullptr, loraFc->BWeightsLoRA().Ptr() );

	// Check that baseFc weights were taken by loraFc without copying
	EXPECT_EQ( baseFcWeights.Ptr(), loraFc->Weights().Ptr() );

	baseFcWeights = baseFcWeights->GetCopy();
	loraFc->MergeWeightsLoRA();
	// Check that in unitialized state merge/split doesn't cause any changes
	EXPECT_TRUE( CompareBlobs( *baseFcWeights, *loraFc->Weights(), FLT_EPSILON ) );
	loraFc->DestroyUnmergedLoRA();
	EXPECT_TRUE( CompareBlobs( *baseFcWeights, *loraFc->Weights(), FLT_EPSILON ) );

	// Copy through serialization
	{
		CMemoryFile file;
		CPtr<CBaseLayer> baseLoraFc = loraFc.Ptr();
		loraFc.Release();
		{
			CArchive archive( &file, CArchive::store );
			SerializeLayer( archive, MathEngine(), baseLoraFc );
		}
		baseLoraFc.Release();
		file.SeekToBegin();
		{
			CArchive archive( &file, CArchive::load );
			SerializeLayer( archive, MathEngine(), baseLoraFc );
		}
		loraFc = CheckCast<CLoraFullyConnectedLayer>( baseLoraFc );
	}

	// Check that uninitialized A and B are nullptrs
	EXPECT_EQ( nullptr, loraFc->AWeightsLoRA().Ptr() );
	EXPECT_EQ( nullptr, loraFc->BWeightsLoRA().Ptr() );
	// After copying through serialization check only blob contents
	EXPECT_TRUE( CompareBlobs( *baseFcWeights, *loraFc->Weights(), FLT_EPSILON ) );
}

TEST( LoraFullyConnectedLayerTest, InferenceAndLearning )
{
	constexpr int inputSize = 32;
	constexpr int outputSize = 16;

	CRandom random( 0x645 );
	CDnn dnn( random, MathEngine() );
	buildLoraFcDnn( dnn, inputSize, outputSize, /*dropout*/0.f );
	setLoraFcInputs( dnn, inputSize, outputSize );

	CPtr<CLoraFullyConnectedLayer> loraFc = CheckCast<CLoraFullyConnectedLayer>( dnn.GetLayer( "full" ) );
	CDnnBlob* baseWeightsPtr = loraFc->Weights().Ptr();
	CPtr<CDnnBlob> copyBaseWeights = baseWeightsPtr->GetCopy();

	// Now merge to base weights before RunOnce
	loraFc->MergeWeightsLoRA();
	// Check that weights of A and B are unintialized
	EXPECT_EQ( NULL, loraFc->AWeightsLoRA().Ptr() );
	EXPECT_EQ( NULL, loraFc->BWeightsLoRA().Ptr() );

	for( int iteration = 0; iteration < 5; ++iteration ) {
		// Check that layer after merge works
		dnn.RunOnce();
		// Check that no reallocation occured
		EXPECT_EQ( baseWeightsPtr, loraFc->Weights().Ptr() );

		// Store the output
		CPtr<CSinkLayer> sink = CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ) );
		CPtr<CDnnBlob> copyOfOutput = sink->GetBlob()->GetCopy();

		loraFc->BuildLoRA( 4, 8.f, /*dropout*/0.f );
		dnn.RunAndLearnOnce();

		// Check that previously uninitialized weights were initialized
		EXPECT_NE( nullptr, loraFc->AWeightsLoRA().Ptr() );
		EXPECT_NE( nullptr, loraFc->BWeightsLoRA().Ptr() );
		// Check that during training layer's original weights are untouched
		EXPECT_TRUE( CompareBlobs( *baseWeightsPtr, *loraFc->Weights() ) );
		// Check that no reallocation
		EXPECT_EQ( baseWeightsPtr, loraFc->Weights().Ptr() );
		// Check that output during first iteration of training matches last output before training
		EXPECT_TRUE( loraFc->GetDropoutRateLoRA() <= 0.f || CompareBlobs( *copyOfOutput, *sink->GetBlob() ) );

		// Now merge to base weights before RunOnce again
		loraFc->MergeWeightsLoRA();
		// The base weights must be at the same location and size
		EXPECT_EQ( baseWeightsPtr, loraFc->Weights().Ptr() );
		EXPECT_TRUE( copyBaseWeights->HasEqualDimensions( loraFc->Weights().Ptr() ) );
		// The base weights must contain different (merged) weights
		EXPECT_FALSE( CompareBlobs( *copyBaseWeights, *loraFc->Weights() ) );
		copyBaseWeights = baseWeightsPtr->GetCopy();
	}
}

//----------------------------------------------------------------------------------------------------------------------

namespace NeoMLTest {

static void loraFcSerializerTestImpl( bool initialize, bool discardBeforeLoad )
{
	// Setting bigger size (full weights will be ioSize x ioSize matrix)
	constexpr int ioSize = 100;
	constexpr __int64 fullMatrixSize = static_cast<__int64>( sizeof( float ) * ioSize * ioSize );

	CRandom random( 0xABBA );
	CDnn dnn( random, MathEngine() );
	buildLoraFcDnn( dnn, ioSize, ioSize, /*dropout*/0.1f );

	if( initialize ) {
		dnn.RunAndLearnOnce();
	}

	CPtr<CLoraFullyConnectedLayer> loraFc = CheckCast<CLoraFullyConnectedLayer>( dnn.GetLayer( "full" ) );
	CPtr<CDnnBlob> aWeights = loraFc->AWeightsLoRA() == nullptr ? nullptr : loraFc->AWeightsLoRA()->GetCopy();
	CPtr<CDnnBlob> bWeights = loraFc->BWeightsLoRA() == nullptr ? nullptr : loraFc->BWeightsLoRA()->GetCopy();

	CMemoryFile file;
	{
		CArchive archive( &file, CArchive::store );
		ASSERT_EQ( 1, CLoraSerializer::Serialize( dnn, archive ) );
	}
	// Check that we didn't serialize full matrix
	ASSERT_GT( fullMatrixSize / 2, file.GetLength() );

	// Let's change weights and roll back to those from serialization
	dnn.RunAndLearnOnce();

	if( initialize ) {
		// check that weights changed after last RunAndLearnOnce
		EXPECT_FALSE( CompareBlobs( *aWeights, *loraFc->AWeightsLoRA(), FLT_EPSILON ) );
		EXPECT_FALSE( CompareBlobs( *bWeights, *loraFc->BWeightsLoRA(), FLT_EPSILON ) );
	}

	if( discardBeforeLoad ) {
		// Destroy previous data
		loraFc->DestroyUnmergedLoRA();
	}

	file.SeekToBegin();
	{
		CArchive archive( &file, CArchive::SD_Loading );
		ASSERT_EQ( 1, CLoraSerializer::Serialize( dnn, archive ) );
	}

	if( initialize ) {
		EXPECT_TRUE( loraFc->GetRankLoRA() == 4 );
		// Check that after serialization A and B matrices were rolled back
		EXPECT_TRUE( CompareBlobs( *aWeights, *loraFc->AWeightsLoRA() ) );
		EXPECT_TRUE( CompareBlobs( *bWeights, *loraFc->BWeightsLoRA() ) );
	} else {
		EXPECT_EQ( nullptr, loraFc->AWeightsLoRA().Ptr() );
		EXPECT_EQ( nullptr, loraFc->BWeightsLoRA().Ptr() );
	}
}

} // namespace NeoMLTest

TEST( LoraFullyConnectedLayerSerializerTest, LoraFc )
{
	for( bool initialize : { true, false } ) {
		for( bool discardBeforeLora : { true, false } ) {
			loraFcSerializerTestImpl( initialize, discardBeforeLora );
		}
	}
}
