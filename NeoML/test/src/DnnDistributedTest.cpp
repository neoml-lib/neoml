/* Copyright © 2021-2024 ABBYY

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

#include <memory>

#include <TestFixture.h>

using namespace NeoML;
using namespace NeoMLTest;

namespace NeoMLTest {

class CCustomDataset : public IDistributedDataset {
public:
	CCustomDataset( int _inputSize, int _labelSize )
		: inputSize( _inputSize ), labelSize( _labelSize ) {}

	int SetInputBatch( CDnn& dnn, int ) override
	{
		CArray<float> inArr;
		inArr.Add( 1, inputSize );
		CPtr<CDnnBlob> in = CDnnBlob::CreateTensor( dnn.GetMathEngine(), CT_Float, { 1, 1, 1, 1, 1, 1, inputSize } );
		in->CopyFrom( inArr.GetPtr() );
		CArray<float> labelArr;
		labelArr.Add( 1, labelSize );
		CPtr<CDnnBlob> labels = CDnnBlob::CreateTensor( dnn.GetMathEngine(), CT_Float, { 1, 1, 1, 1, 1, 1, labelSize } );
		labels->CopyFrom( labelArr.GetPtr() );
		CheckCast<CSourceLayer>( dnn.GetLayer( "data" ) )->SetBlob( in );
		CheckCast<CSourceLayer>( dnn.GetLayer( "label" ) )->SetBlob( labels );
		return 1;
	}

private:
	const int inputSize;
	const int labelSize;
};

static void buildDnn( CDnn& dnn, int outputSize )
{
	CPtr<CSourceLayer> data = Source( dnn, "data" );
	CPtr<CSourceLayer> label = Source( dnn, "label" );

	CPtr<CFullyConnectedLayer> full = FullyConnected( outputSize, /*freeTerm*/true )( "full", data.Ptr() );
	CPtr<CEuclideanLossLayer> loss = EuclideanLoss()( "loss", full.Ptr(), label.Ptr() );

	( void ) Sink( full.Ptr(), "sink" );

	CPtr<CDnnSolver> solver = new CDnnAdaptiveGradientSolver( dnn.GetMathEngine() );
	dnn.SetSolver( solver.Ptr() );
}

static constexpr int inputSize = 1000;
static constexpr int outputSize = 5;

} // namespace NeoMLTest

//---------------------------------------------------------------------------------------------------------------------

TEST( CDnnDistributedTest, DnnDistributedNoArchiveTest )
{
	std::unique_ptr<IMathEngine> mathEngine( CreateCpuMathEngine( /*memoryLimit*/0u ) );
	CRandom rand( 42 );

	CDnn dnn( rand, *mathEngine );
	buildDnn( dnn, outputSize );

	CDistributedTraining distributed( dnn, 2 );
	CCustomDataset dataset( inputSize, outputSize );
	distributed.RunOnce( dataset );
	distributed.RunAndLearnOnce( dataset );

	CObjectArray<CDnnBlob> blobs;
	distributed.GetLastBlob( "sink", blobs );
	EXPECT_EQ( outputSize, blobs[0]->GetDataSize() );
	for( int i = 1; i < 2; i++ ) {
		EXPECT_EQ( outputSize, blobs[i]->GetDataSize() );
		EXPECT_TRUE( CompareBlobs( *( blobs[0] ), *( blobs[i] ) ) );
	}

	CArray<float> losses;
	distributed.GetLastLoss( "loss", losses );
	EXPECT_EQ( 2, losses.Size() );
	EXPECT_EQ( losses[0], losses[1] );
	EXPECT_EQ( 2, distributed.GetModelCount() );
}

TEST( CDnnDistributedTest, DnnDistributedArchiveTest )
{
	std::unique_ptr<IMathEngine> mathEngine( CreateCpuMathEngine( /*memoryLimit*/0u ) );

	CRandom rand( 42 );
	CDnn dnn( rand, *mathEngine );
	buildDnn( dnn, outputSize );

	CCustomDataset dataset( inputSize, outputSize );
	CString archiveName = "distributed";
	{
		CDistributedTraining distributed( dnn, 2 );
		distributed.RunAndLearnOnce( dataset );

		CArray<float> losses;
		distributed.GetLastLoss( "loss", losses );
		EXPECT_EQ( 2, losses.Size() );
		EXPECT_EQ( losses[0], losses[1] );

		distributed.RunOnce( dataset );

		CObjectArray<CDnnBlob> blobs;
		distributed.GetLastBlob( "sink", blobs );
		for( int i = 1; i < blobs.Size(); ++i ) {
			EXPECT_TRUE( CompareBlobs( *( blobs[0] ), *( blobs[i] ) ) );
		}

		{ // store trained dnn also to check distributed inference
			CArchiveFile file( archiveName, CArchive::store, GetPlatformEnv() );
			CArchive archive( &file, CArchive::store );
			distributed.Serialize( archive );
		}
		{ // store trained output to check distributed inference
			CArchiveFile out_file( archiveName + ".out", CArchive::store, GetPlatformEnv() );
			CArchive archive( &out_file, CArchive::store );
			SerializeBlob( *mathEngine, archive, blobs[0] );
		}
	}

	CArchiveFile archiveFile( archiveName, CArchive::load, GetPlatformEnv() );
	CArchive archive( &archiveFile, CArchive::load );
	CDistributedTraining distributed( archive, 2 );
	EXPECT_EQ( 2, distributed.GetModelCount() );
	archive.Close();
	archiveFile.Close();

	CString archiveSolverName = "distributed.solver";
	{
		CPtr<CDnnSolver> solver = new CDnnAdaptiveGradientSolver( dnn.GetMathEngine() );
		dnn.SetSolver( solver.Ptr() );

		CArchiveFile storeFile( archiveSolverName, CArchive::store, GetPlatformEnv() );
		CArchive storeArchive( &storeFile, CArchive::store );
		SerializeSolver( storeArchive, dnn, solver );
	}
	{
		CArchiveFile loadFile( archiveSolverName, CArchive::load, GetPlatformEnv() );
		CArchive loadArchive( &loadFile, CArchive::load );
		distributed.SetSolver( loadArchive );
	}

	distributed.RunAndBackwardOnce( dataset );
	distributed.Train();

	CArray<float> losses;
	distributed.GetLastLoss( "loss", losses );
	EXPECT_EQ( 2, losses.Size() );
	EXPECT_EQ( losses[0], losses[1] );
}

TEST( CDnnDistributedTest, DnnDistributedSerializeTest )
{
	std::unique_ptr<IMathEngine> mathEngine( CreateCpuMathEngine( /*memoryLimit*/0u ) );

	CRandom rand( 42 );
	CDnn dnn( rand, *mathEngine );
	buildDnn( dnn, outputSize );

	CDistributedTraining distributed( dnn, 3 );
	CCustomDataset dataset( inputSize, outputSize );
	distributed.RunAndLearnOnce( dataset );
	distributed.RunOnce( dataset );

	CArray<float> losses;
	distributed.GetLastLoss( "loss", losses );

	CString archiveName = "distributedSerialized";
	{
		CArchiveFile archiveFile( archiveName, CArchive::store, GetPlatformEnv() );
		CArchive archive( &archiveFile, CArchive::SD_Storing );
		distributed.Serialize( archive );
	}

	CRandom rand2( 42 );
	CDnn serializedCnn( rand2, *mathEngine );
	{
		CArchiveFile archiveFile( archiveName, CArchive::load, GetPlatformEnv() );
		CArchive archive( &archiveFile, CArchive::SD_Loading );
		serializedCnn.Serialize( archive );
	}

	dataset.SetInputBatch( serializedCnn, 0 );
	serializedCnn.RunOnce();
	float loss = static_cast< CLossLayer* >( serializedCnn.GetLayer( "loss" ).Ptr() )->GetLastLoss();
	EXPECT_EQ( loss, losses[0] );

	CArray<float> distributedWeights;
	CPtr<CDnnBlob> weightsBlob = static_cast< CFullyConnectedLayer* >( serializedCnn.GetLayer( "full" ).Ptr() )->GetWeightsData();
	distributedWeights.SetSize( weightsBlob->GetDataSize() );
	weightsBlob->CopyTo( distributedWeights.GetPtr() );

	dataset.SetInputBatch( dnn, 0 );
	dnn.RunAndLearnOnce();
	CArray<float> weights;
	weightsBlob = static_cast< CFullyConnectedLayer* >( dnn.GetLayer( "full" ).Ptr() )->GetWeightsData();
	weights.SetSize( weightsBlob->GetDataSize() );
	weightsBlob->CopyTo( weights.GetPtr() );

	EXPECT_EQ( weights.Size(), distributedWeights.Size() );
	for( int i = 0; i < weights.Size(); i++ ) {
		EXPECT_NEAR( weights[i], distributedWeights[i], 1e-4 );
	}
}

TEST( CDnnDistributedTest, DnnDistributedAutoThreadCountTest )
{
	std::unique_ptr<IMathEngine> mathEngine( CreateCpuMathEngine( /*memoryLimit*/0u ) );
	CRandom rand( 42 );

	CDnn dnn( rand, *mathEngine );
	buildDnn( dnn, outputSize );

	CDistributedTraining distributed( dnn, 0 );
	GTEST_LOG_( INFO ) << "Distributed default thread count is " << distributed.GetModelCount();
	EXPECT_LT( 0, distributed.GetModelCount() );
	EXPECT_EQ( GetAvailableCpuCores(), distributed.GetModelCount() );
}

//---------------------------------------------------------------------------------------------------------------------

TEST( CDnnDistributedTest, DnnDistributedInferenceArchived )
{
	IMathEngine& mathEngine = MathEngine();
	if( mathEngine.GetType() != MET_Cpu ) {
		GTEST_LOG_( INFO ) << "Skipped for mathEngine type != MET_Cpu";
		return;
	}

	CString archiveName = "distributed";
	CCustomDataset dataset( inputSize, outputSize );

	CRandom random( 42 );
	CDnn dnn( random, MathEngine() );

	CPtr<CDnnBlob> expected;
	{ // Check the dnn is stored in the file valid
		{
			CArchiveFile file( archiveName, CArchive::load, GetPlatformEnv() );
			CArchive archive( &file, CArchive::load );
			dnn.Serialize( archive );

			dataset.SetInputBatch( dnn, 0 );
			dnn.RunOnce();
		}
		CPtr<CDnnBlob> blob = CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ) )->GetBlob();
		{
			CArchiveFile out( archiveName + ".out", CArchive::load, GetPlatformEnv() );
			CArchive archive( &out, CArchive::load );
			SerializeBlob( mathEngine, archive, expected );
		}
		EXPECT_TRUE( CompareBlobs( *blob, *expected ) );
	}

	{ // Check dnn constructor
		CDistributedInference distributed( dnn, /*count*/0, dataset );
		EXPECT_LT( 0, distributed.GetModelCount() );
		EXPECT_EQ( GetAvailableCpuCores(), distributed.GetModelCount() );

		distributed.RunOnce( dataset );

		CObjectArray<CDnnBlob> blobs;
		distributed.GetLastBlob( "sink", blobs );
		for( int i = 0; i < blobs.Size(); ++i ) {
			EXPECT_TRUE( CompareBlobs( *( blobs[i] ), *expected ) );
		}

		distributed.RunOnce( dataset );

		distributed.GetLastBlob( "sink", blobs );
		for( int i = 0; i < blobs.Size(); ++i ) {
			EXPECT_TRUE( CompareBlobs( *( blobs[i] ), *expected ) );
		}
	}

	{ // Check archive constructor
		CArchiveFile file( archiveName, CArchive::load, GetPlatformEnv() );
		CArchive archive( &file, CArchive::load );
		CDistributedInference distributed( mathEngine, archive, /*count*/4, dataset, /*seed*/42 );
		EXPECT_EQ( 4, distributed.GetModelCount() );

		distributed.RunOnce( dataset );

		CObjectArray<CDnnBlob> blobs;
		distributed.GetLastBlob( "sink", blobs );
		for( int i = 0; i < blobs.Size(); ++i ) {
			EXPECT_TRUE( CompareBlobs( *( blobs[i] ), *expected ) );
		}
	}
}
