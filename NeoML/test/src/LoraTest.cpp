/* Copyright © 2023-2024 ABBYY

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

//----------------------------------------------------------------------------------------------------------------------

namespace NeoMLTest {

static CPtr<CDnnBlob> generateLoraFcBLob( IMathEngine& mathEngine, int objectCount, int objectSize )
{
	CPtr<CDnnBlob> blob = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, 1, objectCount, objectSize );
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

	CPtr<CDnnAdaptiveGradientSolver> solver = new CDnnAdaptiveGradientSolver( mathEngine );
	solver->SetLearningRate( 0.1f );
	dnn.SetSolver( solver.Ptr() );
}

static void buildLoraFcDnn( CDnn& dnn, int inputSize, int outputSize, float dropout )
{
	CLoraParams params( 4, 2.f, dropout ); // dropout must be off otherwise math won't match
	CPtr<CDnnBlob> baseFcWeights = generateLoraFcBLob( dnn.GetMathEngine(), outputSize, inputSize );
	CPtr<CDnnBlob> baseFcFreeTerms = generateLoraFcBLob( dnn.GetMathEngine(), 1, outputSize );

	CPtr<CLoraFullyConnectedLayer> full = new CLoraFullyConnectedLayer( *baseFcWeights, baseFcFreeTerms.Ptr(), params );
	full->SetName( "full" );
	dnn.AddLayer( *full );

	buildLoraTestDnn( *full );

	setLoraFcInputs( dnn, inputSize, outputSize );
}

static void buildLoraTransformerDnn( CDnn& dnn, int inputSize )
{
	CPtr<CTransformerEncoderLayer> transformer = new CTransformerEncoderLayer( dnn.GetMathEngine() );
	transformer->SetName( "transformer" );
	transformer->SetHiddenSize( 32 );
	transformer->SetHeadCount( 4 );
	transformer->SetFeedForwardSize( 64 );
	dnn.AddLayer( *transformer );
	buildLoraTestDnn( *transformer );

	setLoraFcInputs( dnn, inputSize, inputSize );
}

template<class T>
static bool checkLayerClass( CDnnLayerGraph& graph, std::initializer_list<const char*> layerPath )
{
	NeoAssert( layerPath.size() > 0 );

	auto currLayer = layerPath.begin();
	CDnnLayerGraph* currGraph = &graph;
	for( size_t i = 0; i < layerPath.size() - 1; ++i, ++currLayer ) {
		currGraph = dynamic_cast<CCompositeLayer*>( currGraph->GetLayer( *currLayer ).Ptr() );
	}

	return ( dynamic_cast<T*>( currGraph->GetLayer( *currLayer ).Ptr() ) != nullptr );
}

} // namespace NeoMLTest

TEST( LoraFullyConnectedLayerTest, Initialization )
{
	const int inputSize = 6;
	const int outputSize = 8;
	CLoraParams params( 4, 2.f, 0.1f );

	CPtr<CDnnBlob> baseFcWeights = generateLoraFcBLob( MathEngine(), outputSize, inputSize );
	CPtr<CLoraFullyConnectedLayer> loraFc = new CLoraFullyConnectedLayer( *baseFcWeights, nullptr, params );

	// Check that uninitialized A and B are nullptrs
	EXPECT_EQ( nullptr, loraFc->GetAWeightsNoCopy().Ptr() );
	EXPECT_EQ( nullptr, loraFc->GetBWeightsNoCopy().Ptr() );

	// Check that baseFc weights were taken by loraFc without copying
	EXPECT_EQ( baseFcWeights.Ptr(), loraFc->GetWeightsNoCopy().Ptr() );

	baseFcWeights = baseFcWeights->GetCopy();
	// Check that in unitialized state merge/split doesn't cause any changes
	EXPECT_TRUE( CompareBlobs( *baseFcWeights, *loraFc->GetSplitWeightsNoCopy(), FLT_EPSILON ) );
	EXPECT_TRUE( CompareBlobs( *baseFcWeights, *loraFc->GetMergedWeightsNoCopy(), FLT_EPSILON ) );

	// Copy through serialization
	{
		CMemoryFile file;
		CPtr<CBaseLayer> baseLoraFc = loraFc.Ptr();
		loraFc.Release();
		{
			CArchive archive( &file, CArchive::SD_Storing );
			SerializeLayer( archive, MathEngine(), baseLoraFc );
		}
		baseLoraFc.Release();
		file.SeekToBegin();
		{
			CArchive archive( &file, CArchive::SD_Loading );
			SerializeLayer( archive, MathEngine(), baseLoraFc );
		}
		loraFc = CheckCast<CLoraFullyConnectedLayer>( baseLoraFc );
	}

	// Check that uninitialized A and B are nullptrs
	EXPECT_EQ( nullptr, loraFc->GetAWeightsNoCopy().Ptr() );
	EXPECT_EQ( nullptr, loraFc->GetBWeightsNoCopy().Ptr() );
	// After copying through serialization check only blob contents
	EXPECT_TRUE( CompareBlobs( *baseFcWeights, *loraFc->GetWeightsNoCopy(), FLT_EPSILON ) );
}

TEST( LoraFullyConnectedLayerTest, InferenceAndLearning )
{
	const auto met = MathEngine().GetType();
	if(met != MET_Cpu && met != MET_Cuda) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		// CEuclideanLossLayer -- > VectorHuber
		return;
	}

	constexpr int inputSize = 32;
	constexpr int outputSize = 16;

	CRandom random( 0x645 );
	CDnn dnn( random, MathEngine() );
	buildLoraFcDnn( dnn, inputSize, outputSize, 0.f );
	setLoraFcInputs( dnn, inputSize, outputSize );

	CPtr<CLoraFullyConnectedLayer> loraFc = CheckCast<CLoraFullyConnectedLayer>( dnn.GetLayer( "full" ) );
	CDnnBlob* originalWeightsPtr = loraFc->GetWeightsNoCopy().Ptr();
	CPtr<CDnnBlob> copyOfOriginalWeights = originalWeightsPtr->GetCopy();

	for( int iteration = 0; iteration < 5; ++iteration ) {
		dnn.RunOnce();
		// Check that layer is in the merged state
		EXPECT_TRUE( loraFc->IsMerged() );
		// Check that no reallocation occured
		EXPECT_EQ( originalWeightsPtr, loraFc->GetWeightsNoCopy().Ptr() );

		// Store the output
		CPtr<CSinkLayer> sink = CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ) );
		CPtr<CDnnBlob> copyOfOutput = sink->GetBlob()->GetCopy();

		dnn.RunAndLearnOnce();

		// Check that previously uninitialized weights were initialized
		EXPECT_NE( nullptr, loraFc->GetAWeightsNoCopy().Ptr() );
		EXPECT_NE( nullptr, loraFc->GetBWeightsNoCopy().Ptr() );
		// Check that during training layer is in split state and original weights are untouched
		EXPECT_FALSE( loraFc->IsMerged() );
		EXPECT_TRUE( CompareBlobs( *copyOfOriginalWeights, *loraFc->GetWeightsNoCopy() ) );
		// Check that no reallocation occured
		EXPECT_EQ( originalWeightsPtr, loraFc->GetWeightsNoCopy().Ptr() );
		// Check that output during first iteration of training matches last output before training
		EXPECT_TRUE( loraFc->Dropout() <= 0.f || CompareBlobs( *copyOfOutput, *sink->GetBlob() ) );

		// Now run the net, the layer must switch to merged state
		dnn.RunOnce();

		// Check that layer is in the merged state
		EXPECT_TRUE( loraFc->IsMerged() );
		// The base weights must be at the same location but must contain different (merged) weights
		EXPECT_EQ( originalWeightsPtr, loraFc->GetWeightsNoCopy().Ptr() );
		EXPECT_TRUE( copyOfOriginalWeights->HasEqualDimensions( loraFc->GetWeightsNoCopy().Ptr() ) );
		EXPECT_FALSE( CompareBlobs( *copyOfOriginalWeights, *loraFc->GetWeightsNoCopy() ) );
	}
}

//----------------------------------------------------------------------------------------------------------------------

TEST( LoraBuilderTest, DefaultTransformer )
{
	constexpr int ioSize = 48;

	CRandom random( 0xABBA );
	CDnn dnn( random, MathEngine() );
	buildLoraTransformerDnn( dnn, ioSize );

	dnn.RunOnce();

	CPtr<CSinkLayer> sink = CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ) );
	CPtr<CDnnBlob> expectedOutput = sink->GetBlob()->GetCopy();

	CLoraBuilder builder;
	CLoraParams params( 4, 2.f, 0.1f );
	// 2 fc's inside transformer directly
	// 4 fc's inside of attention (inside transformer)
	EXPECT_EQ( 6, builder.BuildAllFcWrappers( dnn, params ) );
	// Let's check a couple of layers (dirty hacks for names...)
	EXPECT_TRUE( checkLayerClass<CLoraFullyConnectedLayer>( dnn, { "transformer", "FullyConnected2" } ) );
	EXPECT_TRUE( checkLayerClass<CLoraFullyConnectedLayer>( dnn, { "transformer", "SelfAttention", "K" } ) );

	dnn.RunOnce();
	EXPECT_TRUE( CompareBlobs( *expectedOutput, *sink->GetBlob() ) );
}

TEST( LoraBuilderTest, BuildNoRecurrentReplacement )
{
	constexpr int ioSize = 48;

	CRandom random( 0xABBA );
	CDnn dnn( random, MathEngine() );
	buildLoraTransformerDnn( dnn, ioSize );

	dnn.RunOnce();

	CPtr<CSinkLayer> sink = CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ) );
	CPtr<CDnnBlob> expectedOutput = sink->GetBlob()->GetCopy();

	CArray<CString> emptyClasses;
	CLoraBuilder builder( emptyClasses ); // Don't allow recursively replace fcs inside any subgraphs (composites)
	CLoraParams params( 4, 2.f, 0.1f );
	// no fc's inside of CDnn directly
	EXPECT_EQ( 0, builder.BuildAllFcWrappers( dnn, params ) );
	// 2 fc's inside transformer
	// 4 fc's inside of attention won't be replaced because of class restrictions
	CPtr<CTransformerEncoderLayer> transformer = CheckCast<CTransformerEncoderLayer>( dnn.GetLayer( "transformer" ) );
	EXPECT_EQ( 2, builder.BuildAllFcWrappers( *transformer, params ) );
	// Let's check a couple of layers (dirty hacks for names...)
	EXPECT_TRUE( checkLayerClass<CLoraFullyConnectedLayer>( dnn, { "transformer", "FullyConnected2" } ) );
	EXPECT_TRUE( checkLayerClass<CFullyConnectedLayer>( dnn, { "transformer", "SelfAttention", "K" } ) );

	dnn.RunOnce();
	EXPECT_TRUE( CompareBlobs( *expectedOutput, *sink->GetBlob() ) );
}

TEST( LoraBuilderTest, MergeAndDiscardTest )
{
	const auto met = MathEngine().GetType();
	if(met != MET_Cpu && met != MET_Cuda) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		// dropout
		return;
	}

	constexpr int ioSize = 12;

	CRandom random( 0xACCA );
	CDnn dnn( random, MathEngine () );
	buildLoraTransformerDnn( dnn, ioSize );
	dnn.RunOnce();

	EXPECT_TRUE( checkLayerClass<CFullyConnectedLayer>( dnn, { "transformer", "FullyConnected2" } ) );
	EXPECT_TRUE( checkLayerClass<CFullyConnectedLayer>( dnn, { "transformer", "SelfAttention", "K" } ) );
	CPtr<CDnnBlob> untrainedOutput = CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ) )->GetBlob()->GetCopy();
	CPtr<CDnnBlob> initialWeights = nullptr;
	{
		CTransformerEncoderLayer* enc = CheckCast<CTransformerEncoderLayer>( dnn.GetLayer( "transformer" ) );
		initialWeights = CheckCast<CFullyConnectedLayer>( enc->GetLayer( "FullyConnected2" ) )->GetWeightsData();
	}

	CLoraParams params( 8, 2.f, 0.2f );
	CLoraBuilder builder;
	EXPECT_EQ( 6, builder.BuildAllFcWrappers( dnn, params ) );
	EXPECT_TRUE( checkLayerClass<CLoraFullyConnectedLayer>( dnn, { "transformer", "FullyConnected2" } ) );
	EXPECT_TRUE( checkLayerClass<CLoraFullyConnectedLayer>( dnn, { "transformer", "SelfAttention", "K" } ) );
	// Before training the net we need to disable training for all non-LoRA layers
	// There are 2 ObjectNormalization layers inside of transformer
	EXPECT_EQ( 2, builder.DisableNonLoraTraining( dnn ) );

	dnn.RunOnce();
	EXPECT_TRUE( CompareBlobs( *untrainedOutput, *CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ) )->GetBlob() ) );

	dnn.RunAndLearnOnce();
	dnn.RunOnce();
	CPtr<CDnnBlob> trainedOutput = CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ) )->GetBlob()->GetCopy();
	EXPECT_FALSE( CompareBlobs( *untrainedOutput, *trainedOutput ) );

	// Store net with lora wrappers after training iteration
	CMemoryFile trainedNetFile;
	{
		CArchive archive( &trainedNetFile, CArchive::SD_Storing );
		dnn.Serialize( archive );
	}

	EXPECT_EQ( 6, builder.MergeAllFcWrappers( dnn ) );
	dnn.RunOnce();

	EXPECT_TRUE( checkLayerClass<CFullyConnectedLayer>( dnn, { "transformer", "FullyConnected2" } ) );
	EXPECT_TRUE( checkLayerClass<CFullyConnectedLayer>( dnn, { "transformer", "SelfAttention", "K" } ) );
	EXPECT_TRUE( CompareBlobs( *trainedOutput, *CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ) )->GetBlob() ) );

	{
		dnn.DeleteAllLayers();
		trainedNetFile.SeekToBegin();
		CArchive archive( &trainedNetFile, CArchive::SD_Loading );
		dnn.Serialize( archive );
	}

	dnn.RunOnce();
	EXPECT_TRUE( checkLayerClass<CLoraFullyConnectedLayer>( dnn, { "transformer", "FullyConnected2" } ) );
	EXPECT_TRUE( checkLayerClass<CLoraFullyConnectedLayer>( dnn, { "transformer", "SelfAttention", "K" } ) );
	CPtr<CTransformerEncoderLayer> transformer = CheckCast<CTransformerEncoderLayer>( dnn.GetLayer( "transformer" ) );
	CPtr<CDnnBlob> splitWeights = CheckCast<CLoraFullyConnectedLayer>(
		transformer->GetLayer( "FullyConnected2" ) )->GetSplitWeightsNoCopy()->GetCopy();
	EXPECT_TRUE( CompareBlobs( *initialWeights, *splitWeights ) );
	EXPECT_TRUE( CompareBlobs( *trainedOutput, *CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ) )->GetBlob() ) );

	EXPECT_EQ( 6, builder.DiscardAllFcWrappers( dnn ) );
	dnn.RunOnce();
	EXPECT_TRUE( checkLayerClass<CFullyConnectedLayer>( dnn, { "transformer", "FullyConnected2" } ) );
	EXPECT_TRUE( checkLayerClass<CFullyConnectedLayer>( dnn, { "transformer", "SelfAttention", "K" } ) );
	EXPECT_TRUE( CompareBlobs( *untrainedOutput, *CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ) )->GetBlob() ) );
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
	buildLoraFcDnn( dnn, ioSize, ioSize, 0.1f );

	if( initialize ) {
		dnn.RunAndLearnOnce();
	}

	CPtr<CLoraFullyConnectedLayer> loraFc = CheckCast<CLoraFullyConnectedLayer>( dnn.GetLayer( "full" ) );
	CPtr<CDnnBlob> aWeights = loraFc->GetAWeightsNoCopy() == nullptr ? nullptr : loraFc->GetAWeightsNoCopy()->GetCopy();
	CPtr<CDnnBlob> bWeights = loraFc->GetBWeightsNoCopy() == nullptr ? nullptr : loraFc->GetBWeightsNoCopy()->GetCopy();

	CMemoryFile file;
	{
		CArchive archive( &file, CArchive::SD_Storing );
		EXPECT_EQ( 1, CLoraSerializer().Serialize( dnn, archive ) );
	}
	EXPECT_GT( fullMatrixSize / 2, file.GetLength() ); // Check that we didn't serialize full matrix

	// Let's change weights and roll back to those from serialization
	dnn.RunAndLearnOnce();

	if( initialize ) {
		// check that weights changed after last RunAndLearnOnce
		EXPECT_FALSE( CompareBlobs( *aWeights, *loraFc->GetAWeightsNoCopy() ) );
		EXPECT_FALSE( CompareBlobs( *bWeights, *loraFc->GetBWeightsNoCopy() ) );
	}

	if( discardBeforeLoad ) {
		// This will replace lora wrappers with fcs which will allow us to test loading over raw fcs
		EXPECT_EQ( 1, CLoraBuilder().DiscardAllFcWrappers( dnn ) );
	}

	file.SeekToBegin();
	{
		CArchive archive( &file, CArchive::SD_Loading );
		EXPECT_EQ( 1, CLoraSerializer().Serialize( dnn, archive ) );
	}

	if( discardBeforeLoad ) {
		// If original loraFc has been discarded then CLoraFullyConnected has been deleted
		// and then created anew during CLoraSerializer::Serialize
		// So we need to update the pointer with new layer
		loraFc = CheckCast<CLoraFullyConnectedLayer>( dnn.GetLayer( "full" ) );
	}

	if( initialize ) {
		// Check that after serialization A and B matrices were rolled back
		EXPECT_TRUE( CompareBlobs( *aWeights, *loraFc->GetAWeightsNoCopy() ) );
		EXPECT_TRUE( CompareBlobs( *bWeights, *loraFc->GetBWeightsNoCopy() ) );
	} else {
		EXPECT_EQ( nullptr, loraFc->GetAWeightsNoCopy().Ptr() );
		EXPECT_EQ( nullptr, loraFc->GetBWeightsNoCopy().Ptr() );
	}
}

} // namespace NeoMLTest

TEST( LoraSerializerTest, LoraFc )
{
	const auto met = MathEngine().GetType();
	if(met != MET_Cpu && met != MET_Cuda) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		// dropout
		return;
	}

	for( bool initialize : { true, false } ) {
		for( bool discardBeforeLora : { true, false } ) {
			loraFcSerializerTestImpl( initialize, discardBeforeLora );
		}
	}
}

//----------------------------------------------------------------------------------------------------------------------

namespace NeoMLTest {

// We can use empty class cause the data blobs in this tests are serialized with CSourceLayer
class CLoraTestDistDataset : public IDistributedDataset {
	int SetInputBatch( CDnn&, int ) override { return 1; }
};

} // namespace NeoMLTest

TEST( LoraSerializerTest, Distributed )
{
	// Setting bigger size (full weights will be ioSize x ioSize matrix)
	constexpr int ioSize = 100;
	constexpr __int64 fullMatrixSize = static_cast<__int64>( sizeof( float ) * ioSize * ioSize );

	CRandom random( 0xABBA );
	CDnn dnn( random, MathEngine() );
	buildLoraFcDnn( dnn, ioSize, ioSize, 0.1f );
	CDistributedTraining distributed( dnn, 4 );
	CLoraTestDistDataset dataset;

	distributed.RunAndLearnOnce( dataset ); // now A and B matrices are initialized

	dnn.RunOnce();
	CObjectArray<CDnnBlob> originalBlobs;
	distributed.GetLastBlob( "sink", originalBlobs );
	for( CPtr<CDnnBlob>& blob : originalBlobs ) {
		blob = blob->GetCopy();
	}

	CMemoryFile file; // dump the matrices
	{
		CArchive archive( &file, CArchive::SD_Storing );
		EXPECT_EQ( 1, CLoraSerializer().Serialize( distributed, archive ) );
	}
	EXPECT_GT( fullMatrixSize / 2, file.GetLength() ); // Check that we didn't serialize full matrix

	// Let's make 1 iteration and check that output has been affected by it
	distributed.RunAndLearnOnce( dataset );
	CObjectArray<CDnnBlob> actualBlobs;
	distributed.GetLastBlob( "sink", actualBlobs );

	EXPECT_EQ( originalBlobs.Size(), actualBlobs.Size() );
	for( int i = 0; i < originalBlobs.Size(); ++i ) {
		EXPECT_FALSE( CompareBlobs( *originalBlobs[i], *actualBlobs[i] ) );
	}

	// now lets load LoRA into the distributed
	{
		file.SeekToBegin();
		CArchive archive( &file, CArchive::SD_Loading );
		EXPECT_EQ( 1, CLoraSerializer().Serialize( distributed, archive ) );
		EXPECT_EQ( file.GetPosition(), file.GetLength() );
	}

	// Check that after loading distributed from archive restored the net to original state
	distributed.RunOnce( dataset );
	distributed.GetLastBlob( "sink", actualBlobs );
	EXPECT_EQ( originalBlobs.Size(), actualBlobs.Size() );
	for( int i = 0; i < originalBlobs.Size(); ++i ) {
		EXPECT_FALSE( CompareBlobs( *originalBlobs[i], *actualBlobs[i] ) );
	}
}

TEST( LoraSerializerTest, DistributedCheckpoint )
{
	constexpr int ioSize = 10;

	CRandom random( 0xABBA );
	CDnn dnn( random, MathEngine() );
	buildLoraTransformerDnn( dnn, ioSize );
	dnn.RunOnce();

	CLoraBuilder builder;
	CLoraParams params( 4, 2.f, 0.f ); // dropout must be zero
	EXPECT_EQ( 6, builder.BuildAllFcWrappers( dnn, params ) );
	EXPECT_EQ( 2, builder.DisableNonLoraTraining( dnn ) );

	CDistributedTraining distributed( dnn, 4 );
	CLoraTestDistDataset dataset;

	// Learn a couple of times (before creating checkpoints)
	for( int i = 0; i < 2; ++i ) {
		distributed.RunAndLearnOnce( dataset );
	}

	CMemoryFile file; // Store checkpoints
	{
		CArchive archive( &file, CArchive::SD_Storing );
		EXPECT_EQ( 6, CLoraSerializer().SerializeCheckpoint( distributed, archive ) );
	}

	const int testedIterations = 20;
	CArray<CObjectArray<CDnnBlob>> storedOutputs;
	CArray<CArray<float>> storedLosses;
	// Learn a few times and store outputs on each iteration after checkpoint
	for( int iter = 0; iter < testedIterations; ++iter ) {
		distributed.RunAndLearnOnce( dataset );
		distributed.GetLastBlob( "sink", storedOutputs.Append() );
		for( CPtr<CDnnBlob>& blob : storedOutputs.Last() ) {
			blob = blob->GetCopy();
		}
		distributed.GetLastLoss( "loss", storedLosses.Append() );
	}

	// now lets load LoRA checkpoint into the distributed
	{
		file.SeekToBegin();
		CArchive archive( &file, CArchive::SD_Loading );
		EXPECT_EQ( 6, CLoraSerializer().SerializeCheckpoint( distributed, archive ) );
		EXPECT_EQ( file.GetPosition(), file.GetLength() );
	}

	CObjectArray<CDnnBlob> currBlobs;
	CArray<float> currLosses;
	for( int iter = 0; iter < testedIterations; ++iter ) {
		distributed.RunAndLearnOnce( dataset );
		distributed.GetLastBlob( "sink", currBlobs );
		distributed.GetLastLoss( "loss", currLosses );
		EXPECT_EQ( currBlobs.Size(), currLosses.Size() );
		for( int i = 0; i < currBlobs.Size(); ++i ) {
			EXPECT_TRUE( CompareBlobs( *currBlobs[i], *storedOutputs[iter][i], FLT_EPSILON ) );
			EXPECT_FLOAT_EQ( currLosses[i], storedLosses[iter][i] );
		}
	}
}

//----------------------------------------------------------------------------------------------------------------------

namespace NeoMLTest {

static void memCheckTest( bool useLora, int optimizeDnnIterations = 0, int encodersCount = 6, int iterationsCount = 10 )
{
	MathEngine().CleanUp();
	GTEST_LOG_( INFO ) << ( useLora ? "Used LoRA" : "no LoRA" ) << "\n "
		<< "Peak memory after clean: " << GetPeakMemScaled( MathEngine() ) << " MB\n";

	// Build the net
	CRandom random( 0x6543 );
	CDnn dnn( random, MathEngine() );

	const int vecSize = 768;
	const int headCount = 4;
	const int tableSize = 10;
	const int ffSize = vecSize * headCount;
	const float dropout = 0.f;

	CSourceLayer* data = Source( dnn, "data" );
	CArray<CLookupDimension> embDims;
	embDims.Add( CLookupDimension( tableSize, vecSize ) );
	CMultichannelLookupLayer* embeddings = MultichannelLookup( embDims, true )( "emb", data );
	CBaseLayer* lastLayer = embeddings;
	for( int i = 0; i < encodersCount; ++i ) {
		lastLayer = TransformerEncoder( headCount, vecSize, dropout, ffSize, AF_ReLU )( "transformer_" + Str( i ), lastLayer );
	}
	Sink( lastLayer, "sink" );

	CSourceLayer* expected = Source( dnn, "expected" );
	CEuclideanLossLayer* loss = EuclideanLoss()( "loss", lastLayer, expected );

	// Generate random data
	const int seqLen = 512;
	const int batchSize = 2;
	{
		CPtr<CDnnBlob> dataBlob = CDnnBlob::CreateListBlob( MathEngine(), CT_Int, 1, batchSize, seqLen, 1 );
		CDnnBlobBuffer<int> dataBuff( *dataBlob, TDnnBlobBufferAccess::Write );
		for( int i = 0; i < dataBuff.Size(); ++i ) {
			dataBuff[i] = random.UniformInt( 0, tableSize - 1 );
		}
		data->SetBlob( dataBlob );
	}
	{
		CPtr<CDnnBlob> expectedBlob = CDnnBlob::CreateListBlob( MathEngine(), CT_Float, 1, batchSize, seqLen, vecSize );
		CDnnBlobBuffer<float> expectedBuff( *expectedBlob, TDnnBlobBufferAccess::Write );
		for( int i = 0; i < expectedBuff.Size(); ++i ) {
			expectedBuff[i] = static_cast<float>( random.Uniform( 0, 1 ) );
		}
		expected->SetBlob( expectedBlob );
	}

	// Initialize weights
	dnn.RunOnce();
	GTEST_LOG_( INFO ) << "\n "
		<< "Peak memory after RunOnce: " << GetPeakMemScaled( MathEngine() ) << " MB\n";

	if( useLora ) {
		CLoraBuilder builder;
		CLoraParams params( 4, 5.f, 0.1f );
		// 2 fc's inside transformer directly
		// 4 fc's inside of attention (inside transformer)
		EXPECT_EQ( encodersCount * 6, builder.BuildAllFcWrappers( dnn, params ) );
		EXPECT_EQ( encodersCount * 2 + 1, builder.DisableNonLoraTraining( dnn ) );
	} else {
		embeddings->DisableLearning();
	}

	MathEngine().ResetPeakMemoryUsage();

	CPtrOwner<IPerformanceCounters> counters( MathEngine().CreatePerformanceCounters() );
	for( int iter = 0; iter < iterationsCount; ++iter ) {
		counters->Synchronise();
		dnn.RunAndLearnOnce();
		counters->Synchronise();

		GTEST_LOG_( INFO ) << "Iter #" << iter
			<< '\t' << "Loss: " << loss->GetLastLoss()
			<< '\t' << "Train Time: " << GetTimeScaled( *counters ) << " ms."
			<< '\t' << "Peak.Mem: " << GetPeakMemScaled( MathEngine() ) << " MB"
			<< '\n';
	}
	GTEST_LOG_( INFO ) << "\n "
		<< "Peak memory after training: " << GetPeakMemScaled( MathEngine() ) << " MB\n";

	if ( optimizeDnnIterations > 0 ) // Check OptimizeDnn
	{
		const int iters = optimizeDnnIterations;

		MathEngine().ResetPeakMemoryUsage();
		dnn.RunOnce(); // Initializing

		counters->Synchronise();
		for( int iter = 0; iter < iters; ++iter ) {
			dnn.RunOnce();
		}
		counters->Synchronise();

		const double unoptTime = GetTimeScaled( *counters );
		CPtr<CDnnBlob> expectedBlob = CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ) )->GetBlob();
		const double unoptPeakMem = GetPeakMemScaled( MathEngine() );

		OptimizeDnn( dnn );

		MathEngine().ResetPeakMemoryUsage();
		dnn.RunOnce(); // Initializing

		counters->Synchronise();
		for( int iter = 0; iter < iters; ++iter ) {
			dnn.RunOnce();
		}
		counters->Synchronise();

		const double optTime = GetTimeScaled( *counters );
		CPtr<CDnnBlob> sinkBlob = CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ) )->GetBlob();

		GTEST_LOG_( INFO )
			<< "\n RunOnce " << iters << " (unopt) Time: " << unoptTime << " ms.,"
				<< "\t per iter: " << ( unoptTime / iters ) << " ms.,"
				<< "\t Peak.Mem: " << unoptPeakMem << " MB"
			<< "\n RunOnce " << iters << "   (opt) Time: " << optTime << " ms.,"
				<< "\t per iter: " << ( optTime / iters ) << " ms.,"
				<< "\t Peak.Mem: " << GetPeakMemScaled( MathEngine() ) << " MB"
			<< "\n";

		// Check for consistence
		EXPECT_TRUE( CompareBlobs( *expectedBlob, *sinkBlob ) );
	}
}

} // namespace NeoMLTest

TEST( LoraMemCheck, OptimizeTest )
{
	const auto met = MathEngine().GetType();
	if( met != MET_Cpu && met != MET_Cuda ) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		// VectorHuberDerivative, dropout
		return;
	}

	memCheckTest( /*lora*/true, /*optimizeDnnIterations*/10, /*encodersCount*/2, /*iterationsCount*/3 );

	memCheckTest( /*lora*/false, /*optimizeDnnIterations*/10, /*encodersCount*/2, /*iterationsCount*/3 );
}

TEST( LoraMemCheck, DISABLED_PerformanceTestWithoutLora )
{
	const auto met = MathEngine().GetType();
	if( met != MET_Cpu && met != MET_Cuda ) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		// VectorHuberDerivative, dropout
		return;
	}

	memCheckTest( /*lora*/false );
	DeleteMathEngine();
}

TEST( LoraMemCheck, DISABLED_PerformanceTestWithLora )
{
	const auto met = MathEngine().GetType();
	if( met != MET_Cpu && met != MET_Cuda ) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		// VectorHuberDerivative, dropout
		return;
	}

	memCheckTest( /*lora*/true );
	DeleteMathEngine();
}

