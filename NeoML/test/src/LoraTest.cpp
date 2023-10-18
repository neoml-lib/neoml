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

	CPtr<CDnnAdaptiveGradientSolver> solver = new CDnnAdaptiveGradientSolver( mathEngine );
	dnn.SetSolver( solver.Ptr() );
}

static void buildLoraFcDnn( CDnn& dnn, int inputSize, int outputSize, float dropout )
{
	CLoraParams params( 4, 2.f, dropout ); // dropout must be off otherwise math won't match
	CPtr<CDnnBlob> baseFcWeights = generateLoraFcBLob( outputSize, inputSize );
	CPtr<CDnnBlob> baseFcFreeTerms = generateLoraFcBLob( 1, outputSize );

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

	CPtr<CDnnBlob> baseFcWeights = generateLoraFcBLob( outputSize, inputSize );
	CPtr<CLoraFullyConnectedLayer> loraFc = new CLoraFullyConnectedLayer( *baseFcWeights, nullptr, params );

	// Check that uninitialized A and B are nullptrs
	EXPECT_EQ( nullptr, loraFc->GetAWeightsNoCopy().Ptr() );
	EXPECT_EQ( nullptr, loraFc->GetBWeightsNoCopy().Ptr() );

	// Check that baseFc weights were taken by loraFc without copying
	EXPECT_EQ( baseFcWeights.Ptr(), loraFc->GetRawBaseWeightsNoCopy().Ptr() );

	baseFcWeights = baseFcWeights->GetCopy();
	// Check that in unitialized state merge/split doesn't cause any changes
	EXPECT_TRUE( CompareBlobs( *baseFcWeights, *loraFc->GetSplitBaseWeightsNoCopy(), FLT_EPSILON ) );
	EXPECT_TRUE( CompareBlobs( *baseFcWeights, *loraFc->GetMergedBaseWeightsNoCopy(), FLT_EPSILON ) );

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
	EXPECT_TRUE( CompareBlobs( *baseFcWeights, *loraFc->GetRawBaseWeightsNoCopy(), FLT_EPSILON ) );
}

TEST( LoraFullyConnectedLayerTest, InferenceAndLearning )
{
	constexpr int inputSize = 32;
	constexpr int outputSize = 16;

	CRandom random( 0x645 );
	CDnn dnn( random, MathEngine() );
	buildLoraFcDnn( dnn, inputSize, outputSize, 0.f );
	setLoraFcInputs( dnn, inputSize, outputSize );

	CPtr<CLoraFullyConnectedLayer> loraFc = CheckCast<CLoraFullyConnectedLayer>( dnn.GetLayer( "full" ) );
	CDnnBlob* originalWeightsPtr = loraFc->GetRawBaseWeightsNoCopy().Ptr();
	CPtr<CDnnBlob> copyOfOriginalWeights = originalWeightsPtr->GetCopy();

	for( int iteration = 0; iteration < 5; ++iteration ) {
		dnn.RunOnce();
		// Check that layer is in the merged state
		EXPECT_TRUE( loraFc->IsMerged() );
		// Check that no reallocation occured
		EXPECT_EQ( originalWeightsPtr, loraFc->GetRawBaseWeightsNoCopy().Ptr() );
		// Check that weights of A and B are still unintialized during first iteration
		if( iteration == 0 ) {
			EXPECT_EQ( nullptr, loraFc->GetAWeightsNoCopy().Ptr() );
			EXPECT_EQ( nullptr, loraFc->GetBWeightsNoCopy().Ptr() );
		}

		// Store the output
		CPtr<CSinkLayer> sink = CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ) );
		CPtr<CDnnBlob> copyOfOutput = sink->GetBlob()->GetCopy();

		dnn.RunAndLearnOnce();

		// Check that previously uninitialized weights were initialized
		EXPECT_NE( nullptr, loraFc->GetAWeightsNoCopy().Ptr() );
		EXPECT_NE( nullptr, loraFc->GetBWeightsNoCopy().Ptr() );
		// Check that during training layer is in split state and original weights are untouched
		EXPECT_FALSE( loraFc->IsMerged() );
		EXPECT_TRUE( CompareBlobs( *copyOfOriginalWeights, *loraFc->GetRawBaseWeightsNoCopy() ) );
		// Check that no reallocation occured
		EXPECT_EQ( originalWeightsPtr, loraFc->GetRawBaseWeightsNoCopy().Ptr() );
		// Check that output during first iteration of training matches last output before training
		EXPECT_TRUE( loraFc->Dropout() <= 0.f || CompareBlobs( *copyOfOutput, *sink->GetBlob() ) );

		// Now run the net, the layer must switch to merged state
		dnn.RunOnce();

		// Check that layer is in the merged state
		EXPECT_TRUE( loraFc->IsMerged() );
		// The base weights must be at the same location but must contain different (merged) weights
		EXPECT_EQ( originalWeightsPtr, loraFc->GetRawBaseWeightsNoCopy().Ptr() );
		EXPECT_TRUE( copyOfOriginalWeights->HasEqualDimensions( loraFc->GetRawBaseWeightsNoCopy().Ptr() ) );
		EXPECT_FALSE( CompareBlobs( *copyOfOriginalWeights, *loraFc->GetRawBaseWeightsNoCopy() ) );
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
	CPtr<CDnnBlob> splitWeights = CheckCast<CLoraFullyConnectedLayer>( 
		CheckCast<CTransformerEncoderLayer>( dnn.GetLayer( "transformer" ) )->GetLayer( "FullyConnected2" ) )->GetSplitBaseWeightsNoCopy()->GetCopy();
	EXPECT_TRUE( CompareBlobs( *initialWeights, *splitWeights ) );
	EXPECT_TRUE( CompareBlobs( *trainedOutput, *CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ) )->GetBlob() ) );

	EXPECT_EQ( 6, builder.DiscardAllFcWrappers( dnn ) );
	dnn.RunOnce();
	EXPECT_TRUE( checkLayerClass<CFullyConnectedLayer>( dnn, { "transformer", "FullyConnected2" } ) );
	EXPECT_TRUE( checkLayerClass<CFullyConnectedLayer>( dnn, { "transformer", "SelfAttention", "K" } ) );
	EXPECT_TRUE( CompareBlobs( *untrainedOutput, *CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ) )->GetBlob() ) );
}

//----------------------------------------------------------------------------------------------------------------------

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
		ASSERT_EQ( 1, CLoraSerializer().Serialize( dnn, archive ) );
	}
	// Check that we didn't serialize full matrix
	ASSERT_GT( fullMatrixSize / 2, file.GetLength() );

	// Let's change weights and roll back to those from serialization
	dnn.RunAndLearnOnce();

	if( initialize ) {
		// check that weights changed after last RunAndLearnOnce
		EXPECT_FALSE( CompareBlobs( *aWeights, *loraFc->GetAWeightsNoCopy() ) );
		EXPECT_FALSE( CompareBlobs( *bWeights, *loraFc->GetBWeightsNoCopy() ) );
	}

	if( discardBeforeLoad ) {
		ASSERT_EQ( 1, CLoraBuilder().DiscardAllFcWrappers( dnn ) );
	}

	file.SeekToBegin();
	{
		CArchive archive( &file, CArchive::SD_Loading );
		ASSERT_EQ( 1, CLoraSerializer().Serialize( dnn, archive ) );
	}

	if( discardBeforeLoad ) {
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

TEST( LoraSerializerTest, LoraFc )
{
	for( bool initialize : { true, false } ) {
		for( bool discardBeforeLora : { true, false } ) {
			loraFcSerializerTestImpl( initialize, discardBeforeLora );
		}
	}
}
