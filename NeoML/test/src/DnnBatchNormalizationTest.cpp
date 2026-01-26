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

#include <TestFixture.h>
#include <DnnSimpleTest.h>

using namespace NeoML;
using namespace NeoMLTest;

namespace NeoMLTest {

// Training items: the input is 5x5, the output is binary
constexpr int BlobWidth = 5;
constexpr int BlobHeight = 5;
constexpr int BlobArea = BlobWidth * BlobHeight;

// 1 epoch processing
enum class TEpochProcess {
	Run, // forward pass only
	RunAndLearn // forward and backward pass
};

// The item of the training sample
struct CTrainItem final {
	float Source[BlobArea]{};
	bool Label = false;
};

// Training samples generation:
// Source is a random elements from [0,1]
// Label = avg(Source) > 0.5 ? 1 : 0
static CTrainItem generateTrainItem( CRandom& random )
{
	CTrainItem item;
	double sum = 0.;
	for( int i = 0; i < BlobArea; ++i ) {
		const float v = static_cast<float>( random.Uniform( 0., 1. ) );
		item.Source[i] = v;
		sum += v;
	}
	item.Label = ( sum / BlobArea ) > 0.5;
	return item;
}

// Build a dnn with a BatchNorm
static void buildDnnBatchNorm( CDnn& dnn )
{
	CSourceLayer* source = Source( dnn, "source" );
	CSourceLayer* labels = Source( dnn, "labels" );

	CBaseLayer* layer = nullptr;
	// 1. conv
	layer = Conv( 3, CConvAxisParams( 3 ), CConvAxisParams( 3 ) )( "conv", source );
	// 2. elu
	layer = Elu()( "elu", layer );
	// 3. batch norm
	layer = BatchNormalization( /*isChannelBased*/true )( "batchNorm", layer );
	// 4. fully connected
	layer = FullyConnected( 1 )( "fc1", layer );

	EuclideanLoss()( "loss", layer, labels );
	Sink( layer, "sink" );

	CDnnAdaptiveGradientSolver* solver = new CDnnAdaptiveGradientSolver( dnn.GetMathEngine() );
	solver->SetLearningRate( 1e-3f );
	solver->SetMomentDecayRate( 0.8f );
	solver->SetSecondMomentDecayRate( 0.8f );
	dnn.SetSolver( solver );
}

// Process the sample, return the average loss
static double processEpoch( CDnn& dnn, CArray<CTrainItem>& trainData, TEpochProcess type, int batchSize = 5 )
{
	CPtr<CSourceLayer> source = CheckCast<CSourceLayer>( dnn.GetLayer( "source" ) );
	CPtr<CSourceLayer> labels = CheckCast<CSourceLayer>( dnn.GetLayer( "labels" ) );
	CPtr<CLossLayer> loss = CheckCast<CLossLayer>( dnn.GetLayer( "loss" ) );

	double epochLoss = 0.;
	for( int batchIdx = 0; batchIdx < trainData.Size() / batchSize; ++batchIdx ) {
		// Fill the input blobs
		CPtr<CDnnBlob> sourceBlob = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1, batchSize, BlobHeight, BlobWidth, 1 );
		CPtr<CDnnBlob> labelsBlob = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1, batchSize, 1, 1, 1 );
		for( int objInx = 0; objInx < batchSize; ++objInx ) {
			const int trainIdx = batchSize * batchIdx + objInx;

			dnn.GetMathEngine().DataExchangeTyped( sourceBlob->GetObjectData( objInx ),
				trainData[trainIdx].Source, sourceBlob->GetObjectSize() );

			const float label = trainData[trainIdx].Label ? 1.f : 0.f;
			dnn.GetMathEngine().DataExchangeTyped( labelsBlob->GetObjectData( objInx ),
				&label, labelsBlob->GetObjectSize() );
		}
		source->SetBlob( sourceBlob );
		labels->SetBlob( labelsBlob );

		// Process the operation
		switch( type ) {
			case TEpochProcess::Run:
				dnn.RunOnce();
				break;
			case TEpochProcess::RunAndLearn:
				dnn.RunAndLearnOnce();
				break;
			default:
				NeoAssert( false );
		}
		// Update the loss
		epochLoss += loss->GetLastLoss();
	}
	return epochLoss;
}

// Check positions and values of the final batch norm params
static void checkFinalParams( CPtr<CDnnBlob> finalParams, const float* gammaData, const float* betaData, int objectSize )
{
	CArray<float> result;
	result.SetSize( objectSize );

	CConstFloatHandle gamma = finalParams->GetObjectData( CBatchNormalizationLayer::PN_Gamma );
	MathEngine().DataExchangeTyped( result.GetPtr(), gamma, objectSize );
	for( int i = 0; i < objectSize; ++i ) {
		EXPECT_NEAR( gammaData[i], result[i], 1e-3f ) << " i=" << i;
	}

	CConstFloatHandle beta = finalParams->GetObjectData( CBatchNormalizationLayer::PN_Beta );
	MathEngine().DataExchangeTyped( result.GetPtr(), beta, objectSize );
	for( int i = 0; i < objectSize; ++i ) {
		EXPECT_NEAR( betaData[i], result[i], 1e-3f ) << " i=" << i;
	}
}

// Check batch norm output
static void checkOutput( const CSinkLayer& sink, CArray<float>& output, const float* expected, int size )
{
	EXPECT_EQ( size, sink.GetBlob()->GetDataSize() );
	output.SetSize( size );
	sink.GetBlob()->CopyTo( output.GetPtr() );

	for( int i = 0; i < size; ++i ) {
		EXPECT_NEAR( output[i], expected[i], 1e-3f ) << " i=" << i;
	}
}

} // namespace NeoMLTest

//---------------------------------------------------------------------------------------------------------------------

TEST( DnnBatchNormalizationTest, SimpleTest )
{
	CRandom random( 45 );

	// Build the dnn
	CDnn dnn( random, MathEngine() );
	buildDnnBatchNorm( dnn );

	// Generate a sample
	const int dataSize = 100;
	CArray<CTrainItem> trainData;
	trainData.SetBufferSize( dataSize );
	for( int i = 0; i < dataSize; ++i ) {
		trainData.Add( generateTrainItem( random ) );
	}

	// Start training for the epoch count
	int epochCount = 1;
	for( int i = 0; i < epochCount; ++i ) {
		processEpoch( dnn, trainData, TEpochProcess::RunAndLearn );
	}

	CDnn deserializedDnn( random, MathEngine() );
	{
		CMemoryFile file;
		// store the dnn
		CArchive archive( &file, CArchive::store );
		archive << dnn;
		archive.Close();
		file.SeekToBegin();
		// load the dnn 
		archive.Open( &file, CArchive::load );
		archive >> deserializedDnn;
	}

	// The quality calculation on a regular dnn
	const double dnnLoss = processEpoch( dnn, trainData, TEpochProcess::Run, 1 );
	// The quality calculation on a deserialized dnn
	const double deserializedDnnLoss = processEpoch( deserializedDnn, trainData, TEpochProcess::Run, 1 );
	EXPECT_DOUBLE_EQ( dnnLoss, deserializedDnnLoss );
}

TEST( DnnBatchNormalizationTest, ConsistencyTest )
{
	CRandom random( 45 );
	CDnn dnn( random, MathEngine() );

	CDnnAdaptiveGradientSolver* solver = new CDnnAdaptiveGradientSolver( dnn.GetMathEngine() );
	dnn.SetSolver( solver );

	const int batchSize = 8; // batchSize >= MinBatchSize=8, for correct operation of the algorithm
	const int channels = 3;

	CSourceLayer* source = Source( dnn, "source" );
	{ // Set source data
		const float sourceData[batchSize * channels]{
			 0.0f, 1.0f, 2.0f, 3.0f,  4.0f,  5.0f,
			 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
			-1.0f,-2.0f,-3.0f,-4.0f, -5.0f, -6.0f,
			 1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f
		};
		CPtr<CDnnBlob> sourceBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, batchSize, channels );
		sourceBlob->CopyFrom( sourceData );
		source->SetBlob( sourceBlob );
	}

	CPtr<CDnnBlob> inputDiffBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, batchSize, channels );
	CBaseLayer* dummy = SimpleTestDummyLearn( inputDiffBlob )( "dummy", source );

	CBatchNormalizationLayer* batchNorm = BatchNormalization( /*isChannelBased*/true )( "batchNorm", dummy );
	EXPECT_NEAR( batchNorm->GetSlowConvergenceRate(), 1.f, 1e-3f );
	EXPECT_EQ( batchNorm->IsUsingFinalParamsForInitialization(), false );
	EXPECT_EQ( batchNorm->IsZeroFreeTerm(), false );
	EXPECT_EQ( batchNorm->IsChannelBased(), true );

	{ // Set normalization parameters: gamma and beta (scaling and bias)
		const float paramsData[CBatchNormalizationLayer::PN_Count * channels]{
			/*gamma*/ 1.0f, 0.9f, 1.1f,
			/*beta*/  0.5f, 0.4f, 0.6f
		};
		CPtr<CDnnBlob> paramsBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float,
			1, CBatchNormalizationLayer::PN_Count, channels );
		paramsBlob->CopyFrom( paramsData );

		batchNorm->SetFinalParams( paramsBlob );

		CPtr<CDnnBlob> finalParams = batchNorm->GetFinalParams();
		EXPECT_EQ( finalParams->GetObjectSize(), channels );
		EXPECT_TRUE( CompareBlobs( *paramsBlob, *finalParams ) );

		checkFinalParams( finalParams,
			&paramsData[CBatchNormalizationLayer::PN_Gamma * channels],
			&paramsData[CBatchNormalizationLayer::PN_Beta * channels],
			channels );
	}

	const CSinkLayer* sink = Sink( batchNorm, "sink" );
	CArray<float> output;

	dnn.RunOnce();
	{ // Check the forward calculations
		const float expected[batchSize * channels]{
			 0.5f, 1.3f, 2.8f, 3.5f, 4.0f, 6.1f,
			 6.5f, 6.7f, 9.4f, 9.5f, 9.4f,12.7f,
			-0.5f,-1.4f,-2.7f,-3.5f,-4.1f,-6.0f,
			 1.5f, 2.2f, 3.9f, 4.5f, 4.9f, 7.2f
		};
		checkOutput( *sink, output, expected, batchSize * channels );
	}

	CSourceLayer* labels = Source( dnn, "labels" );
	CLossLayer* loss = EuclideanLoss()( "loss", batchNorm, labels );
	{ // Set labels data
		const float labelsData[batchSize * channels]{
			-1.f, -1.f, -1.f, 1.f, 1.f, 1.f,
			-1.f, -1.f, -1.f, 1.f, 1.f, 1.f,
			-1.f, -1.f, -1.f, 1.f, 1.f, 1.f,
			-1.f, -1.f, -1.f, 1.f, 1.f, 1.f
		};
		CPtr<CDnnBlob> labelsBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, batchSize, channels );
		labelsBlob->CopyFrom( labelsData );
		labels->SetBlob( labelsBlob );
	}

	// Check the backward calculations
	const int iterations = 2;
	const float inputDiff[iterations + 1][batchSize * channels]{
		{
			0.017703, 0.018084, 0.017680,-0.030949,-0.025404,-0.021357,
			0.018387, 0.015713, 0.013419, 0.001115,-0.001781,-0.003979,
			0.012359, 0.006273, 0.003198,-0.016056,-0.013438,-0.011078,
			0.023047, 0.022021, 0.020577,-0.025605,-0.021467,-0.018461
		},
		{
			0.017540, 0.017902, 0.017492,-0.030717,-0.025221,-0.021208,
			0.018667, 0.015947, 0.013619, 0.000846,-0.001973,-0.004107,
			0.012279, 0.006278, 0.003241,-0.015959,-0.013363,-0.011021,
			0.022800, 0.021776, 0.020342,-0.025456,-0.021346,-0.018358
		},
		{
			0.013234, 0.015846, 0.016654,-0.024966,-0.019428,-0.015613,
			0.029649, 0.025347, 0.021876, 0.017228, 0.011410, 0.007358,
			0.006202, 0.000427,-0.002489,-0.029649,-0.025347,-0.021876,
			0.020266, 0.020986, 0.020482,-0.017934,-0.014289,-0.011785
		}
	};
	const float expected[iterations + 1][batchSize * channels]{
		{
			-0.582162,-0.387101,-0.238637, 0.194054, 0.276501, 0.334092,
			 0.970269, 0.940102, 0.906821, 1.746485, 1.603704, 1.479549,
			-0.840900,-1.050702,-1.193185,-1.617115,-1.714304,-1.765914,
			-0.323423,-0.165900,-0.047727, 0.452792, 0.497701, 0.525001
		},
		{
			-0.586339,-0.393229,-0.246250, 0.182115, 0.263737, 0.320752,
			 0.950568, 0.920702, 0.887754, 1.719022, 1.577668, 1.454755,
			-0.842490,-1.050195,-1.191252,-1.610944,-1.707160,-1.758254,
			-0.330188,-0.174240,-0.057249, 0.438266, 0.482725, 0.509752
		},
		{
			-0.553631,-0.374826,-0.238731, 0.157934, 0.233505, 0.286296,
			 0.869499, 0.841836, 0.811322, 1.581064, 1.450166, 1.336349,
			-0.790820,-0.983156,-1.113775,-1.502385,-1.591487,-1.638802,
			-0.316443,-0.172049,-0.063722, 0.395122, 0.436282, 0.461304
		}
	};
	const float gamma[iterations + 1][channels]{
		{ 0.239607, 0.204845, 0.176793 },
		{ 0.237188, 0.202777, 0.175009 },
		{ 0.237188, 0.202777, 0.175009 }
	};	 
	const float beta[iterations + 1][channels]{
		{ -0.549115, -0.573322, -0.584578 },
		{ -0.553631, -0.577602, -0.588749 },
		{ -0.553631, -0.577602, -0.588749 }
	};

	float lastLoss = FLT_MAX;
	for( int i = 0; i < iterations; ++i ) {
		inputDiffBlob->CopyFrom( inputDiff[i] );
		dnn.RunAndLearnOnce();

		EXPECT_GT( lastLoss, loss->GetLastLoss() );
		lastLoss = loss->GetLastLoss();

		checkOutput( *sink, output, expected[i], batchSize * channels );
		checkFinalParams( batchNorm->GetFinalParams(), gamma[i], beta[i], channels );
	}

	{ // Check the backward NO learnings calculations
		batchNorm->DisableLearning();

		inputDiffBlob->CopyFrom( inputDiff[iterations] );
		dnn.RunAndBackwardOnce();
		EXPECT_GT( lastLoss, loss->GetLastLoss() );

		checkOutput( *sink, output, expected[iterations], batchSize * channels );
		checkFinalParams( batchNorm->GetFinalParams(), gamma[iterations], beta[iterations], channels );
	}
}
