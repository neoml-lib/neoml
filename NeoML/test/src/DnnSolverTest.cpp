/* Copyright © 2021 ABBYY Production LLC

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

// Returns coefficient of neuron with one in and one out.
static float getFcCoeff( const CFullyConnectedLayer* fcLayer )
{
	NeoAssert( fcLayer != 0 );
	return fcLayer->GetWeightsData()->GetData().GetValue();
}

// Check build/change correctness with gradient accumulation enabled
TEST( CDnnSolverTest, NetworkModificationOnGradientAccumulation )
{
	// Sinus approximation.
	const int DataCount = 10;
	CArray<float> x, y;
	for( int i = 0; i < DataCount; i++ ) {
		const float f = i * .1f;
		x.Add(f);
		y.Add( sinf(f) );
	}

	// Layers creation
	CPtr<CSourceLayer> xLayer = new CSourceLayer( MathEngine() );
	xLayer->SetName( "xLayer" );
	CPtr<CDnnBlob> xBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, DataCount, 1 );
	xBlob->CopyFrom( x.GetPtr() );
	xLayer->SetBlob( xBlob );

	CPtr<CSourceLayer> yLayer = new CSourceLayer( MathEngine() );
	yLayer->SetName( "yLayer" );
	CPtr<CDnnBlob> yBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, DataCount, 1 );
	yBlob->CopyFrom( y.GetPtr() );
	yLayer->SetBlob( yBlob );

	CPtr<CFullyConnectedLayer> fcLayer = new CFullyConnectedLayer( MathEngine() );
	fcLayer->SetNumberOfElements(1);
	fcLayer->SetZeroFreeTerm( true );

	CPtr<CEuclideanLossLayer> lossLayer = new CEuclideanLossLayer( MathEngine() );
	CPtr<CSinkLayer> sinkLayer = new CSinkLayer( MathEngine() );

	// Solver trains once in two steps and optimizes only loss
	// (regularization disabled).
	CPtr<CDnnSolver> solver = new NeoML::CDnnAdaptiveGradientSolver( MathEngine() );
	solver->SetL1Regularization(0);
	solver->SetL2Regularization(0);

	// Create network.
	NeoML::CRandom random( 43 );
	NeoML::CDnn dnn( random, MathEngine() );
	dnn.AddLayer( *xLayer );
	dnn.AddLayer( *yLayer );
	dnn.AddLayer( *fcLayer );
	dnn.AddLayer( *lossLayer );
	fcLayer->Connect( *xLayer );
	lossLayer->Connect( 0, *fcLayer );
	lossLayer->Connect( 1, *yLayer );
	dnn.SetSolver( solver );
	EXPECT_TRUE( dnn.IsRebuildRequested() );

	// Make steps with solver
	// Training in even steps.

	dnn.RunAndBackwardOnce();
	const float c1 = getFcCoeff( fcLayer );
	EXPECT_FALSE( dnn.IsRebuildRequested() );

	dnn.RunAndBackwardOnce();
	solver->Train();
	const float c2 = getFcCoeff( fcLayer );
	EXPECT_NE( c1, c2 );

	dnn.RunAndBackwardOnce();
	const float c3 = getFcCoeff( fcLayer );
	EXPECT_EQ( c2, c3 );

	// Solver training.
	solver->Train();
	const float c3a = getFcCoeff( fcLayer );
	EXPECT_NE( c3, c3a );
	EXPECT_FALSE( dnn.IsRebuildRequested() );

	// Continue, make even training steps.
	dnn.RunAndBackwardOnce();
	const float c4 = getFcCoeff( fcLayer );
	EXPECT_EQ( c3a, c4 );

	dnn.RunAndBackwardOnce();
	solver->Train();
	const float c5 = getFcCoeff( fcLayer );
	EXPECT_NE( c4, c5 );

	dnn.RunAndBackwardOnce();
	const float c6 = getFcCoeff( fcLayer );
	EXPECT_EQ( c5, c6 );

	// Add layer
	dnn.AddLayer( *sinkLayer );
	sinkLayer->Connect( *fcLayer );
	EXPECT_TRUE( dnn.IsRebuildRequested() );

	// Check that solver has been reset.
	dnn.RunAndBackwardOnce();
	const float d1 = getFcCoeff( fcLayer );
	EXPECT_EQ( c6, d1 );
	EXPECT_FALSE( dnn.IsRebuildRequested() );

	dnn.RunAndBackwardOnce();
	solver->Train();
	const float d2 = getFcCoeff( fcLayer );
	EXPECT_NE( d1, d2 );

	dnn.RunAndBackwardOnce();
	const float d3 = getFcCoeff( fcLayer );
	EXPECT_EQ( d2, d3 );

	// Delete loss and target layers.
	dnn.DeleteLayer( *lossLayer );
	dnn.DeleteLayer( *yLayer );
	EXPECT_TRUE( dnn.IsRebuildRequested() );

	// Check that solver has been reset and has no impact.
	dnn.RunAndBackwardOnce();
	const float e1 = getFcCoeff( fcLayer );
	EXPECT_EQ( d3, e1 );

	dnn.RunAndBackwardOnce();
	const float e2 = getFcCoeff( fcLayer );
	EXPECT_EQ( e1, e2 );
}

// Net for weight check.
class CWeightCheckNet {
public:
	CWeightCheckNet();
	void SetSolver( CDnnSolver* solver ) { dnn.SetSolver( solver ); }
	float RunAndLearnOnce();
	void GetWeights( CArray<float>& weights ) const;
private:
	CRandom random;
	CDnn dnn;
	CPtr<CFullyConnectedLayer> fc;
	CPtr<CLossLayer> loss;
};

CWeightCheckNet::CWeightCheckNet() :
	random( 0xAAAAAAAA ),
	dnn( random, MathEngine() )
{
	CPtr<CSourceLayer> data = AddLayer<CSourceLayer>( "data", dnn );
	{
		CPtr<CDnnBlob> dataBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 1, 2 );
		CArray<float> buff( { 0.25f, -0.345f } );
		dataBlob->CopyFrom( buff.GetPtr() );
		data->SetBlob( dataBlob );
	}

	CPtr<CSourceLayer> label = AddLayer<CSourceLayer>( "label", dnn );
	{
		CPtr<CDnnBlob> labelBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Int, 1, 1, 1 );
		CArray<int> buff( { 0 } );
		labelBlob->CopyFrom( buff.GetPtr() );
		label->SetBlob( labelBlob );
	}

	fc = AddLayer<CFullyConnectedLayer>( "fc", { data } );
	fc->SetNumberOfElements( 2 );
	fc->SetZeroFreeTerm( true );
	{
		CPtr<CDnnBlob> weightBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 2, 2 );
		CArray<float> buff = { -0.5f, 0.9f, 0.3f, -0.7f };
		weightBlob->CopyFrom( buff.GetPtr() );
		fc->SetWeightsData( weightBlob );
	}

	loss = AddLayer<CCrossEntropyLossLayer>( "loss", { fc, label } );
}

float CWeightCheckNet::RunAndLearnOnce()
{
	dnn.RunAndLearnOnce();
	return loss->GetLastLoss();
}

void CWeightCheckNet::GetWeights( CArray<float>& weights ) const
{
	CPtr<CDnnBlob> weightBlob = fc->GetWeightsData();
	weights.SetSize( weightBlob->GetDataSize() );
	weightBlob->CopyTo( weights.GetPtr() );
}

void testSolver( CDnnSolver* solver, const CArray<CArray<float>>& expected )
{
	CWeightCheckNet net;
	CArray<float> weights;
	net.SetSolver( solver );
	for( int i = 0; i < expected.Size(); ++i ) {
		float loss = net.RunAndLearnOnce();
		loss;
		net.GetWeights( weights );
		ASSERT_EQ( expected[i].Size(), weights.Size() );
		for( int j = 0; j < weights.Size(); ++j ) {
			ASSERT_TRUE( FloatEq( expected[i][j], weights[j] ) );
		}
	}
}

// ====================================================================================================================
// Sgd.

TEST( CDnnSolverTest, SgdNoReg )
{
	CArray<CArray<float>> expected;
	expected.SetSize( 5 );
	expected[0] = { -0.415048f, 0.782767f, 0.215048f, -0.582767f };
	expected[1] = { -0.257069f, 0.564755f, 0.057069f, -0.364755f };
	expected[2] = { -0.040076f, 0.265304f, -0.159924f, -0.065304f };
	expected[3] = { 0.220345f, -0.094076f, -0.420345f, 0.294076f };
	expected[4] = { 0.508099f, -0.491177f, -0.708099f, 0.691177f };
	CPtr<CDnnSimpleGradientSolver> sgd = new CDnnSimpleGradientSolver( MathEngine() );
	sgd->SetL1Regularization( 0 );
	sgd->SetL2Regularization( 0 );
	sgd->SetMomentDecayRate( 0.9f );
	sgd->SetLearningRate( 0.5f );
	sgd->SetCompatibilityMode( false );
	testSolver( sgd, expected );
}

TEST( CDnnSolverTest, SgdL1 )
{
	CArray<CArray<float>> expected;
	expected.SetSize( 5 );
	expected[0] = { -0.165048f, 0.332767f, 0.065048f, -0.232767f };
	expected[1] = { 0.289286f, -0.441214f, -0.249286f, 0.401214f };
	expected[2] = { 0.602950f, -0.985371f, -0.456950f, 0.839371f };
	expected[3] = { 0.620045f, -1.032483f, -0.451645f, 0.864083f };
	expected[4] = { 0.360971f, -0.623960f, -0.256611f, 0.503358f };
	CPtr<CDnnSimpleGradientSolver> sgd = new CDnnSimpleGradientSolver( MathEngine() );
	sgd->SetL1Regularization( 1 );
	sgd->SetL2Regularization( 0 );
	sgd->SetMomentDecayRate( 0.9f );
	sgd->SetLearningRate( 0.5f );
	sgd->SetCompatibilityMode( false );
	testSolver( sgd, expected );
}

TEST( CDnnSolverTest, SgdL2 )
{
	CArray<CArray<float>> expected;
	expected.SetSize( 5 );
	expected[0] = { -0.290048f, 0.557766f, 0.140048f, -0.407767f };
	expected[1] = { 0.047472f, 0.005364f, -0.114972f, 0.062136f };
	expected[2] = { 0.399991f, -0.576794f, -0.376366f, 0.553169f };
	expected[3] = { 0.662014f, -1.018298f, -0.562283f, 0.918567f };
	expected[4] = { 0.766585f, -1.208346f, -0.623291f, 1.065052f };
	CPtr<CDnnSimpleGradientSolver> sgd = new CDnnSimpleGradientSolver( MathEngine() );
	sgd->SetL1Regularization( 0 );
	sgd->SetL2Regularization( 0.5f );
	sgd->SetMomentDecayRate( 0.9f );
	sgd->SetLearningRate( 0.5f );
	sgd->SetCompatibilityMode( false );
	testSolver( sgd, expected );
}

// Sgd with backward compatibility.

TEST( CDnnSolverTest, SgdCompatNoReg )
{
	CArray<CArray<float>> expected;
	expected.SetSize( 5 );
	expected[0] = { -0.491505f, 0.888277f, 0.291505f, -0.688277f };
	expected[1] = { -0.475398f, 0.866049f, 0.275398f, -0.666049f };
	expected[2] = { -0.452504f, 0.834455f, 0.252504f, -0.634455f };
	expected[3] = { -0.423594f, 0.794560f, 0.223594f, -0.594560f };
	expected[4] = { -0.389388f, 0.747356f, 0.189388f, -0.547356f };
	CPtr<CDnnSimpleGradientSolver> sgd = new CDnnSimpleGradientSolver( MathEngine() );
	sgd->SetL1Regularization( 0 );
	sgd->SetL2Regularization( 0 );
	sgd->SetMomentDecayRate( 0.9f );
	sgd->SetLearningRate( 0.5f );
	sgd->SetCompatibilityMode( true );
	testSolver( sgd, expected );
}

TEST( CDnnSolverTest, SgdCompatL1 )
{
	CArray<CArray<float>> expected;
	expected.SetSize( 5 );
	expected[0] = { -0.241505f, 0.638277f, 0.141505f, -0.438277f };
	expected[1] = { 0.119577f, 0.142122f, -0.079577f, 0.027016f };
	expected[2] = { 0.390981f, -0.384060f, -0.244981f, 0.440854f };
	expected[3] = { 0.444640f, -0.672336f, -0.276240f, 0.599623f };
	expected[4] = { 0.274988f, -0.687823f, -0.170628f, 0.498553f };
	CPtr<CDnnSimpleGradientSolver> sgd = new CDnnSimpleGradientSolver( MathEngine() );
	sgd->SetL1Regularization( 0.5f );
	sgd->SetL2Regularization( 0 );
	sgd->SetMomentDecayRate( 0.9f );
	sgd->SetLearningRate( 0.5f );
	sgd->SetCompatibilityMode( true );
	testSolver( sgd, expected );
}

TEST( CDnnSolverTest, SgdCompatL2 )
{
	CArray<CArray<float>> expected;
	expected.SetSize( 5 );
	expected[0] = { -0.479005f, 0.865777f, 0.284005f, -0.670777f };
	expected[1] = { -0.439724f, 0.801725f, 0.254099f, -0.616100f };
	expected[2] = { -0.385130f, 0.712654f, 0.212583f, -0.540107f };
	expected[3] = { -0.318349f, 0.603608f, 0.161887f, -0.447145f };
	expected[4] = { -0.242559f, 0.479710f, 0.104483f, -0.341635f };
	CPtr<CDnnSimpleGradientSolver> sgd = new CDnnSimpleGradientSolver( MathEngine() );
	sgd->SetL1Regularization( 0 );
	sgd->SetL2Regularization( 0.5f );
	sgd->SetMomentDecayRate( 0.9f );
	sgd->SetLearningRate( 0.5f );
	sgd->SetCompatibilityMode( true );
	testSolver( sgd, expected );
}

// ====================================================================================================================
// Adam.

TEST( CDnnSolverTest, AdamNoReg )
{
	CArray<CArray<float>> expected;
	expected.SetSize( 5 );
	expected[0] = { -0.000001f, 0.400001f, -0.199999f, -0.200001f };
	expected[1] = { 0.493727f, -0.093727f, -0.693727f, 0.293727f };
	expected[2] = { 0.973341f, -0.573342f, -1.173342f, 0.773342f };
	expected[3] = { 1.430382f, -1.030383f, -1.630382f, 1.230382f };
	expected[4] = { 1.858475f, -1.458476f, -2.058475f, 1.658476f };
	CPtr<CDnnAdaptiveGradientSolver> adam = new CDnnAdaptiveGradientSolver( MathEngine() );
	adam->SetL1Regularization( 0 );
	adam->SetL2Regularization( 0 );
	adam->SetMomentDecayRate( 0.9f );
	adam->SetSecondMomentDecayRate( 0.999f );
	adam->SetEpsilon( 1e-8f );
	adam->SetLearningRate( 0.5f );
	adam->SetCompatibilityMode( false );

	testSolver( adam, expected );
}

TEST( CDnnSolverTest, AdamL1 )
{
	CArray<CArray<float>> expected;
	expected.SetSize( 5 );
	expected[0] = { -0.000000f, 0.400000f, -0.200000f, -0.200000f };
	expected[1] = { 0.401868f, -0.068456f, -0.480680f, 0.251741f };
	expected[2] = { 0.559106f, -0.448867f, -0.452042f, 0.565161f };
	expected[3] = { 0.512657f, -0.678728f, -0.282221f, 0.684238f };
	expected[4] = { 0.358081f, -0.750478f, -0.070446f, 0.649429f };
	CPtr<CDnnAdaptiveGradientSolver> adam = new CDnnAdaptiveGradientSolver( MathEngine() );
	adam->SetL1Regularization( 1 );
	adam->SetL2Regularization( 0 );
	adam->SetMomentDecayRate( 0.9f );
	adam->SetSecondMomentDecayRate( 0.999f );
	adam->SetEpsilon( 1e-8f );
	adam->SetLearningRate( 0.5f );
	adam->SetCompatibilityMode( false );

	testSolver( adam, expected );
}

TEST( CDnnSolverTest, AdamL2 )
{
	CArray<CArray<float>> expected;
	expected.SetSize( 5 );
	expected[0] = { -0.000000f, 0.400000f, -0.199999f, -0.200000f };
	expected[1] = { 0.432754f, -0.074625f, -0.573311f, 0.264508f };
	expected[2] = { 0.676471f, -0.480197f, -0.664668f, 0.627873f };
	expected[3] = { 0.707804f, -0.762206f, -0.563969f, 0.826470f };
	expected[4] = { 0.602233f, -0.897641f, -0.371725f, 0.865617f };
	CPtr<CDnnAdaptiveGradientSolver> adam = new CDnnAdaptiveGradientSolver( MathEngine() );
	adam->SetL1Regularization( 0 );
	adam->SetL2Regularization( 0.5f );
	adam->SetMomentDecayRate( 0.9f );
	adam->SetSecondMomentDecayRate( 0.999f );
	adam->SetEpsilon( 1e-8f );
	adam->SetLearningRate( 0.5f );
	adam->SetCompatibilityMode( false );

	testSolver( adam, expected );
}

// Adam with backward compatibility.
TEST( CDnnSolverTest, AdamCompatNoReg )
{
	CArray<CArray<float>> expected;
	expected.SetSize( 5 );
	expected[0] = { -0.450000f, 0.850000f, 0.250000f, -0.650000f };
	expected[1] = { -0.355053f, 0.755053f, 0.155053f, -0.555053f };
	expected[2] = { -0.219870f, 0.619870f, 0.019870f, -0.419870f };
	expected[3] = { -0.048983f, 0.448983f, -0.151017f, -0.248983f };
	expected[4] = { 0.153073f, 0.246927f, -0.353073f, -0.046927f };
	CPtr<CDnnAdaptiveGradientSolver> adam = new CDnnAdaptiveGradientSolver( MathEngine() );
	adam->SetL1Regularization( 0 );
	adam->SetL2Regularization( 0 );
	adam->SetMomentDecayRate( 0.9f );
	adam->SetSecondMomentDecayRate( 0.999f );
	adam->SetEpsilon( 1e-8f );
	adam->SetLearningRate( 0.5f );
	adam->SetCompatibilityMode( true );

	testSolver( adam, expected );
}

TEST( CDnnSolverTest, AdamCompatL1 )
{
	CArray<CArray<float>> expected;
	expected.SetSize( 5 );
	expected[0] = { -0.450000f, 0.850000f, 0.250000f, -0.650000f };
	expected[1] = { -0.355287f, 0.755016f, 0.155469f, -0.555016f };
	expected[2] = { -0.221760f, 0.619607f, 0.023460f, -0.419607f };
	expected[3] = { -0.057464f, 0.447948f, -0.133797f, -0.249066f };
	expected[4] = { 0.124829f, 0.244757f, -0.292839f, -0.051891f };
	CPtr<CDnnAdaptiveGradientSolver> adam = new CDnnAdaptiveGradientSolver( MathEngine() );
	adam->SetL1Regularization( 0.5f );
	adam->SetL2Regularization( 0 );
	adam->SetMomentDecayRate( 0.9f );
	adam->SetSecondMomentDecayRate( 0.999f );
	adam->SetEpsilon( 1e-8f );
	adam->SetLearningRate( 0.5f );
	adam->SetCompatibilityMode( true );

	testSolver( adam, expected );
}

TEST( CDnnSolverTest, AdamCompatL2 )
{
	CArray<CArray<float>> expected;
	expected.SetSize( 5 );
	expected[0] = { -0.450000f, 0.850000f, 0.250000f, -0.650000f };
	expected[1] = { -0.355230f, 0.755132f, 0.155331f, -0.555160f };
	expected[2] = { -0.221272f, 0.620465f, 0.022159f, -0.420689f };
	expected[3] = { -0.055124f, 0.451449f, -0.140531f, -0.252440f };
	expected[4] = { 0.133067f, 0.254512f, -0.317486f, -0.057773f };
	CPtr<CDnnAdaptiveGradientSolver> adam = new CDnnAdaptiveGradientSolver( MathEngine() );
	adam->SetL1Regularization( 0 );
	adam->SetL2Regularization( 0.5f );
	adam->SetMomentDecayRate( 0.9f );
	adam->SetSecondMomentDecayRate( 0.999f );
	adam->SetEpsilon( 1e-8f );
	adam->SetLearningRate( 0.5f );
	adam->SetCompatibilityMode( true );

	testSolver( adam, expected );
}

// ====================================================================================================================
// Nadam.

TEST( CDnnSolverTest, NadamNoReg )
{
	CArray<CArray<float>> expected;
	expected.SetSize( 5 );
	expected[0] = { 0.028226f, 0.371774f, -0.228226f, -0.171774f };
	expected[1] = { 0.374790f, 0.025210f, -0.574790f, 0.174790f };
	expected[2] = { 0.669376f, -0.269376f, -0.869376f, 0.469376f };
	expected[3] = { 0.937282f, -0.537282f, -1.137282f, 0.737282f };
	expected[4] = { 1.185859f, -0.785859f, -1.385859f, 0.985859f };
	CPtr<CDnnNesterovGradientSolver> adam = new CDnnNesterovGradientSolver( MathEngine() );
	adam->SetL1Regularization( 0 );
	adam->SetL2Regularization( 0 );
	adam->SetMomentDecayRate( 0.9f );
	adam->SetSecondMomentDecayRate( 0.999f );
	adam->SetEpsilon( 1e-8f );
	adam->SetLearningRate( 0.5f );

	testSolver( adam, expected );
}

TEST( CDnnSolverTest, NadamL1 )
{
	CArray<CArray<float>> expected;
	expected.SetSize( 5 );
	expected[0] = { -0.806726f, 0.504541f, 0.036822f, -1.039008f };
	expected[1] = { -0.990388f, 0.204585f, -0.046561f, -1.244838f };
	expected[2] = { -1.129002f, 0.054318f, -0.114567f, -1.402833f };
	expected[3] = { -1.242327f, -0.058420f, -0.176361f, -1.534744f };
	expected[4] = { -1.337595f, -0.153383f, -0.234135f, -1.648358f };
	CPtr<CDnnNesterovGradientSolver> adam = new CDnnNesterovGradientSolver( MathEngine() );
	adam->SetL1Regularization( 0.5f );
	adam->SetL2Regularization( 0 );
	adam->SetMomentDecayRate( 0.9f );
	adam->SetSecondMomentDecayRate( 0.999f );
	adam->SetEpsilon( 1e-8f );
	adam->SetLearningRate( 0.5f );

	testSolver( adam, expected );
}

TEST( CDnnSolverTest, NadamL2 )
{
	CArray<CArray<float>> expected;
	expected.SetSize( 5 );
	expected[0] = { -0.681726f, 0.529541f, 0.111822f, -0.964008f };
	expected[1] = { -0.825902f, 0.341368f, 0.003389f, -1.186084f };
	expected[2] = { -0.963771f, 0.189185f, -0.092676f, -1.411075f };
	expected[3] = { -1.107378f, 0.052502f, -0.184331f, -1.658277f };
	expected[4] = { -1.265272f, -0.074576f, -0.272888f, -1.943230f };
	CPtr<CDnnNesterovGradientSolver> adam = new CDnnNesterovGradientSolver( MathEngine() );
	adam->SetL1Regularization( 0 );
	adam->SetL2Regularization( 0.5f );
	adam->SetMomentDecayRate( 0.9f );
	adam->SetSecondMomentDecayRate( 0.999f );
	adam->SetEpsilon( 1e-8f );
	adam->SetLearningRate( 0.5f );

	testSolver( adam, expected );
}

static void checkBlobEquality( CDnnBlob& firstBlob, CDnnBlob& secondBlob )
{
	ASSERT_TRUE( firstBlob.HasEqualDimensions( &secondBlob ) );
	const int dataSize = firstBlob.GetDataSize();
	float* first = firstBlob.GetBuffer<float>( 0, dataSize, true );
	float* second = secondBlob.GetBuffer<float>( 0, dataSize, true );
	for( int i = 0; i < dataSize; ++i ) {
		EXPECT_TRUE( FloatEq( first[i], second[i], 1e-4f ) ) << first[i] << '\t' << second[i];
	}
	secondBlob.ReleaseBuffer( second, false );
	firstBlob.ReleaseBuffer( first, false );
}

static void solverSerializationTestImpl( CPtr<CDnnSolver> firstSolver, bool trainEveryStep )
{
	const int batchLength = 7;
	const int batchSize = 16;
	const int imageHeight = 32;
	const int imageWidth = 32;
	const int imageChannels = 3;
	const int classCount = 3;

	CRandom random( 0x1234 );
	CDnn firstNet( random, MathEngine() );
	firstNet.SetSolver( firstSolver );

	CPtr<CSourceLayer> source = AddLayer<CSourceLayer>( "source", firstNet );
	CPtr<CConvLayer> conv = AddLayer<CConvLayer>( "conv", { source } );
	conv->SetFilterCount( 5 );
	conv->SetFilterHeight( 3 );
	conv->SetFilterWidth( 3 );
	conv->SetPaddingHeight( 1 );
	conv->SetPaddingWidth( 1 );
	conv->SetStrideHeight( 2 );
	conv->SetStrideWidth( 2 );
	CPtr<CReLULayer> relu6 = AddLayer<CReLULayer>( "relu6", { conv } );
	relu6->SetUpperThreshold( 6.f );

	CPtr<CLstmLayer> direct = AddLayer<CLstmLayer>( "direct_lstm", { relu6 } );
	direct->SetHiddenSize( 15 );
	CPtr<CLstmLayer> reverse = AddLayer<CLstmLayer>( "reverse_lstm", { relu6 } );
	reverse->SetHiddenSize( 15 );
	reverse->SetReverseSequence( true );

	CPtr<CEltwiseSumLayer> sum = AddLayer<CEltwiseSumLayer>( "sum", { direct, reverse } );

	CPtr<CFullyConnectedLayer> fc = AddLayer<CFullyConnectedLayer>( "fc", { sum } );
	fc->SetNumberOfElements( classCount );

	CPtr<CSourceLayer> label = AddLayer<CSourceLayer>( "label", firstNet );
	CPtr<CCrossEntropyLossLayer> loss = AddLayer<CCrossEntropyLossLayer>( "loss", { fc, label } );

	CPtr<CDnnBlob> dataBlob = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, batchLength, batchSize,
		imageHeight, imageWidth, imageChannels );
	{
		float* data = dataBlob->GetBuffer<float>( 0, dataBlob->GetDataSize(), false );
		for( int i = 0; i < dataBlob->GetDataSize(); ++i ) {
			data[i] = static_cast< float >( random.Uniform( -1., 3. ) );
		}
		dataBlob->ReleaseBuffer( data, true );
	}
	source->SetBlob( dataBlob );

	CPtr<CDnnBlob> labelBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Int, batchLength, batchSize, 1 );
	{
		int* labelBuff = labelBlob->GetBuffer<int>( 0, labelBlob->GetDataSize(), false );
		for( int i = 0; i < labelBlob->GetDataSize(); ++i ) {
			labelBuff[i] = random.UniformInt( 0, classCount - 1 );
		}
		labelBlob->ReleaseBuffer( labelBuff, true );
	}
	label->SetBlob( labelBlob );

	for( int step = 1; step <= 10; ++step ) {
		firstNet.RunAndBackwardOnce();
		if( trainEveryStep || step % 4 == 0 ) {
			firstSolver->Train();
		}
	}

	// Cloning net and solver via serialization
	CDnn secondNet( random, MathEngine() );
	CString archiveFileName = "test_solver";
	{
		CArchiveFile file( archiveFileName, CArchive::store, GetPlatformEnv() );
		CArchive archive( &file, CArchive::SD_Storing );
		firstNet.SerializeCheckpoint( archive );
	}

	{
		CArchiveFile file( archiveFileName, CArchive::load, GetPlatformEnv() );
		CArchive archive( &file, CArchive::SD_Loading );
		secondNet.SerializeCheckpoint( archive );
	}

	CPtr<CDnnSolver> secondSolver = secondNet.GetSolver();
	CPtr<CSourceLayer> secondSource = CheckCast<CSourceLayer>( secondNet.GetLayer( "source" ) );
	secondSource->SetBlob( dataBlob );
	CPtr<CSourceLayer> secondLabel = CheckCast<CSourceLayer>( secondNet.GetLayer( "label" ) );
	secondLabel->SetBlob( labelBlob );
	CPtr<CCrossEntropyLossLayer> secondLoss = CheckCast<CCrossEntropyLossLayer>( secondNet.GetLayer( "loss" ) );

	for( int step = 11; step <= 20; ++step ) {
		firstNet.RunAndBackwardOnce();
		secondNet.RunAndBackwardOnce();
		if( trainEveryStep || step % 4 == 0 ) {
			firstSolver->Train();
			secondSolver->Train();
		}
		ASSERT_TRUE( FloatEq( loss->GetLastLoss(), secondLoss->GetLastLoss() ) );
	}

	CPtr<CConvLayer> secondConv = CheckCast<CConvLayer>( secondNet.GetLayer( "conv" ) );
	CPtr<CFullyConnectedLayer> secondFc = CheckCast<CFullyConnectedLayer>( secondNet.GetLayer( "fc" ) );
	CPtr<CLstmLayer> secondDirect = CheckCast<CLstmLayer>( secondNet.GetLayer( "direct_lstm" ) );
	CPtr<CLstmLayer> secondReverse = CheckCast<CLstmLayer>( secondNet.GetLayer( "reverse_lstm" ) );

	checkBlobEquality( *conv->GetFilterData(), *secondConv->GetFilterData() );
	checkBlobEquality( *conv->GetFreeTermData(), *secondConv->GetFreeTermData() );
	checkBlobEquality( *fc->GetWeightsData(), *secondFc->GetWeightsData() );
	checkBlobEquality( *fc->GetFreeTermData(), *secondFc->GetFreeTermData() );
	checkBlobEquality( *direct->GetRecurWeightsData(), *secondDirect->GetRecurWeightsData() );
	checkBlobEquality( *direct->GetInputWeightsData(), *secondDirect->GetInputWeightsData() );
	checkBlobEquality( *direct->GetRecurFreeTermData(), *secondDirect->GetRecurFreeTermData() );
	checkBlobEquality( *direct->GetInputFreeTermData(), *secondDirect->GetInputFreeTermData() );
	checkBlobEquality( *reverse->GetRecurWeightsData(), *secondReverse->GetRecurWeightsData() );
	checkBlobEquality( *reverse->GetInputWeightsData(), *secondReverse->GetInputWeightsData() );
	checkBlobEquality( *reverse->GetRecurFreeTermData(), *secondReverse->GetRecurFreeTermData() );
	checkBlobEquality( *reverse->GetInputFreeTermData(), *secondReverse->GetInputFreeTermData() );
}

TEST( CDnnSimpleGradientSolverTest, Serialization1 )
{
	CPtr<CDnnSimpleGradientSolver> sgd = new CDnnSimpleGradientSolver( MathEngine() );
	sgd->SetCompatibilityMode( false );
	sgd->SetL1Regularization( 1e-3 );
	sgd->SetL2Regularization( 0.f );
	sgd->SetLearningRate( 0.01f );
	sgd->SetMaxGradientNorm( 2.f );
	sgd->SetMomentDecayRate( 0.9 );
	solverSerializationTestImpl( sgd.Ptr(), true );
}

TEST( CDnnSimpleGradientSolverTest, Serialization2 )
{
	CPtr<CDnnSimpleGradientSolver> sgd = new CDnnSimpleGradientSolver( MathEngine() );
	sgd->SetCompatibilityMode( true );
	sgd->SetL1Regularization( 0.f );
	sgd->SetL2Regularization( 1e-2 );
	sgd->SetLearningRate( 0.05f );
	sgd->SetMaxGradientNorm( 3.f );
	sgd->SetMomentDecayRate( 0.999 );
	solverSerializationTestImpl( sgd.Ptr(), false );
}

TEST( CDnnAdaptiveGradientSolverTest, Serialization1 )
{
	CPtr<CDnnAdaptiveGradientSolver> adam = new CDnnAdaptiveGradientSolver( MathEngine() );
	adam->EnableAmsGrad( false );
	adam->EnableDecoupledWeightDecay( false );
	adam->SetCompatibilityMode( false );
	adam->SetEpsilon( 1e-3 );
	adam->SetL1Regularization( 1e-3 );
	adam->SetL2Regularization( 0.f );
	adam->SetLearningRate( 0.01f );
	adam->SetMaxGradientNorm( 2.f );
	adam->SetMomentDecayRate( 0.9 );
	adam->SetSecondMomentDecayRate( 0.777 );
	solverSerializationTestImpl( adam.Ptr(), true );
}

TEST( CDnnAdaptiveGradientSolverTest, Serialization2 )
{
	CPtr<CDnnAdaptiveGradientSolver> adam = new CDnnAdaptiveGradientSolver( MathEngine() );
	adam->EnableAmsGrad( true );
	adam->EnableDecoupledWeightDecay( true );
	adam->SetCompatibilityMode( true );
	adam->SetEpsilon( 3e-4 );
	adam->SetL1Regularization( 0.f );
	adam->SetL2Regularization( 1e-2 );
	adam->SetLearningRate( 0.05f );
	adam->SetMaxGradientNorm( 3.f );
	adam->SetMomentDecayRate( 0.999 );
	adam->SetSecondMomentDecayRate( 0.888 );
	solverSerializationTestImpl( adam.Ptr(), false );
}

TEST( CDnnNesterovGradientSolverTest, Serialization1 )
{
	CPtr<CDnnNesterovGradientSolver> nadam = new CDnnNesterovGradientSolver( MathEngine() );
	nadam->EnableAmsGrad( false );
	nadam->EnableDecoupledWeightDecay( false );
	nadam->SetEpsilon( 1e-3 );
	nadam->SetL1Regularization( 1e-3 );
	nadam->SetL2Regularization( 0.f );
	nadam->SetLearningRate( 0.01f );
	nadam->SetMaxGradientNorm( 2.f );
	nadam->SetMomentDecayRate( 0.9 );
	nadam->SetSecondMomentDecayRate( 0.777 );
	solverSerializationTestImpl( nadam.Ptr(), true );
}

TEST( CDnnNesterovGradientSolverTest, Serialization2 )
{
	CPtr<CDnnNesterovGradientSolver> nadam = new CDnnNesterovGradientSolver( MathEngine() );
	nadam->EnableAmsGrad( true );
	nadam->EnableDecoupledWeightDecay( true );
	nadam->SetEpsilon( 3e-4 );
	nadam->SetL1Regularization( 0.f );
	nadam->SetL2Regularization( 1e-2 );
	nadam->SetLearningRate( 0.05f );
	nadam->SetMaxGradientNorm( 3.f );
	nadam->SetMomentDecayRate( 0.999 );
	nadam->SetSecondMomentDecayRate( 0.888 );
	solverSerializationTestImpl( nadam.Ptr(), false );
}

TEST( CDnnLambGradientSolverTest, Serialization1 )
{
	CPtr<CDnnLambGradientSolver> lamb = new CDnnLambGradientSolver( MathEngine() );
	lamb->ExcludeWeightDecayLayer( "_lstm", CDnnLambGradientSolver::ELNMT_Include, 1 );
	lamb->SetEpsilon( 1e-3 );
	lamb->SetL1Regularization( 1e-4f );
	lamb->SetL2Regularization( 1e-3f );
	lamb->SetLearningRate( 0.024f );
	lamb->SetMaxGradientNorm( 2.f );
	lamb->SetMomentDecayRate( 0.99 );
	lamb->SetSecondMomentDecayRate( 0.222 );
	lamb->SetUseNVLamb( false );
	lamb->SetUseTrustRatio( false );
	lamb->SetWeightDecayClip( 0.75f );
	solverSerializationTestImpl( lamb.Ptr(), false );
}

TEST( CDnnLambGradientSolverTest, Serialization2 )
{
	CPtr<CDnnLambGradientSolver> lamb = new CDnnLambGradientSolver( MathEngine() );
	lamb->SetEpsilon( 2e-5 );
	lamb->SetL1Regularization( 1e-5f );
	lamb->SetL2Regularization( 1e-6f );
	lamb->SetLearningRate( 0.75f );
	lamb->SetMaxGradientNorm( 1.23f );
	lamb->SetMomentDecayRate( 0.9 );
	lamb->SetSecondMomentDecayRate( 0.99999 );
	lamb->SetUseNVLamb( true );
	lamb->SetUseTrustRatio( true );
	lamb->SetWeightDecayClip( 0.5f );
	solverSerializationTestImpl( lamb.Ptr(), true );
}

TEST( CDnnLambGradientSolverTest, Serialization3 )
{
	CPtr<CDnnLambGradientSolver> lamb = new CDnnLambGradientSolver( MathEngine() );
	lamb->SetEpsilon( 1e-6 );
	lamb->SetL1Regularization( 3e-5f );
	lamb->SetL2Regularization( 0 );
	lamb->SetLearningRate( 0.004f );
	lamb->SetMaxGradientNorm( -1.f );
	lamb->SetMomentDecayRate( 0.999 );
	lamb->SetSecondMomentDecayRate( 0.999 );
	lamb->SetUseNVLamb( true );
	lamb->SetUseTrustRatio( false );
	lamb->SetWeightDecayClip( 1.5f );
	solverSerializationTestImpl( lamb.Ptr(), false );
}

TEST( CDnnLambGradientSolverTest, Serialization4 )
{
	CPtr<CDnnLambGradientSolver> lamb = new CDnnLambGradientSolver( MathEngine() );
	lamb->SetEpsilon( 1e-4 );
	lamb->SetL1Regularization( 0 );
	lamb->SetL2Regularization( 1e-4f );
	lamb->SetLearningRate( 0.07f );
	lamb->SetMaxGradientNorm( 2.f );
	lamb->SetMomentDecayRate( 0.9 );
	lamb->SetSecondMomentDecayRate( 0.999 );
	lamb->SetUseNVLamb( false );
	lamb->SetUseTrustRatio( true );
	lamb->SetWeightDecayClip( 2.5f );
	solverSerializationTestImpl( lamb.Ptr(), true );
}
