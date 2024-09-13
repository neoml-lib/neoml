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

using namespace NeoML;
using namespace NeoMLTest;

namespace NeoMLTest {

// Test data
static const int batchSize = 2;
static const int classCount = 5;

static const float inputData[]{
	0, 0.1f, 0.2f, 0.3f, 0.4f,
	0, 0.1f, 0.2f, 0.3f, 0.4f
};
static const float labelsData[]{
	0, 0, 1, 0, 0,
	0, 0, 0, 0, 1,
};

// Generate a correct input value (normalized to 1)
static void generateInput( float* input, int batchSize, int classCount )
{
	CRandom probabilitySource( 42 );
	for( int i = 0; i < batchSize; ++i ) {
		float total = 0.f;
		for( int j = 0; j < classCount; ++j ) {
			input[classCount * i + j] = static_cast<float>( exp( probabilitySource.Uniform( 0.f, 1.f ) ) );
			total += input[classCount * i + j];
		}
		for( int j = 0; j < classCount; ++j ) {
			input[classCount * i + j] /= total;
		}
	}
}

// Generating input values ​​and labels
static void generateDataset( float* input, float* labels, int batchSize, int classCount )
{
	generateInput( input, batchSize, classCount );
	CRandom labelSource( 42 );
	for( int i = 0; i < batchSize; ++i ) {
		for( int j = 0; j < classCount; ++j ) {
			labels[classCount * i + j] = 0.0f;
		}
		labels[classCount * i + labelSource.UniformInt( 0, classCount - 1 )] = 1.f;
	}
}

// Gradient calculation (manual)
static void calculateLossGradient( const float* input, const float* labels, int batchSize,
	int classCount, float focalForse, float* lossGradient )
{
	for( int i = 0; i < batchSize; ++i ) {
		float loss = 0.f;
		float probs = 0.f;
		float factor = 0.f;
		for( int j = 0; j < classCount; ++j ) {
			int index = i * classCount + j;
			probs += labels[index] * input[index];
		}
		factor = pow( 10.f - probs, focalForse - 1 );
		loss = logf( probs ) * factor * focalForse;
		factor = pow( 10.f - probs, focalForse );
		probs = max( probs, 1e-6f );
		factor = factor / probs;
		loss -= factor;
		for( int j = 0; j < classCount; ++j ) {
			int index = i * classCount + j;
			lossGradient[index] = labels[index] * loss;
		}
	}
}

// Loss calculation (manual)
static void calculateLoss( const float* input, const float* labels, int batchSize,
	int classCount, float focalForse, float* loss)
{
	for( int i = 0; i < batchSize; ++i ) {
		loss[i] = 0.f;
		float factor = 0.f;
		for( int j = 0; j < classCount; ++j ) {
			int index = i * classCount + j;
			loss[i] += labels[index] * input[index];
		}
		factor = pow(1.f - loss[i], focalForse);
		loss[i] = logf( loss[i] ) * factor * -1;
	}	
}

// Manual calculation of the square of the batch - averaged L2-norm between the linear approximation and the true value
static float calculateL2Diff( const float* input, const float* labelData, const float* gradient,
	const float* dataDelta, float focalForce, int batchSize, int classCount,
	float* newPoint, float* oldPointLoss, float* newPointLoss )
{
	for( int i = 0; i < batchSize * classCount; ++i ) {
		newPoint[i] = input[i] + dataDelta[i];
	}
	calculateLoss( input, labelData, batchSize, classCount, focalForce, oldPointLoss );
	calculateLoss( newPoint, labelData, batchSize, classCount, focalForce, newPointLoss );

	double totalL2 = 0.;
	for( int i = 0; i < batchSize; ++i ) {
		double sumOfGradientValues = 0.;
		for( int j = 0; j < classCount; ++j ) {
			int index = i * classCount + j;
			sumOfGradientValues += dataDelta[index] * gradient[index];
		}
		totalL2 += ( newPointLoss[i] - ( oldPointLoss[i] + sumOfGradientValues ) )
			* ( newPointLoss[i] - ( oldPointLoss[i] + sumOfGradientValues ) );
	}
	return static_cast<float>( totalL2 / batchSize );
}

} // namespace NeoMLTest

//---------------------------------------------------------------------------------------------------------------------

// Forward Test with default focalForce = 2
TEST( CFocalLossLayerTest, ForwardTest )
{
	float expectedLoss[batchSize];
	calculateLoss( inputData, labelsData, batchSize, classCount,
		CFocalLossLayer::DefaultFocalForceValue, expectedLoss );

	float expectedForward = 0.f;
	for( int i = 0; i < batchSize; ++i ) {
		expectedForward += expectedLoss[i];
	}
	expectedForward /= batchSize;

	CRandom random( 123456 );
	CDnn dnn( random, MathEngine() );

	CPtr<CSourceLayer> dataLayer = Source( dnn, "Data" );
	CPtr<CSourceLayer> labelLayer = Source( dnn, "Labels" );
	CPtr<CFocalLossLayer> lossLayer = FocalLoss()( dataLayer.Ptr(), labelLayer.Ptr() );
	EXPECT_EQ( CFocalLossLayer::DefaultFocalForceValue, lossLayer->GetFocalForce() );

	CPtr<CDnnBlob> sourceBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, batchSize, classCount );
	CPtr<CDnnBlob> labelBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, batchSize, classCount );

	dataLayer->SetBlob( sourceBlob );
	labelLayer->SetBlob( labelBlob );

	sourceBlob->CopyFrom<float>( inputData );
	labelBlob->CopyFrom<float>( labelsData );

	dnn.RunOnce();

	const float output = lossLayer->GetLastLoss();
	EXPECT_NEAR( expectedForward, output, 1e-3 );
}

// Backward Test
TEST( CFocalLossLayerTest, BackwardTest )
{
	const float focalForce = 2.0f;
	const float expectedBackward[] = {
		0, 0, 2 * 0.8f * -1.6094379f - 0.64f / 0.2f, 0, 0,
		0, 0, 0, 0, 2 * 0.6f * -0.9162907f - 0.36f / 0.4f
	};
	float newPoint[classCount * batchSize];
	float oldLoss[classCount * batchSize];
	float newLoss[classCount * batchSize];
	float delta[classCount * batchSize];
	for( int i = 0; i < classCount * batchSize; ++i ) {
		delta[i] = 1.f;
	}

	const float l2Expected = calculateL2Diff( inputData, labelsData, expectedBackward, delta, focalForce, batchSize,
		classCount, newPoint, oldLoss, newLoss );

	CPtr<CDnnBlob> sourceBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, batchSize, classCount );
	CPtr<CDnnBlob> labelBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, batchSize, classCount );

	CPtr<CFocalLossLayer> lossLayer = new CFocalLossLayer( MathEngine() );

	sourceBlob->CopyFrom<float>( inputData );
	labelBlob->CopyFrom<float>( labelsData );

	CFloatHandleStackVar deltaVar( MathEngine(), classCount * batchSize );
	MathEngine().DataExchangeTyped( deltaVar.GetHandle(), delta, classCount * batchSize );

	const float l2Actual = lossLayer->Test( batchSize, sourceBlob->GetData(), classCount,
		labelBlob->GetData(), classCount, deltaVar );
	EXPECT_TRUE( FloatEq( l2Actual, l2Expected, 1e-2f ) );
}

// Test on generated data
TEST( CFocalLossLayerTest, GeneratedTest )
{
	const int iterationCount = 1000;
	const int batchSize = 200;
	const int classCount = 20;
	float input[batchSize * classCount];
	float labels[batchSize * classCount];
	float loss[batchSize];
	float expectedGradient[batchSize * classCount];
	float newPoint[classCount * batchSize];
	float oldLoss[classCount * batchSize];
	float newLoss[classCount * batchSize];
	float delta[classCount * batchSize];

	CFloatHandleStackVar deltaVar( MathEngine(), classCount * batchSize );
	CFloatHandleStackVar gradient( MathEngine(), classCount * batchSize );

	CRandom random( 123456 );
	CDnn dnn( random, MathEngine() );

	CPtr<CSourceLayer> dataLayer = Source( dnn, "Data" );
	CPtr<CSourceLayer> labelLayer = Source( dnn, "Labels" );
	CPtr<CFocalLossLayer> lossLayer = FocalLoss()( dataLayer.Ptr(), labelLayer.Ptr() );

	CPtr<CDnnBlob> sourceBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, batchSize, classCount );
	CPtr<CDnnBlob> labelBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, batchSize, classCount );

	dataLayer->SetBlob( sourceBlob );
	labelLayer->SetBlob( labelBlob );

	// Test and check the value of loss and gradient
	for( int i = 0; i < iterationCount; ++i ) {
		const float focalForce = static_cast<float>( random.Uniform( 0.1, 6.0 ) );
		lossLayer->SetFocalForce( focalForce );
		EXPECT_EQ( focalForce, lossLayer->GetFocalForce() );

		generateDataset( input, labels, batchSize, classCount );
		// Loss and gradient calculation (manual)
		calculateLoss( input, labels, batchSize, classCount, focalForce, loss );
		float expectedLossValue = 0.f;
		for( int j = 0; j < batchSize; ++j ) {
			expectedLossValue += loss[j];
		}
		expectedLossValue /= batchSize;
		calculateLossGradient( input, labels, batchSize, classCount, focalForce, expectedGradient );

		generateInput( newPoint, batchSize, classCount );
		for( int j = 0; j < classCount * batchSize; ++j ) {
			delta[j] = newPoint[j] - input[j];
		}
		const float l2Expected = calculateL2Diff( input, labels, expectedGradient, delta, focalForce, batchSize,
			classCount, newPoint, oldLoss, newLoss );

		sourceBlob->CopyFrom<float>( input );
		labelBlob->CopyFrom<float>( labels );

		dnn.RunOnce();

		const float output = lossLayer->GetLastLoss();
		EXPECT_NEAR( expectedLossValue, output, 1e-3f );

		MathEngine().DataExchangeTyped( deltaVar.GetHandle(), delta, classCount * batchSize );
		const float l2Actual = lossLayer->Test( batchSize, sourceBlob->GetData(), classCount,
			labelBlob->GetData(), classCount, deltaVar );
		EXPECT_NEAR( l2Actual, l2Expected, 1e-3f );
	}
}
