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

enum { TP_PositivesCorrect, TP_PositivesTotal, TP_NegativesCorrect, TP_NegativesTotal, TP_Count };

static void precisionRecallNaive( const float* calculLogit, const float* groundTruth, int vectorSize, int* current )
{
	for( int i = 0; i < vectorSize; ++i ) {
		if( groundTruth[i] >= 1.f ) {
			++current[TP_PositivesTotal];
			if( calculLogit[i] == groundTruth[i] ) {
				++current[TP_PositivesCorrect];
			}
		} else {
			++current[TP_NegativesTotal];
			if( calculLogit[i] == groundTruth[i] ) {
				++current[TP_NegativesCorrect];
			}
		}
	}
}

//---------------------------------------------------------------------------------------------------------------------

struct CPrecisionRecallLayerTest : public CNeoMlTestFixtureWithParams {
	CPrecisionRecallLayerTest();

	CArray<float> CalculLogit;
	CArray<float> GroundTruth;
	CArray<int> Result;
	CArray<int> Expected;

	CRandom Random;
	CDnn Dnn;

	CSourceLayer* Data;
	CSourceLayer* Truth;
	CPrecisionRecallLayer* PrecRecall;
	CSinkLayer* Sink;

	void CheckExpected();
	void TestImpl( int vectorSize );
	void ParameterizedTestImpl( const CTestParams& params, int seed );
};

CPrecisionRecallLayerTest::CPrecisionRecallLayerTest() :
	Dnn( Random, MathEngine() ),
	Data( Source( Dnn, "data" ) ),
	Truth( Source( Dnn, "truth" ) ),
	PrecRecall( PrecisionRecall()( Data, Truth ) ),
	Sink( ::NeoML::Sink( PrecRecall, "sink" ) )
{
	CalculLogit.SetBufferSize( 100 );
	GroundTruth.SetBufferSize( 100 );
	Expected.SetSize( TP_Count );
	PrecRecall->SetReset( false );
}

void CPrecisionRecallLayerTest::CheckExpected()
{
	float Current[TP_Count];
	Sink->GetBlob()->CopyTo( Current );
	PrecRecall->GetLastResult( Result );

	for( int i = 0; i < TP_Count; ++i ) {
		EXPECT_EQ( Expected[i], static_cast<int>( Current[i] ) ) << i;
		EXPECT_EQ( Expected[i], Result[i] ) << i;
	}
}

void CPrecisionRecallLayerTest::TestImpl( int vectorSize )
{
	CPtr<CDnnBlob> logitData = CDnnBlob::CreateVector( MathEngine(), CT_Float, vectorSize );
	CPtr<CDnnBlob> truthData = CDnnBlob::CreateVector( MathEngine(), CT_Float, vectorSize );

	Data->SetBlob( logitData );
	Truth->SetBlob( truthData );

	logitData->CopyFrom( CalculLogit.GetPtr() );
	truthData->CopyFrom( GroundTruth.GetPtr() );

	precisionRecallNaive( CalculLogit.GetPtr(), GroundTruth.GetPtr(), vectorSize, Expected.GetPtr() );

	Dnn.RunOnce();
	CheckExpected();

	PrecRecall->SetReset( true );

	Expected.DeleteAll();
	Expected.SetSize( TP_Count );
	precisionRecallNaive( CalculLogit.GetPtr(), GroundTruth.GetPtr(), vectorSize, Expected.GetPtr() );

	Dnn.RunOnce();
	CheckExpected();

	PrecRecall->SetReset( false );
}

void CPrecisionRecallLayerTest::ParameterizedTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval vectorSizeInterval = params.GetInterval( "VectorSize" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int vectorSize = random.UniformInt( vectorSizeInterval.Begin, vectorSizeInterval.End );
	CalculLogit.SetSize( vectorSize );
	GroundTruth.SetSize( vectorSize );

	for( int i = 0, j = 0; i < vectorSize || j < vectorSize; ) {
		const float logit = static_cast<float>( random.UniformInt( valuesInterval.Begin, valuesInterval.End ) );
		if( i < vectorSize && logit != 0 )
			CalculLogit[i++] = logit;

		const float truth = static_cast<float>( random.UniformInt( valuesInterval.Begin, valuesInterval.End ) );
		if( j < vectorSize && truth != 0 )
			GroundTruth[j++] = truth;
	}

	TestImpl( vectorSize );
}

} // namespace NeoMLTest

//---------------------------------------------------------------------------------------------------------------------

// Simple Test
TEST_F( CPrecisionRecallLayerTest, SimpleTest )
{
	const int vectorSize = 6;
	CalculLogit = { 1,  1, -1, 1, -1, -1 };
	GroundTruth = { 1, -1, -1, 1, -1,  1 };

	TestImpl( vectorSize );
}

// Test on generated data
TEST_P( CPrecisionRecallLayerTest, GeneratedTest )
{
	RUN_TEST_IMPL( ParameterizedTestImpl );
}

INSTANTIATE_TEST_CASE_P( CPrecisionRecallLayerTestInstantiation, CPrecisionRecallLayerTest,
	::testing::Values(
		CTestParams(
			"VectorSize = (30..100);"
			"Values = (-1..1);"
			"TestCount = 10;"
		)
	)
);

