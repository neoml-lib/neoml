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

struct CCrossEntropyLossLayerTestParams final {
	CCrossEntropyLossLayerTestParams( float target, float result, float lossValue ) :
		Target( target ), Result( result ), LossValue( lossValue ) {}

	float Target;
	float Result;
	float LossValue;
};

class CCrossEntropyLossLayerTest :
		public CNeoMLTestFixture, public ::testing::WithParamInterface<CCrossEntropyLossLayerTestParams> {
public:
	static bool InitTestFixture() { return true; }
	static void DeinitTestFixture() {}
};

} // namespace NeoMLTest

//---------------------------------------------------------------------------------------------------------------------

TEST_F( CCrossEntropyLossLayerTest, ZeroBackwardDiffTest )
{
	const auto met = MathEngine().GetType();
	if( met != MET_Cpu && met != MET_Cuda ) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		return;
	}

	const float rawData[8]{
		0.7f, 0.3f,
		0.3f, 0.7f,
		0.2f, 0.8f,
		0.4f, 0.6f
	};
	const float rawExpectedDiff[8]{
		-0.10032f, 0.10032f,
		0.10032f, -0.10032f,
		0.08859f, -0.08859f,
		0.0f, 0.0f
	};
	const float rawFloatLabels[8]{
		1.f, 0.f,
		0.f, 1.f,
		0.f, 1.f,
		0.f, 0.f
	};
	const int rawIntLabels[4]{ 0, 1, 1, -1 };

	CRandom random;
	CDnn dnn( random, MathEngine() );

	CPtr<CSourceLayer> data = Source( dnn, "data" );
	CPtr<CSourceLayer> label = Source( dnn, "label" );

	CPtr<CDnnSimpleTestDummyLearningLayer> learn = new CDnnSimpleTestDummyLearningLayer( MathEngine() );
	learn->SetName( "learn" );
	learn->Connect( *data );
	dnn.AddLayer( *learn );

	CPtr<CCrossEntropyLossLayer> loss = CrossEntropyLoss()( learn.Ptr(), label.Ptr() );

	CPtr<CDnnBlob> dataBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 2, 2, 2 );
	dataBlob->CopyFrom( rawData );
	data->SetBlob( dataBlob );

	CPtr<CDnnBlob> expectedDiff = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 2, 2, 2 );
	expectedDiff->CopyFrom( rawExpectedDiff );
	learn->ExpectedDiff = expectedDiff;

	CPtr<CDnnBlob> labelBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 2, 2, 2 );
	labelBlob->CopyFrom( rawFloatLabels );
	label->SetBlob( labelBlob );

	dnn.RunAndBackwardOnce();

	labelBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Int, 2, 2, 1 );
	labelBlob->CopyFrom( rawIntLabels );
	label->SetBlob( labelBlob );

	dnn.RunAndBackwardOnce();
}

TEST_F( CCrossEntropyLossLayerTest, NoSoftmax )
{
	const auto met = MathEngine().GetType();
	if( met != MET_Cpu && met != MET_Cuda ) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		return;
	}
	const float resultBuff[]{ 1.f / 15, 2.f / 15, 3.f / 15, 4.f / 15, 5.f / 15,
		5.f / 15, 4.f / 15, 3.f / 15, 2.f / 15, 1.f / 15,
		0.2f, 0.2f, 0.2f, 0.2f, 0.2f,
		0.2f, 0.2f, 0.2f, 0.2f, 0.2f,
		0.2f, 0.2f, 0.2f, 0.2f, 0.2f,
		0.2f, 0.2f, 0.2f, 0.2f, 0.2f,
		0.2f, 0.2f, 0.2f, 0.2f, 0.2f
	};

	CRandom random;
	CDnn dnn( random, MathEngine() );

	CPtr<CSourceLayer> result = Source( dnn, "result" );
	CPtr<CSourceLayer> target = Source( dnn, "target" );

	CPtr<CDnnSimpleTestDummyLearningLayer> learn = new CDnnSimpleTestDummyLearningLayer( MathEngine() );
	learn->Connect( *result );
	dnn.AddLayer( *learn );

	CPtr<CCrossEntropyLossLayer> loss = CrossEntropyLoss( /*softmax*/false )( learn.Ptr(), target.Ptr() );

	CPtr<CDnnBlob> resultBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 7, 5 );
	resultBlob->CopyFrom( resultBuff );
	result->SetBlob( resultBlob );

	CPtr<CDnnBlob> floatTargetBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 7, 5 );
	floatTargetBlob->Fill( 0.f );

	floatTargetBlob->GetData().SetValueAt( 2, 1 );
	floatTargetBlob->GetObjectData( 1 ).SetValueAt( 4, 1 );
	for( int i = 2; i < 7; ++i ) {
		floatTargetBlob->GetObjectData( i ).SetValueAt( 3, 1 );
	}
	target->SetBlob( floatTargetBlob );

	CPtr<CDnnBlob> expectedDiffBLob = resultBlob->GetCopy();
	expectedDiffBLob->Fill( 1.f / 7 );

	expectedDiffBLob->GetData().SetValueAt( 2, -4.f / 7 );
	expectedDiffBLob->GetObjectData( 1 ).SetValueAt( 4, -2 );
	for( int i = 2; i < 7; ++i ) {
		expectedDiffBLob->GetObjectData( i ).SetValueAt( 3, -4.f / 7 );
	}
	learn->ExpectedDiff = expectedDiffBLob;

	dnn.RunAndBackwardOnce();

	{
		dnn.DeleteLayer( *learn );
		loss->Connect( 0, *result );
		dnn.RunAndBackwardOnce();

		CMemoryFile file;
		{
			CArchive archive( &file, CArchive::store );
			archive.Serialize( dnn );
		}
		file.SeekToBegin();
		{
			CArchive archive( &file, CArchive::load );
			archive.Serialize( dnn );
		}

		result = CheckCast<CSourceLayer>( dnn.GetLayer( result->GetName() ) );
		target = CheckCast<CSourceLayer>( dnn.GetLayer( target->GetName() ) );
		loss = CheckCast<CCrossEntropyLossLayer>( dnn.GetLayer( loss->GetName() ) );

		result->SetBlob( resultBlob );

		learn->Connect( *result );
		loss->Connect( *learn );
		dnn.AddLayer( *learn );
	}

	CPtr<CDnnBlob> intTargetBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Int, 1, 7, 1 );
	intTargetBlob->FillObject<int>( 0, 2 );
	intTargetBlob->FillObject<int>( 1, 4 );
	for( int i = 2; i < 7; ++i ) {
		intTargetBlob->GetObjectData<int>( i ).SetValue( 3 );
	}
	target->SetBlob( intTargetBlob );

	dnn.RunAndBackwardOnce();
}

TEST_P( CCrossEntropyLossLayerTest, CrossEntropyLossSignTest )
{
	const CCrossEntropyLossLayerTestParams params = GetParam();

	CRandom random;
	CTextStream debugOutput;
	CDnn dnn( random, MathEngine() );
	dnn.SetLog( &debugOutput );

	CPtr<CSourceLayer> result = Source( dnn, "result" );
	CPtr<CSourceLayer> target = Source( dnn, "target" );

	CPtr<CBinaryCrossEntropyLossLayer> loss = BinaryCrossEntropyLoss()( result.Ptr(), target.Ptr() );

	CPtr<CDnnBlob> resultBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 1, 1 );
	resultBlob->Fill<float>( params.Result );
	result->SetBlob( resultBlob );

	CPtr<CDnnBlob> targetBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 1, 1 );
	targetBlob->Fill<float>( params.Target );
	target->SetBlob( targetBlob );

	dnn.RunAndBackwardOnce();

	EXPECT_TRUE( FloatEq( loss->GetLastLoss(), params.LossValue ) );
}

INSTANTIATE_TEST_CASE_P( CnnCrossEntropyLossTestInstantiation, CCrossEntropyLossLayerTest,
	::testing::Values(
		CCrossEntropyLossLayerTestParams( -1.f, -2.f, 0.126928f ),
		CCrossEntropyLossLayerTestParams( -1.f, -0.999f, 0.313531f ),
		CCrossEntropyLossLayerTestParams( -1.f, -1.f, 0.313262f ),
		CCrossEntropyLossLayerTestParams( -1.f, 1.f, 1.313262f ),
		CCrossEntropyLossLayerTestParams( 1.f, 1.f, 0.313262f ),
		CCrossEntropyLossLayerTestParams( 1.f, -1.f, 1.313262f ),
		CCrossEntropyLossLayerTestParams( 1.f, 3.1415926f, 4.2306e-2f ),
		CCrossEntropyLossLayerTestParams( -1.f, -0.451f, 0.49286f ),
		CCrossEntropyLossLayerTestParams( 1.f, 10000.f, 0.f ),
		CCrossEntropyLossLayerTestParams( -1.f, -10000.f, 0.f )
	)
);
