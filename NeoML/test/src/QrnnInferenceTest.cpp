/* Copyright © 2017-2020 ABBYY Production LLC

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

static void naiveFPooling( int seqLength, int objSize, const float* z, const float* f, const float* h0, float* res )
{
	const float* hPrev = res;
	if( h0 == nullptr ) {
		for( int index = 0; index < objSize; ++index ) {
			*res = *z * ( 1.f - *f );
			++z;
			++f;
			++res;
		}
	} else {
		for( int index = 0; index < objSize; ++index ) {
			*res = *f * *h0 + ( 1.f - *f ) * *z;
			++z;
			++f;
			++h0;
			++res;
		}
	}

	for( int index = 0; index < objSize * ( seqLength - 1 ); ++index ) {
		*res = *f * *hPrev + ( 1.f - *f ) * *z;
		++z;
		++f;
		++hPrev;
		++res;
	}
}

class CFPoolingTestLayer : public CRecurrentLayer {
public:
	explicit CFPoolingTestLayer( IMathEngine& mathEngine, int objectSize );
};

CFPoolingTestLayer::CFPoolingTestLayer( IMathEngine& mathEngine, int objectSize ) :
	CRecurrentLayer( mathEngine, "fPooling" )
{
	CPtr<CBackLinkLayer> backLink = new CBackLinkLayer( mathEngine );
	backLink->SetDimSize( BD_Channels, objectSize );
	AddBackLink( *backLink );

	CPtr<CLinearLayer> negZ = new CLinearLayer( mathEngine );
	negZ->SetName( "negZ" );
	negZ->SetMultiplier( -1.f );
	negZ->SetFreeTerm( 0.f );
	SetInputMapping( 0, *negZ );
	AddLayer( *negZ );

	CPtr<CEltwiseSumLayer> prevMinusZ = new CEltwiseSumLayer( mathEngine );
	prevMinusZ->SetName( "prevMinusZ" );
	prevMinusZ->Connect( *backLink );
	prevMinusZ->Connect( 1, *negZ );
	AddLayer( *prevMinusZ );

	CPtr<CEltwiseMulLayer> fMulPrevMinusZ = new CEltwiseMulLayer( mathEngine );
	fMulPrevMinusZ->SetName( "fMulPrevMinusZ" );
	SetInputMapping( 1, *fMulPrevMinusZ );
	fMulPrevMinusZ->Connect( 1, *prevMinusZ );
	AddLayer( *fMulPrevMinusZ );

	CPtr<CEltwiseSumLayer> outputLayer = new CEltwiseSumLayer( mathEngine );
	outputLayer->SetName( "outputLayer" );
	outputLayer->Connect( *fMulPrevMinusZ );
	SetInputMapping( 0, *outputLayer, 1 );
	AddLayer( *outputLayer );
	SetOutputMapping( *outputLayer );

	backLink->Connect( *outputLayer );
	SetInputMapping( 2, *backLink, 1 );
}

static void fPoolingImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const CInterval batchLengthInterval = params.GetInterval( "BatchLength" );
	const CInterval batchWidthInterval = params.GetInterval( "BatchWidth" );
	const CInterval channelsInterval = params.GetInterval( "Channels" );

	const int batchLength = random.UniformInt( batchLengthInterval.Begin, batchLengthInterval.End );
	const int batchWidth = random.UniformInt( batchWidthInterval.Begin, batchWidthInterval.End );
	const int channels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const bool hasInitialState = random.Next() % 2 == 1;

	const int stateSize = channels * batchWidth;
	const int dataSize = batchLength * stateSize;

	CREATE_FILL_FLOAT_ARRAY( zData, -2.f, 2.f, dataSize, random );
	CREATE_FILL_FLOAT_ARRAY( fData, -2.f, 2.f, dataSize, random );
	CREATE_FILL_FLOAT_ARRAY( h0Data, -2.f, 2.f, stateSize, random );

	CDnn dnn( random, MathEngine() );

	CPtr<CDnnBlob> zBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, batchLength, batchWidth, channels );
	ASSERT_EQ( dataSize, zBlob->GetDataSize() );
	zBlob->CopyFrom( zData.GetPtr() );
	CPtr<CSourceLayer> zSource = new CSourceLayer( MathEngine() );
	zSource->SetName( "zSource" );
	dnn.AddLayer( *zSource );
	zSource->SetBlob( zBlob );

	CPtr<CDnnBlob> fBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, batchLength, batchWidth, channels );
	ASSERT_EQ( dataSize, fBlob->GetDataSize() );
	fBlob->CopyFrom( fData.GetPtr() );
	CPtr<CSourceLayer> fSource = new CSourceLayer( MathEngine() );
	fSource->SetName( "fSource" );
	dnn.AddLayer( *fSource );
	fSource->SetBlob( fBlob );

	CPtr<CFPoolingTestLayer> fPooling = new CFPoolingTestLayer( MathEngine(), channels );
	fPooling->Connect( 0, *zSource );
	fPooling->Connect( 1, *fSource );
	if( hasInitialState ) {
		CPtr<CDnnBlob> h0Blob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, batchWidth, channels );
		ASSERT_EQ( stateSize, h0Blob->GetDataSize() );
		h0Blob->CopyFrom( h0Data.GetPtr() );
		CPtr<CSourceLayer> h0Source = new CSourceLayer( MathEngine() );
		h0Source->SetName( "h0Source" );
		dnn.AddLayer( *h0Source );
		h0Source->SetBlob( h0Blob );
		fPooling->Connect( 2, *h0Source );
	}
	dnn.AddLayer( *fPooling );

	CPtr<CSinkLayer> sink = new CSinkLayer( MathEngine() );
	sink->Connect( *fPooling );
	dnn.AddLayer( *sink );

	CArray<float> expected;
	expected.SetSize( dataSize );

	naiveFPooling( batchLength, stateSize, zData.GetPtr(), fData.GetPtr(),
		hasInitialState ? h0Data.GetPtr() : nullptr,
		expected.GetPtr() );

	dnn.RunOnce();

	CPtr<CDnnBlob> actualBlob = sink->GetBlob();
	ASSERT_EQ( dataSize, actualBlob->GetDataSize() );
	float* actual = actualBlob->GetBuffer<float>( 0, dataSize );
	for( int i = 0; i < dataSize; ++i ) {
		EXPECT_TRUE( FloatEq( expected[i], actual[i], 1e-4f ) );
	}
	actualBlob->ReleaseBuffer( actual, false );
}

//------------------------------------------------------------------------------------------------------------

static void naiveIfPooling( int seqLength, int objSize, const float* z, const float* f, const float* i,
	const float* h0, float* res )
{
	const float* hPrev = res;
	if( h0 == nullptr ) {
		for( int index = 0; index < objSize; ++index ) {
			*res = *z * *i;
			++z;
			++f;
			++i;
			++res;
		}
	} else {
		for( int index = 0; index < objSize; ++index ) {
			*res = *h0 * *f + *i * *z;
			++z;
			++f;
			++i;
			++h0;
			++res;
		}
	}

	for( int index = 0; index < ( seqLength - 1 ) * objSize; ++index ) {
		*res = *hPrev * *f + *i * *z;
		++z;
		++f;
		++i;
		++hPrev;
		++res;
	}
}

class CIfPoolingTestLayer : public CRecurrentLayer {
public:
	explicit CIfPoolingTestLayer( IMathEngine& mathEngine, int objectSize );
};

CIfPoolingTestLayer::CIfPoolingTestLayer( IMathEngine& mathEngine, int objectSize ) :
	CRecurrentLayer( mathEngine, "fPooling" )
{
	CPtr<CBackLinkLayer> backLink = new CBackLinkLayer( mathEngine );
	backLink->SetDimSize( BD_Channels, objectSize );
	AddBackLink( *backLink );

	CPtr<CEltwiseMulLayer> fMulPrev = new CEltwiseMulLayer( mathEngine );
	fMulPrev->SetName( "fMulPrev" );
	SetInputMapping( 1, *fMulPrev );
	fMulPrev->Connect( 1, *backLink );
	AddLayer( *fMulPrev );

	CPtr<CEltwiseMulLayer> iMulZ = new CEltwiseMulLayer( mathEngine );
	iMulZ->SetName( "iMulZ" );
	SetInputMapping( 2, *iMulZ );
	SetInputMapping( 0, *iMulZ, 1 );
	AddLayer( *iMulZ );

	CPtr<CEltwiseSumLayer> outputLayer = new CEltwiseSumLayer( mathEngine );
	outputLayer->SetName( "outputLayer" );
	outputLayer->Connect( 0, *fMulPrev );
	outputLayer->Connect( 1, *iMulZ );
	AddLayer( *outputLayer );
	SetOutputMapping( *outputLayer );

	backLink->Connect( *outputLayer );
	SetInputMapping( 3, *backLink, 1 );
}

static void ifPoolingImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const CInterval batchLengthInterval = params.GetInterval( "BatchLength" );
	const CInterval batchWidthInterval = params.GetInterval( "BatchWidth" );
	const CInterval channelsInterval = params.GetInterval( "Channels" );

	const int batchLength = random.UniformInt( batchLengthInterval.Begin, batchLengthInterval.End );
	const int batchWidth = random.UniformInt( batchWidthInterval.Begin, batchWidthInterval.End );
	const int channels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const bool hasInitialState = random.Next() % 2 == 1;

	const int stateSize = channels * batchWidth;
	const int dataSize = batchLength * stateSize;

	CREATE_FILL_FLOAT_ARRAY( zData, -2.f, 2.f, dataSize, random );
	CREATE_FILL_FLOAT_ARRAY( fData, -2.f, 2.f, dataSize, random );
	CREATE_FILL_FLOAT_ARRAY( iData, -2.f, 2.f, dataSize, random );
	CREATE_FILL_FLOAT_ARRAY( h0Data, -2.f, 2.f, stateSize, random );

	CDnn dnn( random, MathEngine() );

	CPtr<CDnnBlob> zBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, batchLength, batchWidth, channels );
	ASSERT_EQ( dataSize, zBlob->GetDataSize() );
	zBlob->CopyFrom( zData.GetPtr() );
	CPtr<CSourceLayer> zSource = new CSourceLayer( MathEngine() );
	zSource->SetName( "zSource" );
	dnn.AddLayer( *zSource );
	zSource->SetBlob( zBlob );

	CPtr<CDnnBlob> fBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, batchLength, batchWidth, channels );
	ASSERT_EQ( dataSize, fBlob->GetDataSize() );
	fBlob->CopyFrom( fData.GetPtr() );
	CPtr<CSourceLayer> fSource = new CSourceLayer( MathEngine() );
	fSource->SetName( "fSource" );
	dnn.AddLayer( *fSource );
	fSource->SetBlob( fBlob );

	CPtr<CDnnBlob> iBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, batchLength, batchWidth, channels );
	ASSERT_EQ( dataSize, iBlob->GetDataSize() );
	iBlob->CopyFrom( iData.GetPtr() );
	CPtr<CSourceLayer> iSource = new CSourceLayer( MathEngine() );
	iSource->SetName( "iSource" );
	dnn.AddLayer( *iSource );
	iSource->SetBlob( iBlob );

	CPtr<CIfPoolingTestLayer> ifPooling = new CIfPoolingTestLayer( MathEngine(), channels );
	ifPooling->Connect( 0, *zSource );
	ifPooling->Connect( 1, *fSource );
	ifPooling->Connect( 2, *iSource );
	if( hasInitialState ) {
		CPtr<CDnnBlob> h0Blob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, batchWidth, channels );
		ASSERT_EQ( stateSize, h0Blob->GetDataSize() );
		h0Blob->CopyFrom( h0Data.GetPtr() );
		CPtr<CSourceLayer> h0Source = new CSourceLayer( MathEngine() );
		h0Source->SetName( "h0Source" );
		dnn.AddLayer( *h0Source );
		h0Source->SetBlob( h0Blob );
		ifPooling->Connect( 3, *h0Source );
	}
	dnn.AddLayer( *ifPooling );

	CPtr<CSinkLayer> sink = new CSinkLayer( MathEngine() );
	sink->Connect( *ifPooling );
	dnn.AddLayer( *sink );

	CArray<float> expected;
	expected.SetSize( dataSize );

	naiveIfPooling( batchLength, stateSize, zData.GetPtr(), fData.GetPtr(), iData.GetPtr(),
		hasInitialState ? h0Data.GetPtr() : nullptr,
		expected.GetPtr() );

	dnn.RunOnce();

	CPtr<CDnnBlob> actualBlob = sink->GetBlob();
	ASSERT_EQ( dataSize, actualBlob->GetDataSize() );
	float* actual = actualBlob->GetBuffer<float>( 0, dataSize );
	for( int i = 0; i < dataSize; ++i ) {
		EXPECT_TRUE( FloatEq( expected[i], actual[i], 1e-4f ) );
	}
	actualBlob->ReleaseBuffer( actual, false );
}

//------------------------------------------------------------------------------------------------------------

class CQrnnTest : public CNeoMlTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CQrnnTest, CQrnnTest,
	::testing::Values(
		CTestParams(
			"BatchLength = (2..20);"
			"BatchWidth = (1..10);"
			"Channels = (1..10);"
			"TestCount = 100;"
		)
	)
);

TEST_P( CQrnnTest, FPoolingInferenceRandom )
{
	RUN_TEST_IMPL( fPoolingImpl );
}

TEST_P( CQrnnTest, IfPoolingInferenceRandom )
{
	RUN_TEST_IMPL( ifPoolingImpl );
}
