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

static void naiveFPooling( bool reverse, int seqLength, int objSize,
	const float* z, const float* f, const float* h0,
	float* res )
{
	const int objOffset = reverse ? -objSize : objSize;

	if( reverse ) {
		const int firstElemOffset = ( seqLength - 1 ) * objSize;
		z += firstElemOffset;
		f += firstElemOffset;
		res += firstElemOffset;
	}

	if( h0 == nullptr ) {
		for( int index = 0; index < objSize; ++index ) {
			res[index] = z[index] * ( 1.f - f[index] );
		}
	} else {
		for( int index = 0; index < objSize; ++index ) {
			res[index] = f[index] * h0[index] + ( 1.f - f[index] ) * z[index];
		}
	}

	const float* hPrev = res;

	for( int step = 0; step < seqLength - 1; ++step ) {
		z += objOffset;
		f += objOffset;
		res += objOffset;
		for( int index = 0; index < objSize; ++index ) {
			res[index] = f[index] * hPrev[index] + ( 1.f - f[index] ) * z[index];
		}
		hPrev += objOffset;
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
	const bool reverse = random.Next() % 2 == 1;

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
	fPooling->SetReverseSequence( reverse );
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

	naiveFPooling( reverse, batchLength, stateSize, zData.GetPtr(), fData.GetPtr(),
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

static void naiveIfPooling( bool reverse, int seqLength, int objSize,
	const float* z, const float* f, const float* i, const float* h0,
	float* res )
{
	const int objOffset = reverse ? -objSize : objSize;

	if( reverse ) {
		const int firstElemOffset = ( seqLength - 1 ) * objSize;
		z += firstElemOffset;
		f += firstElemOffset;
		i += firstElemOffset;
		res += firstElemOffset;
	}

	if( h0 == nullptr ) {
		for( int index = 0; index < objSize; ++index ) {
			res[index] = i[index] * z[index];
		}
	} else {
		for( int index = 0; index < objSize; ++index ) {
			res[index] = f[index] * h0[index] + i[index] * z[index];
		}
	}

	const float* hPrev = res;

	for( int step = 0; step < seqLength - 1; ++step ) {
		z += objOffset;
		f += objOffset;
		i += objOffset;
		res += objOffset;
		for( int index = 0; index < objSize; ++index ) {
			res[index] = f[index] * hPrev[index] + i[index] * z[index];
		}
		hPrev += objOffset;
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
	const bool reverse = random.Next() % 2 == 1;

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
	ifPooling->SetReverseSequence( reverse );
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

	naiveIfPooling( reverse, batchLength, stateSize, zData.GetPtr(), fData.GetPtr(), iData.GetPtr(),
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

class CDummyLearnLayer : public CBaseLayer {
public:
	explicit CDummyLearnLayer( IMathEngine& mathEngine ) : CBaseLayer( mathEngine, "CDummyLearnLayer", true ) {}

	CPtr<CDnnBlob> ExpectedDiff;
	CPtr<CDnnBlob> ActualDiff;

protected:
	void Reshape() override
	{
		NeoAssert( GetInputCount() == 1 );
		outputDescs[0] = inputDescs[0];
	}

	void RunOnce() override
	{
		outputBlobs[0]->CopyFrom( inputBlobs[0] );
	}

	void BackwardOnce() override
	{
		MathEngine().VectorCopy( inputDiffBlobs[0]->GetData(),
			outputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetDataSize() );
	}

	void LearnOnce() override
	{
		if( ActualDiff != nullptr && ActualDiff->HasEqualDimensions( outputDiffBlobs[0] ) ) {
			ActualDiff->CopyFrom( outputDiffBlobs[0] );
		} else {
			ActualDiff = outputDiffBlobs[0]->GetCopy();
		}

		if(ExpectedDiff.Ptr() == 0) {
			return;
		}

		ASSERT_EQ(ExpectedDiff->GetObjectCount(), outputDiffBlobs[0]->GetObjectCount());
		ASSERT_EQ(ExpectedDiff->GetHeight(), outputDiffBlobs[0]->GetHeight());
		ASSERT_EQ(ExpectedDiff->GetWidth(), outputDiffBlobs[0]->GetWidth());
		ASSERT_EQ(ExpectedDiff->GetDepth(), outputDiffBlobs[0]->GetDepth());
		ASSERT_EQ(ExpectedDiff->GetChannelsCount(), outputDiffBlobs[0]->GetChannelsCount());

		CArray<float> expectedDiffBuf;
		expectedDiffBuf.SetSize(ExpectedDiff->GetDataSize());
		ExpectedDiff->CopyTo(expectedDiffBuf.GetPtr(), expectedDiffBuf.Size());

		CArray<float> oputputDiffBuf;
		oputputDiffBuf.SetSize(outputDiffBlobs[0]->GetDataSize());
		outputDiffBlobs[0]->CopyTo(oputputDiffBuf.GetPtr(), oputputDiffBuf.Size());

		for(int i = 0; i < ExpectedDiff->GetDataSize(); ++i) {
			ASSERT_TRUE(FloatEq(expectedDiffBuf[i], oputputDiffBuf[i], 1e-4));
		}
	}
};

class CDummyLossLayer : public CLossLayer {
public:
	explicit CDummyLossLayer( IMathEngine& mathEngine ) : CLossLayer( mathEngine, "CDummyLossLayer" ) {}

	CPtr<CDnnBlob> Diff;

protected:
	void BatchCalculateLossAndGradient( int batchSize, CConstFloatHandle, int vectorSize, CConstIntHandle, int,
		CFloatHandle lossValue, CFloatHandle lossGradient ) override
	{
		int totalSize = batchSize * vectorSize;
		MathEngine().VectorFill( lossValue, 0, batchSize );

		if( Diff.Ptr() == 0 ) {
			MathEngine().VectorFill( lossGradient, 0, totalSize );
		} else {
			// anti-множитель для того, чтобы покрыть эффект от множителя в базовом loss-слое
			CFloatHandleStackVar antiMult( MathEngine(), batchSize );
			MathEngine().VectorFill( antiMult, batchSize / GetLossWeight(), batchSize );
			MathEngine().VectorEltwiseDivide( antiMult, GetWeights()->GetData(), antiMult, batchSize );

			ASSERT_EQ( Diff->GetDataSize(), totalSize );
			MathEngine().MultiplyDiagMatrixByMatrix( antiMult, batchSize,
				Diff->GetData(), vectorSize, lossGradient, totalSize );
		}
	}

	void BatchCalculateLossAndGradient(int batchSize, CConstFloatHandle data, int vectorSize, CConstFloatHandle label,
		int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient) override
	{
		BatchCalculateLossAndGradient( batchSize, data, vectorSize, label, labelSize, lossValue, lossGradient, CFloatHandle() );
	}

	void BatchCalculateLossAndGradient( int batchSize, CConstFloatHandle data, int vectorSize, CConstFloatHandle label,
		int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient, CFloatHandle labelLossGradient ) override
	{
		if(inputBlobs.Size() > 1) {
			ASSERT_TRUE(inputBlobs[0]->HasEqualDimensions(inputBlobs[1]));
		}
		ASSERT_TRUE(labelSize == vectorSize);

		int totalSize = batchSize * vectorSize;

		CArray<float> buf0;
		buf0.SetSize(totalSize);
		MathEngine().DataExchangeRaw(buf0.GetPtr(), data, buf0.Size() * sizeof( float ) );
		CArray<float> buf1;
		buf1.SetSize(totalSize);
		MathEngine().DataExchangeRaw(buf1.GetPtr(), label, buf1.Size() * sizeof( float ));

		for( int i = 0; i < totalSize; ++i ) {
			ASSERT_TRUE(FloatEq(buf0[i], buf1[i], 1e-2f));
		}

		MathEngine().VectorFill(lossValue, 0, batchSize);

		if(lossGradient.IsNull()) {
			return;
		}

		if(Diff.Ptr() == 0) {
			MathEngine().VectorFill(lossGradient, 0, totalSize);
		} else {
			// anti-множитель для того, чтобы покрыть эффект от множителя в базовом loss-слое
			CFloatHandleStackVar antiMult( MathEngine(), batchSize );
			MathEngine().VectorFill(antiMult, batchSize / GetLossWeight(), batchSize);
			MathEngine().VectorEltwiseDivide(antiMult, GetWeights()->GetData(), antiMult, batchSize);

			ASSERT_EQ(Diff->GetDataSize(), totalSize);
			MathEngine().MultiplyDiagMatrixByMatrix(antiMult, batchSize,
				Diff->GetData(), vectorSize, lossGradient, totalSize);
		}

		if( labelLossGradient.IsNull() ) {
			return;
		}

		MathEngine().VectorCopy( labelLossGradient, lossGradient, totalSize );
	}
};

//------------------------------------------------------------------------------------------------------------

static void naiveFPoolingBackward( bool reverse, int seqLength, int objSize,
	const float* z, const float* f, const float* h0,
	const float* out, float* outDiff,
	float* zDiff, float* fDiff )
{
	const int objOffset = reverse ? -objSize : objSize;

	if( reverse ) {
		const int firstElemOffset = ( seqLength - 1 ) * objSize;
		z += firstElemOffset;
		f += firstElemOffset;
		out += firstElemOffset;
		outDiff += firstElemOffset;
		zDiff += firstElemOffset;
		fDiff += firstElemOffset;
	}

	for( int step = 0; step < seqLength - 1; ++step ) {
		for( int index = 0; index < objSize; ++index ) {
			zDiff[index] = outDiff[index] * ( 1.f - f[index] );
			fDiff[index] = outDiff[index] * ( out[objOffset + index] - z[index] );
			outDiff[objOffset + index] += outDiff[index] * f[index];
		}
		z += objOffset;
		f += objOffset;
		out += objOffset;
		outDiff += objOffset;
		zDiff += objOffset;
		fDiff += objOffset;
	}

	if( h0 == nullptr ) {
		for( int index = 0; index < objSize; ++index ) {
			zDiff[index] = outDiff[index] * ( 1.f - f[index] );
			fDiff[index] = -z[index] * outDiff[index];
		}
	} else {
		for( int index = 0; index < objSize; ++index ) {
			zDiff[index] = outDiff[index] * ( 1.f - f[index] );
			fDiff[index] = outDiff[index] * ( h0[index] - z[index] );
		}
	}
}

static void fPoolingBackwardImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const CInterval batchLengthInterval = params.GetInterval( "BatchLength" );
	const CInterval batchWidthInterval = params.GetInterval( "BatchWidth" );
	const CInterval channelsInterval = params.GetInterval( "Channels" );

	const int batchLength = random.UniformInt( batchLengthInterval.Begin, batchLengthInterval.End );
	const int batchWidth = random.UniformInt( batchWidthInterval.Begin, batchWidthInterval.End );
	const int channels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const bool hasInitialState = random.Next() % 2 == 1;
	const bool reverse = random.Next() % 2 == 1;

	const int stateSize = channels * batchWidth;
	const int dataSize = batchLength * stateSize;

	CREATE_FILL_FLOAT_ARRAY( zData, -2.f, 2.f, dataSize, random );
	CREATE_FILL_FLOAT_ARRAY( fData, -2.f, 2.f, dataSize, random );
	CREATE_FILL_FLOAT_ARRAY( h0Data, -2.f, 2.f, stateSize, random );
	CREATE_FILL_FLOAT_ARRAY( outDiffData, -2.f, 2.f, dataSize, random );

	CDnn dnn( random, MathEngine() );

	CPtr<CDnnBlob> zBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, batchLength, batchWidth, channels );
	ASSERT_EQ( dataSize, zBlob->GetDataSize() );
	zBlob->CopyFrom( zData.GetPtr() );
	CPtr<CSourceLayer> zSource = new CSourceLayer( MathEngine() );
	zSource->SetName( "zSource" );
	dnn.AddLayer( *zSource );
	zSource->SetBlob( zBlob );
	CPtr<CDummyLearnLayer> zLearn = new CDummyLearnLayer( MathEngine() );
	zLearn->SetName( "zLearn" );
	zLearn->Connect( *zSource );
	dnn.AddLayer( *zLearn );

	CPtr<CDnnBlob> fBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, batchLength, batchWidth, channels );
	ASSERT_EQ( dataSize, fBlob->GetDataSize() );
	fBlob->CopyFrom( fData.GetPtr() );
	CPtr<CSourceLayer> fSource = new CSourceLayer( MathEngine() );
	fSource->SetName( "fSource" );
	dnn.AddLayer( *fSource );
	fSource->SetBlob( fBlob );
	CPtr<CDummyLearnLayer> fLearn = new CDummyLearnLayer( MathEngine() );
	fLearn->SetName( "fLearn" );
	fLearn->Connect( *fSource );
	dnn.AddLayer( *fLearn );

	CPtr<CFPoolingTestLayer> fPooling = new CFPoolingTestLayer( MathEngine(), channels );
	fPooling->SetReverseSequence( reverse );
	fPooling->Connect( 0, *zLearn );
	fPooling->Connect( 1, *fLearn );
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

	CArray<float> outData;
	outData.SetSize( dataSize );
	naiveFPooling( reverse, batchLength, stateSize, zData.GetPtr(), fData.GetPtr(),
		hasInitialState ? h0Data.GetPtr() : nullptr,
		outData.GetPtr() );
	CPtr<CDnnBlob> outBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, batchLength, batchWidth, channels );
	ASSERT_EQ( dataSize, outBlob->GetDataSize() );
	outBlob->CopyFrom( outData.GetPtr() );
	CPtr<CSourceLayer> outSource = new CSourceLayer( MathEngine() );
	outSource->SetName( "outSource" );
	dnn.AddLayer( *outSource );
	outSource->SetBlob( outBlob );

	CPtr<CDnnBlob> outDiffBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, batchLength, batchWidth, channels );
	ASSERT_EQ( dataSize, outDiffBlob->GetDataSize() );
	outDiffBlob->CopyFrom( outDiffData.GetPtr() );
	CPtr<CDummyLossLayer> loss = new CDummyLossLayer( MathEngine() );
	loss->Connect( *fPooling );
	loss->Connect( 1, *outSource );
	dnn.AddLayer( *loss );
	loss->Diff = outDiffBlob;

	CArray<float> zDiffData;
	zDiffData.SetSize( dataSize );
	CArray<float> fDiffData;
	fDiffData.SetSize( dataSize );
	naiveFPoolingBackward( !reverse, batchLength, stateSize, zData.GetPtr(), fData.GetPtr(),
		hasInitialState ? h0Data.GetPtr() : nullptr,
		outData.GetPtr(), outDiffData.GetPtr(), zDiffData.GetPtr(), fDiffData.GetPtr() );

	CPtr<CDnnBlob> zDiffBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, batchLength, batchWidth, channels );
	ASSERT_EQ( dataSize, zDiffBlob->GetDataSize() );
	zDiffBlob->CopyFrom( zDiffData.GetPtr() );
	zLearn->ExpectedDiff = zDiffBlob;

	CPtr<CDnnBlob> fDiffBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, batchLength, batchWidth, channels );
	ASSERT_EQ( dataSize, fDiffBlob->GetDataSize() );
	fDiffBlob->CopyFrom( fDiffData.GetPtr() );
	fLearn->ExpectedDiff = fDiffBlob;

	dnn.RunAndLearnOnce();
}

//------------------------------------------------------------------------------------------------------------

static void naiveIfPoolingBackward( bool reverse, int seqLength, int objSize,
	const float* z, const float* f, const float* i, const float* h0,
	const float* out, float* outDiff,
	float* zDiff, float* fDiff, float* iDiff )
{
	const int objOffset = reverse ? -objSize : objSize;

	if( reverse ) {
		const int firstElemOffset = ( seqLength - 1 ) * objSize;
		z += firstElemOffset;
		f += firstElemOffset;
		i += firstElemOffset;
		out += firstElemOffset;
		outDiff += firstElemOffset;
		zDiff += firstElemOffset;
		fDiff += firstElemOffset;
		iDiff += firstElemOffset;
	}

	for( int step = 0; step < seqLength - 1; ++step ) {
		for( int index = 0; index < objSize; ++index ) {
			zDiff[index] = outDiff[index] * i[index];
			fDiff[index] = outDiff[index] * out[objOffset + index];
			iDiff[index] = outDiff[index] * z[index];
			outDiff[objOffset + index] += outDiff[index] * f[index];
		}
		z += objOffset;
		f += objOffset;
		i += objOffset;
		out += objOffset;
		outDiff += objOffset;
		zDiff += objOffset;
		fDiff += objOffset;
		iDiff += objOffset;
	}

	if( h0 == nullptr ) {
		for( int index = 0; index < objSize; ++index ) {
			zDiff[index] = outDiff[index] * i[index];
			fDiff[index] = 0.f;
			iDiff[index] = outDiff[index] * z[index];
		}
	} else {
		for( int index = 0; index < objSize; ++index ) {
			zDiff[index] = outDiff[index] * i[index];
			fDiff[index] = outDiff[index] * h0[index];
			iDiff[index] = outDiff[index] * z[index];
		}
	}
}

static void ifPoolingBackwardImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const CInterval batchLengthInterval = params.GetInterval( "BatchLength" );
	const CInterval batchWidthInterval = params.GetInterval( "BatchWidth" );
	const CInterval channelsInterval = params.GetInterval( "Channels" );

	const int batchLength = random.UniformInt( batchLengthInterval.Begin, batchLengthInterval.End );
	const int batchWidth = random.UniformInt( batchWidthInterval.Begin, batchWidthInterval.End );
	const int channels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const bool hasInitialState = random.Next() % 2 == 1;
	const bool reverse = random.Next() % 2 == 1;

	const int stateSize = channels * batchWidth;
	const int dataSize = batchLength * stateSize;

	CREATE_FILL_FLOAT_ARRAY( zData, -2.f, 2.f, dataSize, random );
	CREATE_FILL_FLOAT_ARRAY( fData, -2.f, 2.f, dataSize, random );
	CREATE_FILL_FLOAT_ARRAY( iData, -2.f, 2.f, dataSize, random );
	CREATE_FILL_FLOAT_ARRAY( h0Data, -2.f, 2.f, stateSize, random );
	CREATE_FILL_FLOAT_ARRAY( outDiffData, -2.f, 2.f, dataSize, random );

	CDnn dnn( random, MathEngine() );

	CPtr<CDnnBlob> zBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, batchLength, batchWidth, channels );
	ASSERT_EQ( dataSize, zBlob->GetDataSize() );
	zBlob->CopyFrom( zData.GetPtr() );
	CPtr<CSourceLayer> zSource = new CSourceLayer( MathEngine() );
	zSource->SetName( "zSource" );
	dnn.AddLayer( *zSource );
	zSource->SetBlob( zBlob );
	CPtr<CDummyLearnLayer> zLearn = new CDummyLearnLayer( MathEngine() );
	zLearn->SetName( "zLearn" );
	zLearn->Connect( *zSource );
	dnn.AddLayer( *zLearn );

	CPtr<CDnnBlob> fBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, batchLength, batchWidth, channels );
	ASSERT_EQ( dataSize, fBlob->GetDataSize() );
	fBlob->CopyFrom( fData.GetPtr() );
	CPtr<CSourceLayer> fSource = new CSourceLayer( MathEngine() );
	fSource->SetName( "fSource" );
	dnn.AddLayer( *fSource );
	fSource->SetBlob( fBlob );
	CPtr<CDummyLearnLayer> fLearn = new CDummyLearnLayer( MathEngine() );
	fLearn->SetName( "fLearn" );
	fLearn->Connect( *fSource );
	dnn.AddLayer( *fLearn );

	CPtr<CDnnBlob> iBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, batchLength, batchWidth, channels );
	ASSERT_EQ( dataSize, iBlob->GetDataSize() );
	iBlob->CopyFrom( iData.GetPtr() );
	CPtr<CSourceLayer> iSource = new CSourceLayer( MathEngine() );
	iSource->SetName( "iSource" );
	dnn.AddLayer( *iSource );
	iSource->SetBlob( iBlob );
	CPtr<CDummyLearnLayer> iLearn = new CDummyLearnLayer( MathEngine() );
	iLearn->SetName( "iLearn" );
	iLearn->Connect( *iSource );
	dnn.AddLayer( *iLearn );

	CPtr<CIfPoolingTestLayer> ifPooling = new CIfPoolingTestLayer( MathEngine(), channels );
	ifPooling->SetReverseSequence( reverse );
	ifPooling->Connect( 0, *zLearn );
	ifPooling->Connect( 1, *fLearn );
	ifPooling->Connect( 2, *iLearn );
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

	CArray<float> outData;
	outData.SetSize( dataSize );
	naiveIfPooling( reverse, batchLength, stateSize, zData.GetPtr(), fData.GetPtr(), iData.GetPtr(),
		hasInitialState ? h0Data.GetPtr() : nullptr,
		outData.GetPtr() );
	CPtr<CDnnBlob> outBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, batchLength, batchWidth, channels );
	ASSERT_EQ( dataSize, outBlob->GetDataSize() );
	outBlob->CopyFrom( outData.GetPtr() );
	CPtr<CSourceLayer> outSource = new CSourceLayer( MathEngine() );
	outSource->SetName( "outSource" );
	dnn.AddLayer( *outSource );
	outSource->SetBlob( outBlob );

	CPtr<CDnnBlob> outDiffBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, batchLength, batchWidth, channels );
	ASSERT_EQ( dataSize, outDiffBlob->GetDataSize() );
	outDiffBlob->CopyFrom( outDiffData.GetPtr() );
	CPtr<CDummyLossLayer> loss = new CDummyLossLayer( MathEngine() );
	loss->Connect( *ifPooling );
	loss->Connect( 1, *outSource );
	dnn.AddLayer( *loss );
	loss->Diff = outDiffBlob;

	CArray<float> zDiffData;
	zDiffData.SetSize( dataSize );
	CArray<float> fDiffData;
	fDiffData.SetSize( dataSize );
	CArray<float> iDiffData;
	iDiffData.SetSize( dataSize );
	naiveIfPoolingBackward( !reverse, batchLength, stateSize, zData.GetPtr(), fData.GetPtr(), iData.GetPtr(),
		hasInitialState ? h0Data.GetPtr() : nullptr,
		outData.GetPtr(), outDiffData.GetPtr(), zDiffData.GetPtr(), fDiffData.GetPtr(), iDiffData.GetPtr() );

	CPtr<CDnnBlob> zDiffBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, batchLength, batchWidth, channels );
	ASSERT_EQ( dataSize, zDiffBlob->GetDataSize() );
	zDiffBlob->CopyFrom( zDiffData.GetPtr() );
	zLearn->ExpectedDiff = zDiffBlob;

	CPtr<CDnnBlob> fDiffBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, batchLength, batchWidth, channels );
	ASSERT_EQ( dataSize, fDiffBlob->GetDataSize() );
	fDiffBlob->CopyFrom( fDiffData.GetPtr() );
	fLearn->ExpectedDiff = fDiffBlob;

	CPtr<CDnnBlob> iDiffBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, batchLength, batchWidth, channels );
	ASSERT_EQ( dataSize, iDiffBlob->GetDataSize() );
	iDiffBlob->CopyFrom( iDiffData.GetPtr() );
	iLearn->ExpectedDiff = iDiffBlob;

	dnn.RunAndLearnOnce();
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
			"TestCount = 500;"
		)
	)
);

//------------------------------------------------------------------------------------------------------------

TEST_P( CQrnnTest, FPoolingInferenceRandom )
{
	RUN_TEST_IMPL( fPoolingImpl );
}

TEST_P( CQrnnTest, IfPoolingInferenceRandom )
{
	RUN_TEST_IMPL( ifPoolingImpl );
}

TEST_P( CQrnnTest, FPoolingBackwardRandom )
{
	RUN_TEST_IMPL( fPoolingBackwardImpl );
}

TEST_P( CQrnnTest, IfPoolingBackwardRandom )
{
	RUN_TEST_IMPL( ifPoolingBackwardImpl );
}
