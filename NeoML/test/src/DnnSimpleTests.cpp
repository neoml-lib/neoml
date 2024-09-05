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

class CDnnSimpleTestConcatParams {
public:
	// Concatenation type for tests
	enum TConcat { CAT_Batch, CAT_Height, CAT_Width, CAT_Depth, CAT_Channel, CAT_Object };

	TConcat Type;

	explicit CDnnSimpleTestConcatParams( TConcat type ) : Type( type ) {}
};

//---------------------------------------------------------------------------------------------------------------------

class CDnnSimpleTest : public CNeoMLTestFixture, public ::testing::WithParamInterface<CDnnSimpleTestConcatParams::TConcat> {
public:
	static bool InitTestFixture();
	static void DeinitTestFixture();
};

static void rebuildDnn( CDnn& dnn )
{
	CArray<const char*> layersNames;
	dnn.GetLayerList( layersNames );

	CArray<CString> sourceNames;
	CObjectArray<CDnnBlob> sourceData;

	for( const char* name : layersNames ) {
		CSourceLayer* layer = dynamic_cast<CSourceLayer*>( dnn.GetLayer( name ).Ptr() );
		if( layer != nullptr ) {
			sourceNames.Add( name );
			sourceData.Add( layer->GetBlob() );
		}
	}
	{
		CMemoryFile file;
		CArchive archive( &file, CArchive::store );
		archive << dnn;
		archive.Close();
		file.SeekToBegin();

		archive.Open( &file, CArchive::load );
		archive >> dnn;
		archive.Close();
	}
	for( int i = 0; i < sourceNames.Size(); ++i ) {
		CheckCast<CSourceLayer>( dnn.GetLayer( sourceNames[i] ).Ptr() )->SetBlob( sourceData[i] );
	}
}

//---------------------------------------------------------------------------------------------------------------------

REGISTER_NEOML_LAYER( CDnnSimpleTestDummyLearningLayer, "NeoMLDnnSimpleTestDummyLearningLayer" )

void CDnnSimpleTestDummyLearningLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( 0 );
	CBaseLayer::Serialize( archive );
	SerializeBlob( MathEngine(), archive, ExpectedDiff );
}

void CDnnSimpleTestDummyLearningLayer::Reshape()
{
	NeoAssert( GetInputCount() == 1 );
	outputDescs[0] = inputDescs[0];
}

void CDnnSimpleTestDummyLearningLayer::BackwardOnce()
{
	MathEngine().VectorCopy( inputDiffBlobs[0]->GetData(),
		outputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetDataSize() );
}

void CDnnSimpleTestDummyLearningLayer::LearnOnce()
{
	if( ActualDiff != nullptr && ActualDiff->HasEqualDimensions( outputDiffBlobs[0] ) ) {
		ActualDiff->CopyFrom( outputDiffBlobs[0] );
	} else {
		ActualDiff = outputDiffBlobs[0]->GetCopy();
	}

	if( ExpectedDiff.Ptr() == nullptr ) {
		return;
	}

	EXPECT_EQ( ExpectedDiff->GetObjectCount(), outputDiffBlobs[0]->GetObjectCount() );
	EXPECT_EQ( ExpectedDiff->GetHeight(), outputDiffBlobs[0]->GetHeight() );
	EXPECT_EQ( ExpectedDiff->GetWidth(), outputDiffBlobs[0]->GetWidth() );
	EXPECT_EQ( ExpectedDiff->GetDepth(), outputDiffBlobs[0]->GetDepth() );
	EXPECT_EQ( ExpectedDiff->GetChannelsCount(), outputDiffBlobs[0]->GetChannelsCount() );

	CArray<float> expectedDiffBuf;
	expectedDiffBuf.SetSize( ExpectedDiff->GetDataSize() );
	ExpectedDiff->CopyTo( expectedDiffBuf.GetPtr(), expectedDiffBuf.Size() );

	CArray<float> outputDiffBuf;
	outputDiffBuf.SetSize( outputDiffBlobs[0]->GetDataSize() );
	outputDiffBlobs[0]->CopyTo( outputDiffBuf.GetPtr(), outputDiffBuf.Size() );

	const int size = ExpectedDiff->GetDataSize();
	for( int i = 0; i < size; ++i ) {
		EXPECT_NEAR( expectedDiffBuf[i], outputDiffBuf[i], 1e-3f ) << " i = " << i << " size = " << size;
	}
}

//---------------------------------------------------------------------------------------------------------------------

REGISTER_NEOML_LAYER( CDnnSimpleTestDummyLossLayer, "NeoMLDnnSimpleTestDummyLossLayer" )

void CDnnSimpleTestDummyLossLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( 0 );
	CLossLayer::Serialize( archive );
	SerializeBlob( MathEngine(), archive, Diff );
}

void CDnnSimpleTestDummyLossLayer::BatchCalculateLossAndGradient( int batchSize, CConstFloatHandle, int vectorSize,
	CConstIntHandle, int, CFloatHandle lossValue, CFloatHandle lossGradient )
{
	const int totalSize = batchSize * vectorSize;
	MathEngine().VectorFill( lossValue, 0, batchSize );

	if( Diff.Ptr() == nullptr ) {
		MathEngine().VectorFill( lossGradient, 0, totalSize );
	} else {
		// anti-multiplier is to cover the effect of the multiplier in the base loss-layer
		CFloatHandleStackVar antiMult( MathEngine(), batchSize );
		MathEngine().VectorFill( antiMult, batchSize / GetLossWeight(), batchSize );
		MathEngine().VectorEltwiseDivide( antiMult, GetWeights()->GetData(), antiMult, batchSize );

		EXPECT_EQ( Diff->GetDataSize(), totalSize );
		MathEngine().MultiplyDiagMatrixByMatrix( antiMult, batchSize,
			Diff->GetData(), vectorSize, lossGradient, totalSize );
	}
}

void CDnnSimpleTestDummyLossLayer::BatchCalculateLossAndGradient( int batchSize, CConstFloatHandle data, int vectorSize,
	CConstFloatHandle label, int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient,
	CFloatHandle labelLossGradient )
{
	EXPECT_TRUE( inputBlobs.Size() <= 1 || inputBlobs[0]->HasEqualDimensions( inputBlobs[1] ) );
	EXPECT_TRUE( labelSize == vectorSize );

	const int totalSize = batchSize * vectorSize;

	CArray<float> dataBuf;
	dataBuf.SetSize( totalSize );
	MathEngine().DataExchangeTyped( dataBuf.GetPtr(), data, dataBuf.Size() );

	CArray<float> labelBuf;
	labelBuf.SetSize( totalSize );
	MathEngine().DataExchangeTyped( labelBuf.GetPtr(), label, labelBuf.Size() );

	for( int i = 0; i < totalSize; ++i ) {
		EXPECT_NEAR( dataBuf[i], labelBuf[i], 1e-2f ) << " i = " << i << " size = " << totalSize;
	}

	MathEngine().VectorFill( lossValue, 0, batchSize );

	if( lossGradient.IsNull() ) {
		return;
	}

	if( Diff.Ptr() == 0 ) {
		MathEngine().VectorFill( lossGradient, 0, totalSize );
	} else {
		// anti-multiplier is to cover the effect of the multiplier in the base loss-layer
		CFloatHandleStackVar antiMult( MathEngine(), batchSize );
		MathEngine().VectorFill( antiMult, batchSize / GetLossWeight(), batchSize );
		MathEngine().VectorEltwiseDivide( antiMult, GetWeights()->GetData(), antiMult, batchSize );

		EXPECT_EQ( Diff->GetDataSize(), totalSize );
		MathEngine().MultiplyDiagMatrixByMatrix( antiMult, batchSize,
			Diff->GetData(), vectorSize, lossGradient, totalSize );
	}

	if( labelLossGradient.IsNull() ) {
		return;
	}

	MathEngine().VectorCopy( labelLossGradient, lossGradient, totalSize );
}

//---------------------------------------------------------------------------------------------------------------------

static constexpr float mergeResults[6][32]{
	{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32 },
	{ 1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 20, 21, 22, 23, 24, 9, 10, 11, 12, 13, 14, 15, 16, 25, 26, 27, 28, 29, 30, 31, 32 },
	{ 1, 2, 3, 4, 17, 18, 19, 20, 5, 6, 7, 8, 21, 22, 23, 24, 9, 10, 11, 12, 25, 26, 27, 28, 13, 14, 15, 16, 29, 30, 31, 32 },
	{ 1, 2, 17, 18, 3, 4, 19, 20, 5, 6, 21, 22, 7, 8, 23, 24, 9, 10, 25, 26, 11, 12, 27, 28, 13, 14, 29, 30, 15, 16, 31, 32 },
	{ 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 16, 32 },
	{ 1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 20, 21, 22, 23, 24, 9, 10, 11, 12, 13, 14, 15, 16, 25, 26, 27, 28, 29, 30, 31, 32 }
};

static void concatTest( IMathEngine& mathEngine, const CDnnSimpleTestConcatParams& params )
{
	CPtr<CDnnBlob> inputs[]{
		CDnnBlob::Create3DImageBlob( mathEngine, CT_Float, 1, 2, 2, 2, 2, 1 ),
		CDnnBlob::Create3DImageBlob( mathEngine, CT_Float, 1, 2, 2, 2, 2, 1 )
	};
	CPtr<CDnnBlob> inputDiffs[]{
		CDnnBlob::Create3DImageBlob( mathEngine, CT_Float, 1, 2, 2, 2, 2, 1 ),
		CDnnBlob::Create3DImageBlob( mathEngine, CT_Float, 1, 2, 2, 2, 2, 1 )
	};

	float value = 0;
	for( int i = 0; i < 2; ++i ) {
		for( int j = 0; j < inputs[i]->GetDataSize(); ++j ) {
			value = value + 1;
			inputs[i]->GetData().SetValueAt( j, value );
			inputDiffs[i]->GetData().SetValueAt( j, -value );
		}
	}

	CPtr<CBaseLayer> concatLayer;
	CPtr<CDnnBlob> resultBlob;

	CRandom random;
	CDnn dnn( random, MathEngine() );

	switch( params.Type ) {
		case CDnnSimpleTestConcatParams::CAT_Batch:
			concatLayer = new CConcatBatchWidthLayer( MathEngine() );
			resultBlob = CDnnBlob::Create3DImageBlob( MathEngine(), CT_Float, 1, 4, 2, 2, 2, 1 );
			break;
		case CDnnSimpleTestConcatParams::CAT_Height:
			concatLayer = new CConcatHeightLayer( MathEngine() );
			resultBlob = CDnnBlob::Create3DImageBlob( MathEngine(), CT_Float, 1, 2, 4, 2, 2, 1 );
			break;
		case CDnnSimpleTestConcatParams::CAT_Width:
			concatLayer = new CConcatWidthLayer( MathEngine() );
			resultBlob = CDnnBlob::Create3DImageBlob( MathEngine(), CT_Float, 1, 2, 2, 4, 2, 1 );
			break;
		case CDnnSimpleTestConcatParams::CAT_Depth:
			concatLayer = new CConcatDepthLayer( MathEngine() );
			resultBlob = CDnnBlob::Create3DImageBlob( MathEngine(), CT_Float, 1, 2, 2, 2, 4, 1 );
			break;
		case CDnnSimpleTestConcatParams::CAT_Channel:
			concatLayer = new CConcatChannelsLayer( MathEngine() );
			resultBlob = CDnnBlob::Create3DImageBlob( MathEngine(), CT_Float, 1, 2, 2, 2, 2, 2 );
			break;
		case CDnnSimpleTestConcatParams::CAT_Object:
			concatLayer = new CConcatObjectLayer( MathEngine() );
			resultBlob = CDnnBlob::Create3DImageBlob( MathEngine(), CT_Float, 1, 2, 1, 1, 1, 16 );
			break;
		default:
			FAIL();
	}

	resultBlob->CopyFrom( mergeResults[params.Type] );

	for( int i = 0; i < 2; ++i ) {
		CPtr<CSourceLayer> source = new CSourceLayer( MathEngine() );
		CString sourceName = source->GetName() + CString( "." ) + Str( i );
		source->SetName( sourceName );
		source->SetBlob( inputs[i] );
		dnn.AddLayer( *source );

		CPtr<CDnnSimpleTestDummyLearningLayer> dummy = new CDnnSimpleTestDummyLearningLayer( MathEngine() );
		CString dummyName = dummy->GetName() + CString( "." ) + Str( i );
		dummy->SetName( dummyName );
		dummy->Connect( *source );
		dummy->ExpectedDiff = inputDiffs[i];
		dnn.AddLayer( *dummy );

		concatLayer->Connect( i, *dummy, 0 );
	}

	dnn.AddLayer( *concatLayer );

	CPtr<CSourceLayer> label = new CSourceLayer( MathEngine() );
	CString labelName = label->GetName() + CString( "." ) + "label";
	label->SetName( labelName );
	label->SetBlob( resultBlob );
	dnn.AddLayer( *label );

	CPtr<CDnnSimpleTestDummyLossLayer> loss = new CDnnSimpleTestDummyLossLayer( MathEngine() );
	CPtr<CDnnBlob> diffBlob = resultBlob->GetClone();
	for( int i = 0; i < diffBlob->GetDataSize(); ++i ) {
		diffBlob->GetData().SetValueAt( i, -resultBlob->GetData().GetValueAt( i ) );
	}
	loss->Diff = diffBlob;
	loss->Connect( 0, *concatLayer, 0 );
	loss->Connect( 1, *label, 0 );
	dnn.AddLayer( *loss );

	dnn.RunAndBackwardOnce();

	rebuildDnn( dnn );
	dnn.RunAndBackwardOnce();
}

static void splitTest( IMathEngine& mathEngine, const CDnnSimpleTestConcatParams& params )
{
	if( params.Type == CDnnSimpleTestConcatParams::CAT_Object ) {
		return; // no such a split
	}

	CPtr<CDnnBlob> results[]{
		CDnnBlob::Create3DImageBlob( mathEngine, CT_Float, 1, 2, 2, 2, 2, 1 ),
		CDnnBlob::Create3DImageBlob( mathEngine, CT_Float, 1, 2, 2, 2, 2, 1 )
	};
	CPtr<CDnnBlob> lossDiffs[]{
		CDnnBlob::Create3DImageBlob( mathEngine, CT_Float, 1, 2, 2, 2, 2, 1 ),
		CDnnBlob::Create3DImageBlob( mathEngine, CT_Float, 1, 2, 2, 2, 2, 1 )
	};

	float value = 0;
	for( int i = 0; i < 2; ++i ) {
		for( int j = 0; j < results[i]->GetDataSize(); ++j ) {
			value = value + 1;
			results[i]->GetData().SetValueAt( j, value );
			lossDiffs[i]->GetData().SetValueAt( j, -value );
		}
	}

	CPtr<CBaseSplitLayer> splitLayer;
	CPtr<CDnnBlob> inputBlob;

	CRandom random;
	CDnn dnn( random, MathEngine() );

	switch( params.Type ) {
		case CDnnSimpleTestConcatParams::CAT_Batch:
			splitLayer = new CSplitBatchWidthLayer( MathEngine() );
			splitLayer->SetOutputCounts2( 2 );
			inputBlob = CDnnBlob::Create3DImageBlob( MathEngine(), CT_Float, 1, 4, 2, 2, 2, 1 );
			break;
		case CDnnSimpleTestConcatParams::CAT_Height:
			splitLayer = new CSplitHeightLayer( MathEngine() );
			splitLayer->SetOutputCounts2( 2 );
			inputBlob = CDnnBlob::Create3DImageBlob( MathEngine(), CT_Float, 1, 2, 4, 2, 2, 1 );
			break;
		case CDnnSimpleTestConcatParams::CAT_Width:
			splitLayer = new CSplitWidthLayer( MathEngine() );
			splitLayer->SetOutputCounts2( 2 );
			inputBlob = CDnnBlob::Create3DImageBlob( MathEngine(), CT_Float, 1, 2, 2, 4, 2, 1 );
			break;
		case CDnnSimpleTestConcatParams::CAT_Depth:
			splitLayer = new CSplitDepthLayer( MathEngine() );
			splitLayer->SetOutputCounts2( 2 );
			inputBlob = CDnnBlob::Create3DImageBlob( MathEngine(), CT_Float, 1, 2, 2, 2, 4, 1 );
			break;
		case CDnnSimpleTestConcatParams::CAT_Channel:
			splitLayer = new CSplitChannelsLayer( MathEngine() );
			splitLayer->SetOutputCounts2( 1 );
			inputBlob = CDnnBlob::Create3DImageBlob( MathEngine(), CT_Float, 1, 2, 2, 2, 2, 2 );
			break;
		default:
			FAIL();
	}

	inputBlob->CopyFrom( mergeResults[params.Type] );

	CPtr<CDnnBlob> inputDiffBlob = inputBlob->GetClone();
	for( int i = 0; i < inputDiffBlob->GetDataSize(); ++i ) {
		inputDiffBlob->GetData().SetValueAt( i, -inputBlob->GetData().GetValueAt( i ) );
	}

	CPtr<CSourceLayer> source = new CSourceLayer( MathEngine() );
	source->SetBlob( inputBlob );
	dnn.AddLayer( *source );

	CPtr<CDnnSimpleTestDummyLearningLayer> dummy = new CDnnSimpleTestDummyLearningLayer( MathEngine() );
	dummy->Connect( *source );
	dummy->ExpectedDiff = inputDiffBlob;
	dnn.AddLayer( *dummy );

	splitLayer->Connect( *dummy );

	dnn.AddLayer( *splitLayer );

	for( int i = 0; i < 2; ++i ) {
		CPtr<CSourceLayer> label = new CSourceLayer( MathEngine() );
		CString labelName = label->GetName() + CString( ".label." ) + Str( i );
		label->SetName( labelName );
		label->SetBlob( results[i] );
		dnn.AddLayer( *label );

		CPtr<CDnnSimpleTestDummyLossLayer> loss = new CDnnSimpleTestDummyLossLayer( MathEngine() );
		CString lossName = loss->GetName() + CString( "." ) + Str( i );
		loss->SetName( lossName );
		loss->Diff = lossDiffs[i];
		loss->Connect( 0, *splitLayer, i );
		loss->Connect( 1, *label, 0 );
		dnn.AddLayer( *loss );
	}

	dnn.RunAndBackwardOnce();

	rebuildDnn( dnn );
	dnn.RunAndBackwardOnce();
}

//---------------------------------------------------------------------------------------------------------------------

class CDnnEltwiseLayerChecker {
public:
	CDnnEltwiseLayerChecker() :
		Output( CDnnBlob::Create3DImageBlob( MathEngine(), CT_Float, 1, 2, 3, 4, 5, 6 ) ),
		Diff( Output->GetClone() )
	{}
	virtual ~CDnnEltwiseLayerChecker() = default;
	void Check( int inputCount );

protected:
	CPtr<CEltwiseBaseLayer> CheckLayer;
	CObjectArray<CDnnBlob> Inputs;
	CPtr<CDnnBlob> Output;
	CPtr<CDnnBlob> Diff;
	CObjectArray<CDnnBlob> ExpectedDiffs;

	virtual void generateExpectedDiff() = 0;
};

void CDnnEltwiseLayerChecker::Check( int inputCount )
{
	Inputs.SetSize( inputCount );

	generateExpectedDiff();

	CRandom random;
	CDnn dnn( random, MathEngine() );

	CPtr<CSourceLayer> labelLayer = new CSourceLayer( MathEngine() );
	labelLayer->SetName( labelLayer->GetName() + CString( ".Label" ) );
	labelLayer->SetBlob( Output );
	dnn.AddLayer( *labelLayer );

	for( int i = 0; i < inputCount; ++i ) {
		CPtr<CSourceLayer> source = new CSourceLayer( MathEngine() );
		source->SetName( source->GetName() + CString( "." ) + Str( i ) );
		source->SetBlob( Inputs[i] );
		dnn.AddLayer( *source );

		CPtr<CDnnSimpleTestDummyLearningLayer> dummy = new CDnnSimpleTestDummyLearningLayer( MathEngine() );
		dummy->SetName( dummy->GetName() + CString( "." ) + Str( i ) );
		dummy->ExpectedDiff = ExpectedDiffs.Size() ? ExpectedDiffs[i] : Diff;
		dummy->Connect( *source );
		dnn.AddLayer( *dummy );

		CheckLayer->Connect( i, *dummy, 0 );
	}

	dnn.AddLayer( *CheckLayer );

	CPtr<CDnnSimpleTestDummyLossLayer> lossLayer = new CDnnSimpleTestDummyLossLayer( MathEngine() );
	lossLayer->Diff = Diff;
	lossLayer->Connect( 0, *CheckLayer, 0 );
	lossLayer->Connect( 1, *labelLayer, 0 );
	dnn.AddLayer( *lossLayer );

	dnn.RunOnce();
	dnn.RunAndBackwardOnce();

	rebuildDnn( dnn );
	dnn.RunOnce();
	dnn.RunAndBackwardOnce();
}

//---------------------------------------------------------------------------------------------------------------------

class CDnnEltwiseSumLayerChecker : public CDnnEltwiseLayerChecker {
	void generateExpectedDiff() override;
};

void CDnnEltwiseSumLayerChecker::generateExpectedDiff()
{
	CheckLayer = new CEltwiseSumLayer( MathEngine() );

	const int inputCount = Inputs.Size();
	const int inputSize = Output->GetDataSize();

	float value = 0;
	for( int i = 0; i < inputCount; ++i ) {
		Inputs[i] = Output->GetClone();
		for( int j = 0; j < inputSize; ++j ) {
			Inputs[i]->GetData().SetValueAt( j, ++value );
		}
	}

	for( int j = 0; j < inputSize; ++j ) {
		Diff->GetData().SetValueAt( j, static_cast<float>( -j ) );
		float sum = 0;
		for( int i = 0; i < inputCount; ++i ) {
			sum += Inputs[i]->GetData().GetValueAt( j );
		}
		Output->GetData().SetValueAt( j, sum );
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CDnnEltwiseMulLayerChecker : public CDnnEltwiseLayerChecker {
public:
	enum Type { Mul, NegMul, Div };
	explicit CDnnEltwiseMulLayerChecker( Type type ) : type( type ) {}

private:
	const Type type = Mul;
	void generateExpectedDiff() override;
};

void CDnnEltwiseMulLayerChecker::generateExpectedDiff()
{
	switch( type ) {
		case Mul: CheckLayer = new CEltwiseMulLayer( MathEngine() ); break;
		case NegMul: CheckLayer = new CEltwiseNegMulLayer( MathEngine() ); break;
		case Div: CheckLayer = new CEltwiseDivLayer( MathEngine() ); break;
		default: EXPECT_TRUE( false );
	}

	const int inputCount = Inputs.Size();
	const int inputSize = Output->GetDataSize();
	ExpectedDiffs.SetSize( inputCount );

	float value = 0;
	for( int i = 0; i < inputCount; ++i ) {
		Inputs[i] = Output->GetClone();
		ExpectedDiffs[i] = Output->GetClone();
		for( int j = 0; j < inputSize; ++j ) {
			Inputs[i]->GetData().SetValueAt( j, ++value );
		}
	}

	for( int j = 0; j < inputSize; ++j ) {
		Diff->GetData().SetValueAt( j, static_cast<float>( type == Div ? 1 : -j ) );
		const float diff = Diff->GetData().GetValueAt( j );

		float mul = 1;
		for( int i = 0; i < inputCount; ++i ) {
			const float input = Inputs[i]->GetData().GetValueAt( j );
			float expectedDiff;

			if( type == Div ) {
				mul = ( i == 0 ) ? input : ( mul / input ); // z = x / y
				expectedDiff = diff;
			} else if( type == NegMul && i == 0 ) {
				mul *= 1 - input;
				expectedDiff = -diff;
			} else {
				mul *= input;
				expectedDiff = diff;
			}

			for( int ii = 0; ii < inputCount; ++ii ) {
				const float input = Inputs[ii]->GetData().GetValueAt( j );
				if( i != ii ) {
					if( type == Div ) {
						// dz = dx * y
						// dz = dy * -x / y^2
						expectedDiff = ( i == 0 ) ? ( 1.f / input ) : ( - mul * mul / input );
					} else if( type == NegMul && ii == 0 ) {
						expectedDiff *= 1 - input;
					} else {
						expectedDiff *= input;
					}
				}
			}

			ExpectedDiffs[i]->GetData().SetValueAt( j, expectedDiff );
		}
		Output->GetData().SetValueAt( j, mul );
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CDnnEltwiseMaxLayerChecker : public CDnnEltwiseLayerChecker {
	void generateExpectedDiff() override;
};

void CDnnEltwiseMaxLayerChecker::generateExpectedDiff()
{
	CheckLayer = new CEltwiseMaxLayer( MathEngine() );

	const int inputCount = Inputs.Size();
	ExpectedDiffs.SetSize( inputCount );

	for( int i = 0; i < inputCount; ++i ) {
		Inputs[i] = Output->GetClone();
		ExpectedDiffs[i] = Output->GetClone();
		ExpectedDiffs[i]->Clear();
	}

	const int inputSize = Output->GetDataSize();
	float value = 0;
	for( int i = 0; i < inputSize; i += inputCount ) {
		//  i -->          i -->
		// k  x o o o o   x o o
		//    o x o o o	  o x o
		//    o o x o o	  o o x
		//    o o o x o	  o o o
		//    o o o o x	  o o o
		//    j -->
		for( int j = 0; j < inputCount && ( i + j ) < inputSize; ++j ) {
			for( int k = 0; k < inputCount; ++k ) {
				if( j != k ) {
					Inputs[k]->GetData().SetValueAt( i + j, ( ++value ) );
				}
			}
			Inputs[j]->GetData().SetValueAt( i + j, ( ++value ) );
		}
	}

	for( int j = 0; j < inputSize; ++j ) {
		Output->GetData().SetValueAt( j, static_cast<float>( ( j + 1 ) * inputCount ) );
		Diff->GetData().SetValueAt( j, -Output->GetData().GetValueAt( j ) );
		ExpectedDiffs[j % inputCount]->GetData().SetValueAt( j, Diff->GetData().GetValueAt( j ) );
	}
}

//---------------------------------------------------------------------------------------------------------------------

using TSoftmaxTestCreateBlobFunc = CPtr<CDnnBlob>( * )( const float* );

static void performSoftmaxTest( TSoftmaxTestCreateBlobFunc createBlob,
	CSoftmaxLayer::TNormalizationArea area )
{
	CRandom random;
	CDnn dnn( random, MathEngine() );

	// network
	CPtr<CSourceLayer> source = new CSourceLayer( MathEngine() );
	dnn.AddLayer( *source );

	CPtr<CDnnSimpleTestDummyLearningLayer> dummy = new CDnnSimpleTestDummyLearningLayer( MathEngine() );
	dummy->Connect( *source );
	dnn.AddLayer( *dummy );

	CPtr<CSoftmaxLayer> softmax = new CSoftmaxLayer( MathEngine() );
	softmax->Connect( *dummy );
	softmax->SetNormalizationArea( area );
	dnn.AddLayer( *softmax );

	CPtr<CSourceLayer> label = new CSourceLayer( MathEngine() );
	label->SetName( "Label" );
	dnn.AddLayer( *label );

	CPtr<CDnnSimpleTestDummyLossLayer> loss = new CDnnSimpleTestDummyLossLayer( MathEngine() );
	loss->Connect( 0, *softmax, 0 );
	loss->Connect( 1, *label, 0 );
	dnn.AddLayer( *loss );

	// data
	const float sourceData[] = { 0, logf( 2 ), logf( 4 ), logf( 8 ), logf( 16 ), logf( 32 ) };
	const float labelData[] = { 1.f / 7, 2.f / 7, 4.f / 7, 1.f / 7, 2.f / 7, 4.f / 7 };
	const float diffData[] = { 1, 1, 1, 2, 2, 2 };
	const float expectedDiffData[] = { 0, 0, 0, 0, 0, 0 };

	source->SetBlob( createBlob( sourceData ) );
	label->SetBlob( createBlob( labelData ) );
	loss->Diff = createBlob( diffData );
	dummy->ExpectedDiff = createBlob( expectedDiffData );

	dnn.RunAndBackwardOnce();
}

//---------------------------------------------------------------------------------------------------------------------

// Initialize blob by values in (-1; 1)
static void initBlob( CPtr<CDnnBlob> blob, CRandom& random )
{
	CArray<float> data;
	data.SetSize( blob->GetDataSize() );
	for( int i = 0; i < data.Size(); ++i ) {
		data[i] = random.UniformInt( -99, 99 ) / 100.f;
	}
	blob->CopyFrom( data.GetPtr() );
}

// Check for blob is nullifed
static void checkBlobFilledWithZero( CPtr<CDnnBlob> blob )
{
	CArray<float> buffer;
	buffer.SetSize( blob->GetDataSize() );
	blob->CopyTo( buffer.GetPtr(), blob->GetDataSize() );

	EXPECT_EQ( blob->GetDataSize(), buffer.Size() );
	for( int i = 0; i < buffer.Size(); ++i ) {
		EXPECT_FLOAT_EQ( 0.f, buffer[i] );
	}
}

} // namespace NeoMLTest

//---------------------------------------------------------------------------------------------------------------------

TEST_P( CDnnSimpleTest, ConcatTest )
{
	const CDnnSimpleTestConcatParams params( GetParam() );

	concatTest( MathEngine(), params );
	splitTest( MathEngine(), params );
}

INSTANTIATE_TEST_CASE_P( CDnnConcatTestInstantiation, CDnnSimpleTest,
	::testing::Values(
		CDnnSimpleTestConcatParams::CAT_Batch,
		CDnnSimpleTestConcatParams::CAT_Height,
		CDnnSimpleTestConcatParams::CAT_Width,
		CDnnSimpleTestConcatParams::CAT_Depth,
		CDnnSimpleTestConcatParams::CAT_Channel,
		CDnnSimpleTestConcatParams::CAT_Object
	)
);

//---------------------------------------------------------------------------------------------------------------------

TEST_F( CDnnSimpleTest, EltwiseSumTest )
{
	CDnnEltwiseSumLayerChecker checker;
	checker.Check( /*inputCount*/2 );
	checker.Check( /*inputCount*/3 );
	checker.Check( /*inputCount*/5 );
}

TEST_F( CDnnSimpleTest, EltwiseMulTest )
{
	CDnnEltwiseMulLayerChecker checker{ CDnnEltwiseMulLayerChecker::Mul };
	checker.Check( /*inputCount*/2 );
	checker.Check( /*inputCount*/3 );
	checker.Check( /*inputCount*/5 );
}

TEST_F( CDnnSimpleTest, EltwiseNegMulTest )
{
	CDnnEltwiseMulLayerChecker checker{ CDnnEltwiseMulLayerChecker::NegMul };
	checker.Check( /*inputCount*/2 );
	checker.Check( /*inputCount*/3 );
	checker.Check( /*inputCount*/5 );
}

TEST_F( CDnnSimpleTest, EltwiseDivTest )
{
	CDnnEltwiseMulLayerChecker{ CDnnEltwiseMulLayerChecker::Div }.Check( /*inputCount*/2 );
}

TEST_F( CDnnSimpleTest, EltwiseMaxTest )
{
	const auto met = MathEngine().GetType();
	if( met != MET_Cpu && met != MET_Cuda ) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		// VectorSpreadValues
		return;
	}

	CDnnEltwiseMaxLayerChecker checker;
	checker.Check( /*inputCount*/2 );
	checker.Check( /*inputCount*/3 );
	checker.Check( /*inputCount*/5 );
}

//---------------------------------------------------------------------------------------------------------------------

TEST_F( CDnnSimpleTest, EnumBinarizationTest )
{
	const int EnumSize = 5;

	CFastArray<float, EnumSize> inputData;
	inputData.SetSize( EnumSize );
	for( int i = 0; i < EnumSize; ++i ) {
		inputData[i] = static_cast<float>( i );
	}

	CPtr<CDnnBlob> inputBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, EnumSize, 1 );
	inputBlob->CopyFrom( inputData.GetPtr() );

	CRandom random;
	CDnn dnn( random, MathEngine() );

	CPtr<CSourceLayer> source = new CSourceLayer( MathEngine() );
	source->SetBlob( inputBlob );
	dnn.AddLayer( *source );

	CPtr<CEnumBinarizationLayer> binarization = new CEnumBinarizationLayer( MathEngine() );
	binarization->SetEnumSize( EnumSize );
	binarization->Connect( *source );
	dnn.AddLayer( *binarization );

	CPtr<CSinkLayer> sink = new CSinkLayer( MathEngine() );
	sink->Connect( *binarization );
	dnn.AddLayer( *sink );

	dnn.RunOnce();

	CPtr<CDnnBlob> resultBlob = sink->GetBlob();
	EXPECT_EQ( resultBlob->GetDataSize(), EnumSize * EnumSize );

	CFastArray<float, EnumSize* EnumSize> resultData;
	resultData.SetSize( EnumSize * EnumSize );
	resultBlob->CopyTo( resultData.GetPtr(), resultData.Size() );

	const float* result = resultData.GetPtr();
	for( int b = 0; b < EnumSize; ++b ) {
		for( int i = 0; i < EnumSize; ++i ) {
			const float value = *result++;
			EXPECT_EQ( value, ( i == b ) ? 1.f : 0.f );
		}
	}
}

TEST_F( CDnnSimpleTest, SoftmaxTest )
{
	const auto met = MathEngine().GetType();
	if( met != MET_Cpu && met != MET_Cuda ) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		// Softmax Backward
		return;
	}

	auto createBlob23 = []( const float* data )
	{
		CPtr<CDnnBlob> blob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 2, 3 );
		blob->CopyFrom( data );
		return blob;
	};
	performSoftmaxTest( createBlob23, CSoftmaxLayer::NA_ObjectSize );

	auto createListBlob23 = []( const float* data )
	{
		CPtr<CDnnBlob> blob = CDnnBlob::CreateListBlob( MathEngine(), CT_Float, 1, 2, 3, 1 );
		blob->CopyFrom( data );
		return blob;
	};
	performSoftmaxTest( createListBlob23, CSoftmaxLayer::NA_ListSize );

	auto createTransposedBlob32 = []( const float* data )
	{
		const float transposed[6]{ data[0], data[3], data[1], data[4], data[2], data[5] };
		CPtr<CDnnBlob> blob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 3, 2, 1 );
		blob->CopyFrom( transposed );
		return blob;
	};
	performSoftmaxTest( createTransposedBlob32, CSoftmaxLayer::NA_BatchLength );
}

TEST_F( CDnnSimpleTest, DropSmallValuesTest )
{
	const auto met = MathEngine().GetType();
	if( met != MET_Cpu && met != MET_Cuda ) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		// FilterLayersParams
		return;
	}

	const int filterHeight = 3;
	const int filterWidth = 4;
	const int filterDepth = 2;
	const int inputChannels = 5;
	const int filterCount = 7;
	const int layerSize = 50;

	for( int attempt = 0; attempt < 2; ++attempt ) {
		CRandom random( 42 );
		CDnn dnn( random, MathEngine() );

		// simple convolution
		CPtr<CConvLayer> conv = new CConvLayer( MathEngine() );
		CPtr<CDnnBlob> filter = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float,
			1, filterCount, filterHeight, filterWidth, inputChannels );
		initBlob( filter, random );
		CPtr<CDnnBlob> freeTerms = CDnnBlob::CreateVector( MathEngine(), CT_Float, filterCount );
		initBlob( freeTerms, random );

		conv->SetFilterCount( filterCount );
		conv->SetFilterHeight( filterHeight );
		conv->SetFilterWidth( filterWidth );
		conv->SetFilterData( filter );
		conv->SetFreeTermData( freeTerms );
		dnn.AddLayer( *conv );

		// rle convolution
		CPtr<CRleConvLayer> rleConv = new CRleConvLayer( MathEngine() );
		initBlob( filter, random );
		initBlob( freeTerms, random );

		rleConv->SetFilterCount( filterCount );
		rleConv->SetFilterHeight( filterHeight );
		rleConv->SetFilterWidth( filterWidth );
		rleConv->SetFilterData( filter );
		rleConv->SetFreeTermData( freeTerms );
		dnn.AddLayer( *rleConv );

		// full connected
		CPtr<CFullyConnectedLayer> full = new CFullyConnectedLayer( MathEngine() );
		CPtr<CDnnBlob> weights = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 1, layerSize );
		initBlob( weights, random );

		full->SetNumberOfElements( layerSize );
		full->SetWeightsData( weights );
		dnn.AddLayer( *full );

		// 3D conv
		CPtr<C3dConvLayer> conv3d = new C3dConvLayer( MathEngine() );
		CPtr<CDnnBlob> filter3d = CDnnBlob::Create3DImageBlob( MathEngine(), CT_Float,
			1, filterCount, filterHeight, filterWidth, filterDepth, inputChannels );
		initBlob( filter3d, random );
		initBlob( freeTerms, random );

		conv3d->SetFilterCount( filterCount );
		conv3d->SetFilterHeight( filterHeight );
		conv3d->SetFilterWidth( filterWidth );
		conv3d->SetFilterDepth( filterDepth );
		conv3d->SetFilterData( filter3d );
		conv3d->SetFreeTermData( freeTerms );
		dnn.AddLayer( *conv3d );

		// 1x1x1 conv
		CPtr<C3dConvLayer> conv1 = new C3dConvLayer( MathEngine() );
		conv1->SetName( "Conv1x1x1" );
		CPtr<CDnnBlob> filter1 = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float,
			1, filterCount, inputChannels );
		initBlob( filter1, random );
		initBlob( freeTerms, random );

		conv1->SetFilterHeight( 1 );
		conv1->SetFilterWidth( 1 );
		conv1->SetFilterDepth( 1 );
		conv1->SetStrideHeight( 1 );
		conv1->SetStrideWidth( 1 );
		conv1->SetStrideDepth( 1 );

		conv1->SetFilterCount( filterCount );
		conv1->SetFilterData( filter1 );
		conv1->SetFreeTermData( freeTerms );
		dnn.AddLayer( *conv1 );

		// time conv
		CPtr<CTimeConvLayer> timeConv = new CTimeConvLayer( MathEngine() );
		CPtr<CDnnBlob> timeFilter = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float,
			1, 1, filterCount, filterHeight, inputChannels );
		initBlob( timeFilter, random );
		initBlob( freeTerms, random );

		timeConv->SetFilterCount( filterCount );
		timeConv->SetFilterSize( filterHeight );
		timeConv->SetFilterData( timeFilter );
		timeConv->SetFreeTermData( freeTerms );
		dnn.AddLayer( *timeConv );

		// composite (with conv inside)
		CPtr<CCompositeLayer> composite = new CCompositeLayer( MathEngine() );
		CPtr<CConvLayer> internalConv = new CConvLayer( MathEngine() );
		initBlob( filter, random );
		initBlob( freeTerms, random );

		internalConv->SetFilterCount( filterCount );
		internalConv->SetFilterHeight( filterHeight );
		internalConv->SetFilterWidth( filterWidth );
		internalConv->SetFilterData( filter );
		internalConv->SetFreeTermData( freeTerms );
		composite->AddLayer( *internalConv );
		dnn.AddLayer( *composite );

		// nullify blobs
		// because the functions is just iterating through the layers,
		// no need in construction links between layers
		if( attempt == 0 ) {
			dnn.FilterLayersParams( 1. );
		} else {
			CArray<const char*> layerNames;
			layerNames.Add( rleConv->GetName() );
			layerNames.Add( timeConv->GetName() );
			layerNames.Add( conv3d->GetName() );
			dnn.FilterLayersParams( layerNames, 1. );
			layerNames.Empty();
			layerNames.Add( conv->GetName() );
			// It is special!
			layerNames.Add( timeConv->GetName() );

			layerNames.Add( full->GetName() );
			layerNames.Add( composite->GetName() );
			layerNames.Add( conv1->GetName() );
			dnn.FilterLayersParams( layerNames, 1. );
		}

		// conv
		checkBlobFilledWithZero( conv->GetFilterData() );
		checkBlobFilledWithZero( conv->GetFreeTermData() );
		// rleConv
		checkBlobFilledWithZero( rleConv->GetFilterData() );
		checkBlobFilledWithZero( rleConv->GetFreeTermData() );
		// fully connected
		checkBlobFilledWithZero( full->GetWeightsData() );
		// conv3d
		checkBlobFilledWithZero( conv3d->GetFilterData() );
		checkBlobFilledWithZero( conv3d->GetFreeTermData() );
		// conv1
		checkBlobFilledWithZero( conv1->GetFilterData() );
		checkBlobFilledWithZero( conv1->GetFreeTermData() );
		// time conv
		checkBlobFilledWithZero( timeConv->GetFilterData() );
		checkBlobFilledWithZero( timeConv->GetFreeTermData() );
		// conv inside composite
		checkBlobFilledWithZero( internalConv->GetFilterData() );
		checkBlobFilledWithZero( internalConv->GetFreeTermData() );
	}
}

