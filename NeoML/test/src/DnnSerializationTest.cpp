// Copyright (c) 1993-2019, ABBYY (BIT Software). All rights reserved.
//	Автор: Федюнин Валерий
//	Система: FineMachineLearningTest
//	Описание: Тест сериализации сверточных сетей.

#include <common.h>
#pragma hdrstop

#include <TestFixture.h>

namespace NeoMLTest {

class CDnnSerializationTest : public CNeoMLTestFixture, public ::testing::WithParamInterface<CTestParams> {
public:
	static bool InitTestFixture() { return true; }
	static void DeinitTestFixture() {}
};

} // namespace NeoMLTest

using namespace NeoML;
using namespace NeoMLTest;

TEST_F( CDnnSerializationTest, CnnUninitializedLayerSerialization )
{
	CRandom random( 42 );
	CDnn cnn( random, MathEngine() );

	CObjectArray<CBaseLayer> layers;
	layers.Add( new CSourceLayer( MathEngine() ) );
	layers.Add( new CSinkLayer( MathEngine() ) );
	layers.Add( new CConcatChannelsLayer( MathEngine() ) );
	layers.Add( new CConcatDepthLayer( MathEngine() ) );
	layers.Add( new CConcatHeightLayer( MathEngine() ) );
	layers.Add( new CConcatWidthLayer( MathEngine() ) );
	layers.Add( new CConcatBatchWidthLayer( MathEngine() ) );
	layers.Add( new CConcatObjectLayer( MathEngine() ) );
	layers.Add( new CSplitChannelsLayer( MathEngine() ) );
	layers.Add( new CSplitDepthLayer( MathEngine() ) );
	layers.Add( new CSplitWidthLayer( MathEngine() ) );
	layers.Add( new CSplitHeightLayer( MathEngine() ) );
	layers.Add( new CSplitBatchWidthLayer( MathEngine() ) );
	layers.Add( new CEltwiseSumLayer( MathEngine() ) );
	layers.Add( new CEltwiseMulLayer( MathEngine() ) );
	layers.Add( new CEltwiseNegMulLayer( MathEngine() ) );
	layers.Add( new CEltwiseMaxLayer( MathEngine() ) );
	layers.Add( new CReLULayer( MathEngine() ) );
	layers.Add( new CAbsLayer( MathEngine() ) );
	layers.Add( new CSigmoidLayer( MathEngine() ) );
	layers.Add( new CTanhLayer( MathEngine() ) );
	layers.Add( new CHardTanhLayer( MathEngine() ) );
	layers.Add( new CHardSigmoidLayer( MathEngine() ) );
	layers.Add( new CHSwishLayer( MathEngine() ) );
	layers.Add( new CPowerLayer( MathEngine() ) );
	layers.Add( new CConvLayer( MathEngine() ) );
	layers.Add( new CRleConvLayer( MathEngine() ) );
	layers.Add( new CMaxPoolingLayer( MathEngine() ) );
	layers.Add( new CMeanPoolingLayer( MathEngine() ) );
	layers.Add( new CFullyConnectedLayer( MathEngine() ) );
	layers.Add( new CCrossEntropyLossLayer( MathEngine() ) );
	layers.Add( new CBinaryCrossEntropyLossLayer( MathEngine() ) );
	layers.Add( new CEuclideanLossLayer( MathEngine() ) );
	layers.Add( new CHingeLossLayer( MathEngine() ) );
	layers.Add( new CSquaredHingeLossLayer( MathEngine() ) );
	layers.Add( new CProblemSourceLayer( MathEngine() ) );
	layers.Add( new CBatchNormalizationLayer( MathEngine() ) );
	layers.Add( new CLinearLayer( MathEngine() ) );
	layers.Add( new CDropoutLayer( MathEngine() ) );
	layers.Add( new CImageResizeLayer( MathEngine() ) );
	layers.Add( new CMultichannelLookupLayer( MathEngine() ) );
	layers.Add( new CCompositeLayer( MathEngine() ) );
	layers.Add( new CSubSequenceLayer( MathEngine() ) );
	layers.Add( new CBackLinkLayer( MathEngine() ) );
	layers.Add( new CCaptureSinkLayer( MathEngine() ) );
	layers.Add( new CEnumBinarizationLayer( MathEngine() ) );
	layers.Add( new CSoftmaxLayer( MathEngine() ) );
	layers.Add( new CGlobalMaxPoolingLayer( MathEngine() ) );
	layers.Add( new CLstmLayer( MathEngine() ) );
	layers.Add( new CGruLayer( MathEngine() ) );
	layers.Add( new CMaxOverTimePoolingLayer( MathEngine() ) );
	layers.Add( new CTimeConvLayer( MathEngine() ) );
	layers.Add( new C3dConvLayer( MathEngine() ) );
	layers.Add( new C3dMaxPoolingLayer( MathEngine() ) );
	layers.Add( new C3dMeanPoolingLayer( MathEngine() ) );
	layers.Add( new CAttentionLayer( MathEngine() ) );
	layers.Add( new CTransformLayer( MathEngine() ) );
	layers.Add( new CTransposeLayer( MathEngine() ) );
	layers.Add( new CCtcLossLayer( MathEngine() ) );
	layers.Add( new CCtcDecodingLayer( MathEngine() ) );
	layers.Add( new CAttentionDecoderLayer( MathEngine() ) );
	layers.Add( new CAttentionRecurrentLayer( MathEngine() ) );
	layers.Add( new CAttentionLayer( MathEngine() ) );
	layers.Add( new CAttentionDotProductLayer( MathEngine() ) );
	layers.Add( new CAttentionSumLayer( MathEngine() ) );
	layers.Add( new CAttentionWeightedSumLayer( MathEngine() ) );

	const CString layerName = "LAYER";
	for( int layerIndex = 0; layerIndex < layers.Size(); ++layerIndex ) {
		layers[layerIndex]->SetName( layerName );
		cnn.AddLayer( *layers[layerIndex] );

		CString newVersionFileName = "test_archive.new_ver";

		{
			CArchiveFile archiveFile( newVersionFileName, CArchive::store, GetPlatformEnv() );
			CArchive archive( &archiveFile, CArchive::SD_Storing );
			archive.Serialize( cnn );
		}

		{
			CArchiveFile archiveFile( newVersionFileName, CArchive::load, GetPlatformEnv() );
			CArchive archive( &archiveFile, CArchive::SD_Loading );
			archive.Serialize( cnn );
		}

		cnn.DeleteLayer( layers[layerIndex]->GetName() );
	}
}

// ====================================================================================================================

struct CNamedBlob {
	CNamedBlob() : Blob( new CDnnBlob( MathEngine() ) ) {}

	CString Name;
	CPtr<CDnnBlob> Blob;
	
	void Serialize( CArchive& archive );
};

void CNamedBlob::Serialize( CArchive& archive )
{
	archive.Serialize( Name );
	Blob->Serialize( archive );
}

static CArchive& operator >> ( CArchive& archive, CNamedBlob& blob )
{
	blob.Serialize( archive );
	return archive;
}

static CArchive& operator << ( CArchive& archive, CNamedBlob& blob )
{
	blob.Serialize( archive );
	return archive;
}

// ====================================================================================================================

static void compareOutputBlobs( CPtr<CDnnBlob> expected, CPtr<CDnnBlob> actual, const CString& fileName )
{
	CArray<float> expectedBuff;
	expectedBuff.SetSize( expected->GetDataSize() );
	expected->CopyTo( expectedBuff.GetPtr() );

	CArray<float> actualBuff;
	actualBuff.SetSize( actual->GetDataSize() );
	CPtr<CDnnBlob> converted = actual->GetCopy();
	converted->CopyTo( actualBuff.GetPtr() );

	EXPECT_EQ( expectedBuff.Size(), actualBuff.Size() ) << fileName;
	for( int i = 0; i < expectedBuff.Size(); ++i ) {
		EXPECT_NEAR( expectedBuff[i], actualBuff[i], 1e-3 ) << fileName;
	}
}

static void checkNet( const CArray<CNamedBlob>& inputBlobs, const CArray<CNamedBlob>& outputBlobs,
	CDnn& cnn, const CString& fileName )
{
	for( int i = 0; i < inputBlobs.Size(); ++i ) {
		CheckCast<CSourceLayer>( cnn.GetLayer( inputBlobs[i].Name ) )->SetBlob( inputBlobs[i].Blob );
	}

	cnn.RunOnce();

	for( int i = 0; i < outputBlobs.Size(); ++i ) {
		compareOutputBlobs( outputBlobs[i].Blob,
			CheckCast<CSinkLayer>( cnn.GetLayer( outputBlobs[i].Name ) )->GetBlob(), fileName );
	}
}

static void checkSerializedNet( const CString& fileName )
{
	GTEST_LOG_( INFO ) << "Checking archive " << fileName;

	CRandom random( 0x12345 );
	CDnn cnn( random, MathEngine() );
	CArray<CNamedBlob> inputBlobs;
	CArray<CNamedBlob> outputBlobs;

	{
		CArchiveFile file( GetTestDataFilePath( "data/SerializationTestData", fileName ), CArchive::load, GetPlatformEnv() );
		CArchive archive( &file, CArchive::SD_Loading );

		archive.Serialize( cnn );
		archive.Serialize( inputBlobs );
		archive.Serialize( outputBlobs );
	}

	checkNet( inputBlobs, outputBlobs, cnn, fileName );

	GTEST_LOG_( INFO ) << "Checking current serialization...";

	{
		CString newVersionFileName = "test_archive.new_ver";
		{
			CArchiveFile archiveFile( newVersionFileName, CArchive::store, GetPlatformEnv() );
			CArchive archive( &archiveFile, CArchive::SD_Storing );
			// Сохраняем сеть в архив.
			archive.Serialize( cnn );
		}
		// Загружаем из этого же архива.
		{
			CArchiveFile archiveFile( newVersionFileName, CArchive::load, GetPlatformEnv() );
			CArchive archive( &archiveFile, CArchive::SD_Loading );
			archive.Serialize( cnn );
		}
	}

	checkNet( inputBlobs, outputBlobs, cnn, fileName );
}

TEST_P( CDnnSerializationTest, PreviousVersions )
{
	CTestParams params = GetParam();
	const CString fileName = params.GetStrValue( "FileName" );
	checkSerializedNet( fileName );
}

INSTANTIATE_TEST_CASE_P(CDnnSerializationTestInstantiation, CDnnSerializationTest,
	::testing::Values(
		CTestParams(
			"FileName = FmlCnn3dConvLayer.1001.arch;"
		),
		CTestParams(
			"FileName = FmlCnn3dConvLayerWith1x1x1.1001.arch;"
		),
		CTestParams(
			"FileName = FmlCnnAccuracyLayer.1001.arch;"
		),
		CTestParams(
			"FileName = FmlCnnAttentionAdditive.1001.arch;"
		),
		CTestParams(
			"FileName = FmlCnnAttentionDotProduct.1001.arch;"
		),
		CTestParams(
			"FileName = FmlCnnChannelwiseConvLayer.1001.arch;"
		),
		CTestParams(
			"FileName = FmlCnnConfusionMatrixLayer.1001.arch;"
		),
		CTestParams(
			"FileName = FmlCnnConvLayer.1001.arch;"
		),
		CTestParams(
			"FileName = FmlCnnConvLayerWith1x1.1001.arch;"
		),
		CTestParams(
			"FileName = FmlCnnFullyConnectedLayer.1001.arch;"
		),
		CTestParams(
			"FileName = FmlCnnGlobalMaxPoolingLayer.1001.arch;"
		),
		CTestParams(
			"FileName = FmlCnnLstmLayer.1001.arch;"
		),
		CTestParams(
			"FileName = FmlCnnPrecisionRecallLayer.1001.arch;"
		),
		CTestParams(
			"FileName = FmlCnnReLULayer.1001.arch;"
		)
	)
);


