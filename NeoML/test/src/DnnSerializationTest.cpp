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

namespace NeoMLTest {

class CDnnSerializationTest : public CNeoMLTestFixture, public ::testing::WithParamInterface<CTestParams> {
public:
	static bool InitTestFixture() { return true; }
	static void DeinitTestFixture() {}
};

} // namespace NeoMLTest

using namespace NeoML;
using namespace NeoMLTest;

// Checks that every layer can be serialized in uninitialized state
TEST_F( CDnnSerializationTest, CnnUninitializedLayerSerialization )
{
	CRandom random( 42 );
	CDnn cnn( random, MathEngine() );

	CArray<const char*> layerClasses;
	GetRegisteredLayerClasses( layerClasses );

	const CString layerName = "LAYER";
	for( int layerIndex = 0; layerIndex < layerClasses.Size(); ++layerIndex ) {
		{
			CPtr<CBaseLayer> layer = CreateLayer( layerClasses[layerIndex], MathEngine() );
			layer->SetName( layerName );
			cnn.AddLayer( *layer );
		}

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

		cnn.DeleteLayer( layerName );
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
	CRandom random( 0x12345 );
	CDnn cnn( random, MathEngine() );
	CArray<CNamedBlob> inputBlobs;
	CArray<CNamedBlob> outputBlobs;

	{
		CArchiveFile file( GetTestDataFilePath( "data/SerializationTestData", fileName ),
			CArchive::load, GetPlatformEnv() );
		CArchive archive( &file, CArchive::SD_Loading );

		archive.Serialize( cnn );
		archive.Serialize( inputBlobs );
		archive.Serialize( outputBlobs );
	}

	checkNet( inputBlobs, outputBlobs, cnn, fileName );

	// Check current serialization
	{
		CString newVersionFileName = "test_archive.new_ver";
		{
			// Store the net
			CArchiveFile archiveFile( newVersionFileName, CArchive::store, GetPlatformEnv() );
			CArchive archive( &archiveFile, CArchive::SD_Storing );
			archive.Serialize( cnn );
		}

		{
			// Load from the same file
			CArchiveFile archiveFile( newVersionFileName, CArchive::load, GetPlatformEnv() );
			CArchive archive( &archiveFile, CArchive::SD_Loading );
			archive.Serialize( cnn );
		}
	}

	checkNet( inputBlobs, outputBlobs, cnn, fileName );
}

// Checks serialization of the old versions of CDnn
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
