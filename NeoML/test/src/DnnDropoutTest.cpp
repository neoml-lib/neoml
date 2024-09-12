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

struct CDropoutTestParam final {
	CDropoutTestParam( bool isBatchwise ) : IsBatchwise( isBatchwise ) {}

	bool IsBatchwise;
};

class CDnnDropoutTest : public CNeoMLTestFixture, public ::testing::WithParamInterface<CDropoutTestParam> {
public:
	static bool InitTestFixture() { return true; }
	static void DeinitTestFixture() {}
};

class CDnnDropoutDummyLearn : public CDnnSimpleTestDummyLearningLayer {
public:
	explicit CDnnDropoutDummyLearn( IMathEngine& mathEngine ) : CDnnSimpleTestDummyLearningLayer( mathEngine ) {}

	CPtr<CDnnBlob> GetDiff() { return diff; }

protected:
	void LearnOnce() override { diff = outputDiffBlobs[0]->GetCopy(); }

private:
	CPtr<CDnnBlob> diff;
};

//---------------------------------------------------------------------------------------------------------------------

static void checkDropoutIsSpatial( int batchLength, int batchWidth, int channels, int objectSize,
	const CArray<float>& forwardData, const CArray<float>& backwardData )
{
	EXPECT_EQ( forwardData.Size(), backwardData.Size() );
	CArray<int> mask;
	int maskSum = 0;
	mask.SetSize( batchWidth * channels );

	int index = 0;
	for( int seq = 0; seq < batchLength; ++seq ) {
		for( int batch = 0; batch < batchWidth; ++batch ) {
			for( int ch = 0; ch < channels; ++ch ) {
				if( seq == 0 ) {
					// we calculate the mask (random Bernoulli vector from dropout) based on the first element
					// it should be the same for all other elements of the sequence
					mask[batch * channels + ch] = forwardData[index + ch] > 0.f ? 1 : 0;
					maskSum += mask[batch * channels + ch];
				}
				for( int obj = 0; obj < objectSize / channels; ++obj ) {
					if( mask[batch * channels + ch] > 0 ) {
						EXPECT_LE( 2.f, forwardData[index + obj * channels + ch] ) << "Index: " << index + obj * channels + ch;
						EXPECT_LE( 2.f, backwardData[index + obj * channels + ch] ) << "Index: " << index + obj * channels + ch;
					} else {
						EXPECT_FLOAT_EQ( 0.f, forwardData[index + obj * channels + ch] ) << "Index: " << index + obj * channels + ch;
						EXPECT_FLOAT_EQ( 0.f, backwardData[index + obj * channels + ch] ) << "Index: " << index + obj * channels + ch;
					}
				}
			}
			index += objectSize;
		}
	}
	EXPECT_LT( 0, maskSum );
	EXPECT_GT( mask.Size(), maskSum );
}

static void checkDropoutIsNotSpatial( int batchLength, int batchWidth, int channels, int objectSize,
	const CArray<float>& forwardData, const CArray<float>& backwardData )
{
	EXPECT_EQ( forwardData.Size(), backwardData.Size() );
	CArray<int> channelFlags;
	CArray<int> mask;
	int maskSum = 0;
	channelFlags.SetSize( batchWidth * channels );
	mask.SetSize( batchWidth * objectSize );

	int index = 0;
	for( int seq = 0; seq < batchLength; ++seq ) {
		for( int i = 0; i < channelFlags.Size(); ++i ) {
			channelFlags[i] = 0;
		}
		for( int batch = 0; batch < batchWidth; ++batch ) {
			for( int ch = 0; ch < channels; ++ch ) {
				for( int obj = 0; obj < objectSize / channels; ++obj ) {
					// since dropout did not use spatial, the size of the objects is large and the probability is 1/2
					// then no channel should be completely zeroed or completely non-zeroed
					channelFlags[batch * channels + ch] |= forwardData[index + obj * channels + ch] > 0.f ? 1 : 2;
					if( seq == 0 ) {
						// calculate mask (random Bernoulli vector from dropout) based on first element
						// it should be the same for all others
						mask[batch * objectSize + ch * ( objectSize / channels ) + obj] = forwardData[index + obj * channels + ch] > 0.f ? 1 : 0;
						maskSum += mask[batch * objectSize + ch * ( objectSize / channels ) + obj];
					}
					if( mask[batch * objectSize + ch * ( objectSize / channels ) + obj] > 0 ) {
						EXPECT_LE( 2.f, forwardData[index + obj * channels + ch] ) << "Index: " << index;
						EXPECT_LE( 2.f, backwardData[index + obj * channels + ch] ) << "Index: " << index;
					} else {
						EXPECT_FLOAT_EQ( 0.f, forwardData[index + obj * channels + ch] ) << "Index: " << index;
						EXPECT_FLOAT_EQ( 0.f, backwardData[index + obj * channels + ch] ) << "Index: " << index;
					}
				}
			}
			index += objectSize;
		}
		// check that spatial didn't work
		for( int i = 0; i < channelFlags.Size(); ++i ) {
			EXPECT_EQ( 3, channelFlags[i] ) << "Index: " << i;
		}
	}
	EXPECT_LT( 0, maskSum );
	EXPECT_GT( mask.Size(), maskSum );
}

static CPtr<CDropoutLayer> testSerialization( CPtr<CDropoutLayer> dropout, CDnn& net,
	CBaseLayer* input, CBaseLayer* loss, CBaseLayer* output )
{
	const CString name = dropout->GetName();
	const float rate = dropout->GetDropoutRate();
	const bool isSpatial = dropout->IsSpatial();
	const bool isBatchwise = dropout->IsBatchwise();

	net.DeleteLayer( *dropout );
	{
		CMemoryFile archiveFile;
		CArchive archive( &archiveFile, CArchive::SD_Storing );
		dropout->Serialize( archive );
		archive.Close();
		archiveFile.SeekToBegin();
		archive.Open( &archiveFile, CArchive::SD_Loading );
		dropout.Release();
		dropout = new CDropoutLayer( MathEngine() );
		dropout->Serialize( archive );
		archive.Close();
	}

	EXPECT_EQ( name, dropout->GetName() );
	EXPECT_TRUE( FloatEq( rate, dropout->GetDropoutRate() ) );
	EXPECT_EQ( isSpatial, dropout->IsSpatial() );
	EXPECT_EQ( isBatchwise, dropout->IsBatchwise() );

	dropout->Connect( *input );
	net.AddLayer( *dropout );
	loss->Connect( 0, *dropout, 0 );
	loss->Connect( 1, *dropout, 0 );
	output->Connect( *dropout );

	return dropout;
}

} // namespace NeoMLTest

//---------------------------------------------------------------------------------------------------------------------

TEST_F( CDnnDropoutTest, ReproducibleRandom )
{
	const int dataSize = 64 * 20 - 3;
	const int runCount = 3;

	CPtr<CDnnBlob> input = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 1, dataSize );
	input->Fill( 1 );
	CPtr<CDnnBlob> output = input->GetCopy();

	const unsigned __int64 expected[] = { 0xb91aa9ed42b44156, 0x4a2fa863cd5728d7, 0x1bded6825caf7369,
		0x74ed0c083c48a072, 0x12b359abc1f84ca6, 0x37e5e6052034e4d7, 0x694795139162370, 0x1d468d6dbf212722,
		0xe1e9f0182fe8913e, 0xa734f6c904d880ef, 0x354b2d8bfb3fab17, 0x2ab9e0be565dce6e, 0xf37adfced74142f3,
		0x1634692360fb4347, 0xef6480851ec66e9a, 0x1d9b2f1ab4d35a9a, 0x33f7dd0769e3d426, 0x2e0274b98b7ce053,
		0x7733133684565913, 0x1e446a05b3d6197b };

	for( int run = 0; run < runCount; ++run ) {
		CRandom random( 0x282 );
		CDropoutDesc* dropoutDesc = MathEngine().InitDropout( 0.5, false, false,
			input->GetDesc(), output->GetDesc(), random.Next() );
		MathEngine().Dropout( *dropoutDesc, input->GetData(), output->GetData() );
		delete dropoutDesc;

		CArray<float> buff;
		buff.SetSize( dataSize );
		output->CopyTo( buff.GetPtr() );

		unsigned __int64 actual[( dataSize + 63 ) / 64];
		for( int i = 0; i < dataSize; ++i ) {
			if( i % 64 == 0 ) {
				actual[i / 64] = 0;
			}
			if( buff[i] > 0 ) {
				actual[i / 64] |= 1ULL << ( i % 64 );
			}
		}

		for( int i = 0; i < ( dataSize + 63 ) / 64; ++i ) {
			EXPECT_EQ( expected[i], actual[i] );
		}
	}
}

TEST_P( CDnnDropoutTest, SpatialForward )
{
	const bool isBatchwise = GetParam().IsBatchwise;

	const int channels = 17;
	const int batchLength = 13;
	const int batchWidth = 11;
	const int height = 7;
	const int width = 5;
	const int depth = 3;
	const int objectSize = channels * depth * height * width;
	const bool isSpatial = true;

	CRandom random( 0xcaef );
	CDnn cnn( random, MathEngine() );

	CPtr<CDnnBlob> inputBlob = CDnnBlob::Create3DImageBlob( MathEngine(), CT_Float,
		batchLength, batchWidth, height, width, depth, channels );

	CArray<float> buffer;
	buffer.SetSize( inputBlob->GetDataSize() );
	for( int i = 0; i < buffer.Size(); ++i ) {
		buffer[i] = static_cast<float>( random.Uniform( 1., 2. ) );
	}
	inputBlob->CopyFrom( buffer.GetPtr() );

	CPtr<CSourceLayer> input = Source( cnn, "input" );
	CPtr<CDnnDropoutDummyLearn> learn = new CDnnDropoutDummyLearn( MathEngine() );
	learn->Connect( *input );
	cnn.AddLayer( *learn );

	CPtr<CDropoutLayer> dropout = Dropout( 0.5f, isSpatial, isBatchwise )( learn.Ptr() );
	CPtr<CSinkLayer> output = Sink( dropout.Ptr(), "Sink" );

	CPtr<CDnnSimpleTestDummyLossLayer> loss = new CDnnSimpleTestDummyLossLayer( MathEngine() );
	loss->Connect( 0, *dropout, 0 );
	loss->Connect( 1, *dropout, 0 );
	cnn.AddLayer( *loss );

	input->SetBlob( inputBlob->GetCopy() );
	loss->Diff = input->GetBlob()->GetCopy();
	loss->Diff->Fill( 1.f );
	dropout = testSerialization( dropout, cnn, learn, loss, output );
	cnn.RunAndBackwardOnce();

	CPtr<CDnnBlob> result = output->GetBlob();/////////////

	CArray<float> backwardBuffer;
	backwardBuffer.SetSize( result->GetDataSize() );
	buffer.SetSize( result->GetDataSize() );
	result->CopyTo( buffer.GetPtr() );
	learn->GetDiff()->CopyTo( backwardBuffer.GetPtr() );

	checkDropoutIsSpatial(
		isBatchwise ? ( batchLength * batchWidth ) : batchLength,
		isBatchwise ? 1 : batchWidth,
		channels, objectSize, buffer, backwardBuffer );

	input->SetBlob( inputBlob->GetCopy() );
	dropout->SetSpatial( false );
	loss->Diff = input->GetBlob()->GetCopy();
	loss->Diff->Fill( 1.f );
	dropout = testSerialization( dropout, cnn, learn, loss, output );
	cnn.RunAndBackwardOnce();

	result = output->GetBlob();/////////////
	result->CopyTo( buffer.GetPtr() );
	learn->GetDiff()->CopyTo( backwardBuffer.GetPtr() );

	checkDropoutIsNotSpatial(
		isBatchwise ? ( batchLength * batchWidth ) : batchLength,
		isBatchwise ? 1 : batchWidth,
		channels, objectSize, buffer, backwardBuffer );
}

INSTANTIATE_TEST_CASE_P( CDnnDropoutTestInstantiation, CDnnDropoutTest,
	::testing::Values(
		CDropoutTestParam( false ),
		CDropoutTestParam( true )
	)
);
