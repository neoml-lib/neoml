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

static void checkOutput( const CSinkLayer& sink, CArray<float>& output, const float* expected, int size )
{
	EXPECT_EQ( size, sink.GetBlob()->GetDataSize() );
	output.SetSize( size );
	sink.GetBlob()->CopyTo( output.GetPtr() );

	for( int i = 0; i < size; ++i ) {
		EXPECT_NEAR( output[i], expected[i], 1e-3f ) << " i=" << i;
	}
}

} // namespace NeoMLTest

//---------------------------------------------------------------------------------------------------------------------

TEST( DnnGeluTest, ConsistencyTest )
{
	CRandom random( 45 );
	CDnn dnn( random, MathEngine() );

	const int batchSize = 8;
	const int channels = 3;

	CSourceLayer* source = Source( dnn, "source" );
	{ // Set source data
		const float sourceData[batchSize * channels]{
			 0.0f, 1.0f, 2.0f, 3.0f,  4.0f,  5.0f,
			 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
			-1.0f,-2.0f,-3.0f,-4.0f, -5.0f, -6.0f,
			 1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f
		};
		CPtr<CDnnBlob> sourceBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, batchSize, channels );
		sourceBlob->CopyFrom( sourceData );
		source->SetBlob( sourceBlob );
	}

	CGELULayer* gelu = Gelu()( "gelu", source );
	EXPECT_EQ( gelu->GetCalculationMode(), CGELUActivationParam::TCalculationMode::CM_SigmoidApproximate );

	const CSinkLayer* sink = Sink( gelu, "sink" );
	CArray<float> output;

	dnn.RunOnce();
	{ // Check the forward calculations
		const float expected[batchSize * channels]{
			 0.000000,  0.845796,  1.935659,  2.981929,  3.995585,  4.998993,
			 5.999780,  6.999952,  7.999990,  8.999998,  9.999999, 10.999999,
			-0.154204, -0.064341, -0.018071, -0.004415, -0.001007, -0.000220,
			 0.845796,  1.935659,  2.981929,  3.995585,  4.998993,  5.999780
		};
		checkOutput( *sink, output, expected, batchSize * channels );
	}

	gelu->SetCalculationMode( CGELUActivationParam::TCalculationMode::CM_Precise );
	EXPECT_EQ( gelu->GetCalculationMode(), CGELUActivationParam::TCalculationMode::CM_Precise );

	dnn.RunOnce();
	{ // Check the forward calculations
		const float expected[batchSize * channels]{
			  0.000000,  0.841345,  1.954500,  2.995950,  3.999873,  4.999999,
			  6.000000,  7.000000,  8.000000,  9.000000, 10.000000, 11.000000,
			 -0.158655, -0.045500, -0.004050, -0.000127, -0.000001, -0.000000,
			  0.841345,  1.954500,  2.995950,  3.999873,  4.999999,  6.000000
		};
		checkOutput( *sink, output, expected, batchSize * channels );
	}
}
