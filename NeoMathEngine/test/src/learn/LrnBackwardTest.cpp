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

#include <TestFixture.h>

using namespace NeoML;
using namespace NeoMLTest;

class CLrnBackwardTest : public CTestFixtureWithParams {
};

TEST_F( CLrnBackwardTest, Precalc )
{
	const int height = 2;
	const int width = 3;
	const int channels = 7;
	const int dataSize = height * width * channels;

	std::vector<float> input = { -1.5f, -1.41463415f, -1.32926829f, -1.24390244f, -1.15853659f, -1.07317073f,
		-0.98780488f, -0.90243902f, -0.81707317f, -0.73170732f, -0.64634146f, -0.56097561f, -0.47560976f,
		-0.3902439f, -0.30487805f, -0.2195122f, -0.13414634f, -0.04878049f, 0.03658537f, 0.12195122f,
		0.20731707f, 0.29268293f, 0.37804878f, 0.46341463f, 0.54878049f, 0.63414634f, 0.7195122f, 0.80487805f,
		0.8902439f, 0.97560976f, 1.06097561f, 1.14634146f, 1.23170732f, 1.31707317f, 1.40243902f, 1.48780488f,
		1.57317073f, 1.65853659f, 1.74390244f, 1.82926829f, 1.91463415f, 2.f };

	CFloatBlob inputBlob( MathEngine(), 1, 1, 1, height, width, 1, channels );
	inputBlob.CopyFrom( input.data() );
	std::unique_ptr<CLrnDesc> desc( MathEngine().InitLrn( inputBlob.GetDesc(), 5, 3.f, 2e-2f, 0.81f ) );

	CFloatBlob invSum( MathEngine(), 1, 1, 1, height, width, 1, channels );

	CFloatBlob invSumBeta( MathEngine(), 1, 1, 1, height, width, 1, channels );

	CFloatBlob outputBlob( MathEngine(), 1, 1, 1, height, width, 1, channels );
	MathEngine().Lrn( *desc, inputBlob.GetData(), invSum.GetData(), invSumBeta.GetData(), outputBlob.GetData() );
	std::vector<float> output( dataSize );
	outputBlob.CopyTo( output.data() );

	std::vector<float> expectedOutput = { -0.6120848f, -0.57629544f, -0.5407432f, -0.50661045f, -0.4723609f, -0.43838462f,
		-0.4041842f, -0.36983216f, -0.33469746f, -0.29962757f, -0.26483864f, -0.22998759f, -0.19510205f, -0.16015589f,
		-0.12519395f, -0.0901394f, -0.05508512f, -0.02003264f, 0.01502456f, 0.05008285f, 0.08514106f, 0.12014931f,
		0.15514243f, 0.19009213f, 0.22500427f, 0.25986353f, 0.29491338f, 0.33001018f, 0.36449975f, 0.3988879f,
		0.43308502f, 0.467459f, 0.50172466f, 0.5371442f, 0.572765f, 0.60618573f, 0.6388899f, 0.67116714f, 0.7046276f,
		0.73792887f, 0.7746171f, 0.81177235f };

	for( int i = 0; i < dataSize; ++i ) {
		ASSERT_NEAR( expectedOutput[i], output[i], 1e-5f ) << " at index " << i;
	}

	std::vector<float> outputDiff = { -0.98930526f, 0.011084413f, 0.011485575f, 0.011884379f, 0.012298462f, 0.012723494f,
		0.013166173f, 0.013626314f, 0.014113582f, 0.014617326f, 0.015134797f, 0.01567156f, 0.016227918f, 0.016805043f,
		0.017402975f, 0.018023841f, 0.018666862f, 0.019332785f, 0.020022556f, 0.020736966f, 0.021476857f, 0.022242045f,
		0.023034142f, 0.023853408f, 0.02470089f, 0.025577132f, 0.026489504f, 0.027435703f, 0.028398458f, 0.029392011f,
		0.030414516f, 0.03147816f, 0.032575477f, 0.03374996f, 0.034973834f, 0.03616243f, 0.03736465f, 0.038590346f,
		0.03990344f, 0.041254655f, 0.042796314f, 0.04441634f };

	CFloatBlob outputDiffBlob( MathEngine(), 1, 1, 1, height, width, 1, channels );
	outputDiffBlob.CopyFrom( outputDiff.data() );

	CFloatBlob inputDiffBlob( MathEngine(), 1, 1, 1, height, width, 1, channels );
	MathEngine().VectorFill( inputDiffBlob.GetData(), 0, dataSize );

	MathEngine().LrnBackward( *desc, inputBlob.GetData(), outputBlob.GetData(), outputDiffBlob.GetData(),
		invSum.GetData(), invSumBeta.GetData(), inputDiffBlob.GetData() );
	
	std::vector<float> expectedInputDiff = {  -0.40178648f, 0.0062948572f, 0.006327679f, 0.0047603627f, 0.0049425554f,
		0.0051452f, 0.0053518494f, 0.005556773f, 0.0057494077f, 0.0059513804f, 0.0061738086f, 0.006403416f, 0.0066431174f,
		0.0068888185f, 0.007143109f, 0.0073987474f, 0.007663832f, 0.007939199f, 0.008222553f, 0.008515513f, 0.008818706f,
		0.009123777f, 0.009439355f, 0.009761674f, 0.01009429f, 0.010435193f, 0.0108124325f, 0.011208206f, 0.011560023f,
		0.011912636f, 0.012264244f, 0.012654357f, 0.013051952f, 0.01356906f, 0.0141198225f, 0.014506743f, 0.014840572f,
		0.015157925f, 0.015600058f, 0.01604618f, 0.016795773f, 0.017605262f };
	
	std::vector<float> inputDiff( dataSize );
	inputDiffBlob.CopyTo( inputDiff.data() );

	for( int i = 0; i < dataSize; ++i ) {
		ASSERT_NEAR( expectedInputDiff[i], inputDiff[i], 1e-5f ) << " at index " << i;
	}
}
