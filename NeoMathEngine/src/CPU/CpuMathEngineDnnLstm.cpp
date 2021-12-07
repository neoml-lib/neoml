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

#include <CpuMathEngine.h>
#include <CpuMathEnginePrivate.h>
#include <MathEngineDnnLstm.h>
#include <MemoryHandleInternal.h>
#include <NeoMathEngine/NeoMathEngineException.h>
#include <NeoMathEngine/OpenMP.h>

namespace NeoML {

CLstmDesc* CCpuMathEngine::InitLstm( const CFloatHandle& inputWeights, const CFloatHandle* inputFreeTerm,
	const CFloatHandle& recurrentWeights, const CFloatHandle* recurrentFreeTerm,
	const CFloatHandle& inputFullyConnectedResult, const CFloatHandle& reccurentFullyConnectedResult,
	int hiddenSize, int objectCount, int objectSize )
{
	return new CMathEngineLstmDesc( inputWeights, inputFreeTerm,
		recurrentWeights, recurrentFreeTerm,
		inputFullyConnectedResult, reccurentFullyConnectedResult,
		hiddenSize, objectCount, objectSize );
}

void CCpuMathEngine::Lstm( CLstmDesc& desc, const CConstFloatHandle& inputStateBackLink, const CConstFloatHandle& inputMainBackLink, const CConstFloatHandle& input,
	const CFloatHandle& outputStateBackLink, const CFloatHandle& outputMainBackLink )
{
	CMathEngineLstmDesc& lstmDesc = dynamic_cast< CMathEngineLstmDesc& >( desc );

	auto fullyConnectedRunOnce = [&]( const CConstFloatHandle& input, int inputHeight, int inputWidth,
		const CFloatHandle& weights, int weightHeight, int weightsWidth,
		const CFloatHandle& output, const CFloatHandle* freeTerm ) {
		MultiplyMatrixByTransposedMatrix( input, inputHeight, inputWidth, inputWidth,
			weights, weightsWidth, weightHeight,
			output, weightsWidth, inputHeight * weightsWidth );

		if( freeTerm != nullptr ) {
			AddVectorToMatrixRows( 1, output, output, inputHeight,
				weightsWidth, *freeTerm );
		}
	};

	//-----------------------------------------------------------------------------------------------------------------

	// Apply fully connected layers
	fullyConnectedRunOnce( input, lstmDesc.objectCount, lstmDesc.objectSize,
		lstmDesc.inputWeights, lstmDesc.objectSize, CMathEngineLstmDesc::GatesNum * lstmDesc.hiddenSize,
		lstmDesc.inputFullyConnectedResult, lstmDesc.inputFreeTerm );

	fullyConnectedRunOnce( inputMainBackLink, lstmDesc.objectCount, lstmDesc.hiddenSize,
		lstmDesc.recurrentWeights, lstmDesc.hiddenSize, CMathEngineLstmDesc::GatesNum * lstmDesc.hiddenSize,
		lstmDesc.reccurentFullyConnectedResult, lstmDesc.recurrentFreeTerm );

	if( lstmDesc.hasSimdImplementations ) {
		lstmDesc.simdRunOnceOfLstm( GetRaw( inputStateBackLink ), GetRaw( outputStateBackLink ), GetRaw( outputMainBackLink ) );
	} else {
		// Elementwise summ of fully connected layers' results (inplace)
		const int ResultMatrixHeight = lstmDesc.objectCount;
		const int ResultMatrixWidth = CMathEngineLstmDesc::GatesNum * lstmDesc.hiddenSize;
		const CFloatHandle& hiddenLayerSum = lstmDesc.inputFullyConnectedResult;
		VectorAdd( lstmDesc.inputFullyConnectedResult, lstmDesc.reccurentFullyConnectedResult, hiddenLayerSum, ResultMatrixHeight * ResultMatrixWidth );

		// Rearrange sum
		const CFloatHandle& hiddenLayerSumRearranged = lstmDesc.reccurentFullyConnectedResult;
		const int DataSize = ResultMatrixHeight * lstmDesc.hiddenSize;
		CFloatHandle inputTanhData = hiddenLayerSumRearranged;
		CFloatHandle forgetData = inputTanhData + DataSize;
		CFloatHandle inputData = forgetData + DataSize;
		CFloatHandle outputData = inputData + DataSize;

		int objectSize = lstmDesc.hiddenSize;
		float* rawFrom = GetRaw( hiddenLayerSum );
		float* rawTo = GetRaw( hiddenLayerSumRearranged );
		for( int x = 0; x < ResultMatrixHeight; x++ ) {
			const float* input = rawFrom + x * ResultMatrixWidth;
			for( int i = 0; i < CMathEngineLstmDesc::GatesNum; ++i ) {
				memcpy( ( rawTo + i * DataSize ) + x * objectSize, input, objectSize * sizeof( float ) );
				input += objectSize;
			}
		}

		// Apply activations
		VectorTanh( inputTanhData, inputTanhData, DataSize );
		VectorSigmoid( forgetData, forgetData, DataSize );
		VectorSigmoid( inputData, inputData, DataSize );
		VectorSigmoid( outputData, outputData, DataSize );

		// Multiply input gates
		VectorEltwiseMultiply( inputData, inputTanhData, inputData, DataSize );

		// Multiply state backlink with forget gate
		VectorEltwiseMultiply( forgetData, inputStateBackLink, forgetData, DataSize );

		// Append input gate to state backlink
		VectorAdd( forgetData, inputData, outputStateBackLink, DataSize );

		// Apply tanh to state baclink
		VectorTanh( outputStateBackLink, inputData, DataSize );

		// Multiply output gate with result of previous operation
		VectorEltwiseMultiply( outputData, inputData, outputMainBackLink, DataSize );
	}
}

} // namespace NeoML
