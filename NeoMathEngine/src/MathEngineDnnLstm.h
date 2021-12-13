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

#pragma once

#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoMathEngine/CrtAllocatedObject.h>
#include <MemoryHandleInternal.h>

namespace NeoML {

struct CMathEngineLstmDesc : public CLstmDesc {
	CMathEngineLstmDesc( const CFloatHandle& _inputWeights, const CFloatHandle* _inputFreeTerm,
		const CFloatHandle& _recurrentWeights, const CFloatHandle* _recurrentFreeTerm,
		const CFloatHandle& _inputFullyConnectedResult, const CFloatHandle& _reccurentFullyConnectedResult,
		int _hiddenSize, int _objectCount, int _objectSize, IMathEngine* _mathEngine, int _threadCount ) :
		inputWeights( _inputWeights ), inputFreeTerm( _inputFreeTerm ),
		recurrentWeights( _recurrentWeights ), recurrentFreeTerm( _recurrentFreeTerm ),
		inputFullyConnectedResult( _inputFullyConnectedResult ), reccurentFullyConnectedResult( _reccurentFullyConnectedResult ),
		hiddenSize( _hiddenSize ), objectCount( _objectCount ), objectSize( _objectSize ),
		mathEngine( _mathEngine ), threadCount( _threadCount )
	{}

	static int constexpr GatesNum = 4;

	const CFloatHandle& inputWeights;
	const CFloatHandle* inputFreeTerm;
	const CFloatHandle& recurrentWeights;
	const CFloatHandle* recurrentFreeTerm;
	const CFloatHandle& inputFullyConnectedResult;
	const CFloatHandle& reccurentFullyConnectedResult;
	int hiddenSize; 
	int objectCount;
	int objectSize;
	IMathEngine* mathEngine;
	int threadCount;

	virtual void SimdRunOnceRestOfLstm( const CConstFloatHandle& inputStateBackLink,
		const CFloatHandle& outputStateBackLink, const CFloatHandle& outputMainBackLink );

	virtual void CalcTanh( float* data, size_t dataSize ) {}
};

void CMathEngineLstmDesc::SimdRunOnceRestOfLstm( const CConstFloatHandle& inputStateBackLink, 
	const CFloatHandle& outputStateBackLink, const CFloatHandle& outputMainBackLink )
{
	// Elementwise summ of fully connected layers' results (inplace)
	const int ResultMatrixHeight = objectCount;
	const int ResultMatrixWidth = CMathEngineLstmDesc::GatesNum * hiddenSize;
	const CFloatHandle& hiddenLayerSum = inputFullyConnectedResult;
	mathEngine->VectorAdd( inputFullyConnectedResult, reccurentFullyConnectedResult, hiddenLayerSum, ResultMatrixHeight* ResultMatrixWidth );

	// Rearrange sum
	const CFloatHandle& hiddenLayerSumRearranged = reccurentFullyConnectedResult;
	const int DataSize = ResultMatrixHeight * hiddenSize;
	CFloatHandle inputTanhData = hiddenLayerSumRearranged;
	CFloatHandle forgetData = inputTanhData + DataSize;
	CFloatHandle inputData = forgetData + DataSize;
	CFloatHandle outputData = inputData + DataSize;

	int objectSize = hiddenSize;
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
	mathEngine->VectorTanh( inputTanhData, inputTanhData, DataSize );
	mathEngine->VectorSigmoid( forgetData, forgetData, DataSize );
	mathEngine->VectorSigmoid( inputData, inputData, DataSize );
	mathEngine->VectorSigmoid( outputData, outputData, DataSize );

	// Multiply input gates
	mathEngine->VectorEltwiseMultiply( inputData, inputTanhData, inputData, DataSize );

	// Multiply state backlink with forget gate
	mathEngine->VectorEltwiseMultiply( forgetData, inputStateBackLink, forgetData, DataSize );

	// Append input gate to state backlink
	mathEngine->VectorAdd( forgetData, inputData, outputStateBackLink, DataSize );

	// Apply tanh to state baclink
	mathEngine->VectorTanh( outputStateBackLink, inputData, DataSize );

	// Multiply output gate with result of previous operation
	mathEngine->VectorEltwiseMultiply( outputData, inputData, outputMainBackLink, DataSize );
}

} // namespace NeoML
