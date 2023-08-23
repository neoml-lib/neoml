/* Copyright © 2017-2023 ABBYY

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
#include <CpuMathEnginePrivate.h>
#include <MemoryHandleInternal.h>

#include <cstring>

namespace NeoML {

struct CMathEngineLstmDesc : public CLstmDesc {
	CMathEngineLstmDesc( int hiddenSize, int objectSize ) :
		hiddenSize( hiddenSize ),
		objectSize( objectSize )
	{}
	~CMathEngineLstmDesc() override;

	void Reset( int newHiddenSize, int newObjectSize )
	{
		hiddenSize = newHiddenSize;
		objectSize = newObjectSize;
	}

	static int constexpr GatesNum = 4;

	int hiddenSize; 
	int objectSize;

	virtual void RunOnceRestOfLstm( int objectCount, float* inputFullyConnectedResult,
		float* recurrentFullyConnectedResult, const float* inputStateBackLink,
		float* outputStateBackLink, float* outputMainBackLink );
};

inline void CMathEngineLstmDesc::RunOnceRestOfLstm( int objectCount, float* inputFullyConnectedResult,
	float* recurrentFullyConnectedResult, const float* inputStateBackLink,
	float* outputStateBackLink, float* outputMainBackLink )
{
	// Elementwise summ of fully connected layers' results (inplace)
	const int resultMatrixWidth = CMathEngineLstmDesc::GatesNum * hiddenSize;
	const int curDataSize = objectCount * hiddenSize;

	float* const hiddenLayerSum = inputFullyConnectedResult;

	NeoML::vectorAdd( inputFullyConnectedResult, recurrentFullyConnectedResult,
		hiddenLayerSum, objectCount * resultMatrixWidth );

	// Rearrange sum
	float* const hiddenLayerSumRearranged = recurrentFullyConnectedResult;
	float* const inputTanhData = hiddenLayerSumRearranged;
	float* const forgetData = inputTanhData + curDataSize;
	float* const inputData = forgetData + curDataSize;
	float* const outputData = inputData + curDataSize;

	float* const rawFrom = hiddenLayerSum;
	float* const rawTo = hiddenLayerSumRearranged;
	for( int x = 0; x < objectCount; ++x ) {
		const float* input = rawFrom + x * resultMatrixWidth;
		for( int i = 0; i < CMathEngineLstmDesc::GatesNum; ++i ) {
			memcpy( ( rawTo + i * curDataSize ) + x * hiddenSize, input, hiddenSize * sizeof( float ) );
			input += hiddenSize;
		}
	}

	// Apply activations
	NeoML::vectorTanh( inputTanhData, inputTanhData, curDataSize );

	NeoML::vectorSigmoid( forgetData, forgetData, curDataSize );
	NeoML::vectorSigmoid( inputData, inputData, curDataSize );
	NeoML::vectorSigmoid( outputData, outputData, curDataSize );

	// Multiply input gates
	NeoML::vectorEltwiseMultiply( inputData, inputTanhData, inputData, curDataSize );

	// Multiply state backlink with forget gate
	NeoML::vectorEltwiseMultiply( forgetData, inputStateBackLink, forgetData, curDataSize );

	// Append input gate to state backlink
	NeoML::vectorAdd( forgetData, inputData, outputStateBackLink, curDataSize );

	// Apply tanh to state baclink
	NeoML::vectorTanh( outputStateBackLink, inputData, curDataSize );

	// Multiply output gate with result of previous operation
	NeoML::vectorEltwiseMultiply( outputData, inputData, outputMainBackLink, curDataSize );
}

} // namespace NeoML
