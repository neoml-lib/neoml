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

	virtual void RunOnceRestOfLstm( int objectCount, float* fullyConnectedResult, const float* inputStateBackLink,
		float* outputStateBackLink, float* outputMainBackLink );
};

inline void CMathEngineLstmDesc::RunOnceRestOfLstm( int objectCount, float* fullyConnectedResult,
	const float* inputStateBackLink, float* outputStateBackLink, float* outputMainBackLink )
{
	// Elementwise summ of fully connected layers' results (inplace)
	const int resultMatrixWidth = CMathEngineLstmDesc::GatesNum * hiddenSize;

	// Rearrange sum
	float* inputTanhData = fullyConnectedResult;
	float* forgetData = inputTanhData + hiddenSize;
	float* inputData = forgetData + hiddenSize;
	float* outputData = inputData + hiddenSize;

	for( int i = 0; i < objectCount; ++i ) {
		// Apply activations
		NeoML::vectorTanh( inputTanhData, inputTanhData, hiddenSize );

		NeoML::vectorSigmoid( forgetData, forgetData, hiddenSize );
		NeoML::vectorSigmoid( inputData, inputData, hiddenSize );
		NeoML::vectorSigmoid( outputData, outputData, hiddenSize );

		// Multiply input gates
		NeoML::vectorEltwiseMultiply( inputData, inputTanhData, inputData, hiddenSize );

		// Multiply state backlink with forget gate
		NeoML::vectorEltwiseMultiply( forgetData, inputStateBackLink, forgetData, hiddenSize );

		// Append input gate to state backlink
		NeoML::vectorAdd( forgetData, inputData, outputStateBackLink, hiddenSize );

		// Apply tanh to state baclink
		NeoML::vectorTanh( outputStateBackLink, inputData, hiddenSize );

		// Multiply output gate with result of previous operation
		NeoML::vectorEltwiseMultiply( outputData, inputData, outputMainBackLink, hiddenSize );

		inputTanhData += resultMatrixWidth;
		forgetData += resultMatrixWidth;
		inputData += resultMatrixWidth;
		outputData += resultMatrixWidth;
		inputStateBackLink += hiddenSize;
		outputStateBackLink += hiddenSize;
		outputMainBackLink += hiddenSize;
	}
}

} // namespace NeoML
