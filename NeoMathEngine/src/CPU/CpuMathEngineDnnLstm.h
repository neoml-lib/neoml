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
#include <memory>

namespace NeoML {

struct CMathEngineLstmDesc : public CLstmDesc {
	CMathEngineLstmDesc( int hiddenSize, int objectSize, const CConstFloatHandle& inputWeights,
		const CConstFloatHandle& inputFreeTerm, const CConstFloatHandle& recurWeights,
		const CConstFloatHandle& recurFreeTerm );
	~CMathEngineLstmDesc() override;

	static int constexpr GatesNum = 4;

	const int HiddenSize;
	const int ObjectSize;
	const std::unique_ptr<CFloatHandleVar> InputWeightsVar;
	const float* const InputWeights;
	const std::unique_ptr<CFloatHandleVar> RecurWeightsVar;
	const float* const RecurWeights;
	const std::unique_ptr<CFloatHandleVar> FreeTermVar;
	const float* FreeTerm;

	virtual void RunOnceRestOfLstm( int objectCount, float* fullyConnectedResult, const float* inputStateBackLink,
		float* outputStateBackLink, float* outputMainBackLink );
};

inline void CMathEngineLstmDesc::RunOnceRestOfLstm( int objectCount, float* fullyConnectedResult,
	const float* inputStateBackLink, float* outputStateBackLink, float* outputMainBackLink )
{
	// Elementwise summ of fully connected layers' results (inplace)
	const int resultMatrixWidth = CMathEngineLstmDesc::GatesNum * HiddenSize;

	// Rearrange sum
	float* inputTanhData = fullyConnectedResult;
	float* forgetData = inputTanhData + HiddenSize;
	float* inputData = forgetData + HiddenSize;
	float* outputData = inputData + HiddenSize;

	for( int i = 0; i < objectCount; ++i ) {
		// Apply activations
		NeoML::vectorTanh( inputTanhData, inputTanhData, HiddenSize );

		NeoML::vectorSigmoid( forgetData, forgetData, HiddenSize );
		NeoML::vectorSigmoid( inputData, inputData, HiddenSize );
		NeoML::vectorSigmoid( outputData, outputData, HiddenSize );

		// Multiply input gates
		NeoML::vectorEltwiseMultiply( inputData, inputTanhData, inputData, HiddenSize );

		// Multiply state backlink with forget gate
		NeoML::vectorEltwiseMultiply( forgetData, inputStateBackLink, forgetData, HiddenSize );

		// Append input gate to state backlink
		NeoML::vectorAdd( forgetData, inputData, outputStateBackLink, HiddenSize );

		// Apply tanh to state baclink
		NeoML::vectorTanh( outputStateBackLink, inputData, HiddenSize );

		// Multiply output gate with result of previous operation
		NeoML::vectorEltwiseMultiply( outputData, inputData, outputMainBackLink, HiddenSize );

		inputTanhData += resultMatrixWidth;
		forgetData += resultMatrixWidth;
		inputData += resultMatrixWidth;
		outputData += resultMatrixWidth;
		inputStateBackLink += HiddenSize;
		outputStateBackLink += HiddenSize;
		outputMainBackLink += HiddenSize;
	}
}

} // namespace NeoML
