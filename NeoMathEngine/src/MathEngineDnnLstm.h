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

namespace NeoML {

struct CMathEngineLstmDesc : public CLstmDesc {
	CMathEngineLstmDesc( const CFloatHandle& _inputWeights, const CFloatHandle* _inputFreeTerm,
		const CFloatHandle& _recurrentWeights, const CFloatHandle* _recurrentFreeTerm,
		const CFloatHandle& _inputFullyConnectedResult, const CFloatHandle& _reccurentFullyConnectedResult,
		int _hiddenSize, int _objectCount, int _objectSize ) :
		inputWeights( _inputWeights ), inputFreeTerm( _inputFreeTerm ),
		recurrentWeights( _recurrentWeights ), recurrentFreeTerm( _recurrentFreeTerm ),
		inputFullyConnectedResult( _inputFullyConnectedResult ), reccurentFullyConnectedResult( _reccurentFullyConnectedResult ),
		hiddenSize( _hiddenSize ), objectCount( _objectCount ), objectSize( _objectSize ),
		hasSimdImplementations( false ), simdRunOnceOfLstm( nullptr )
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

	bool hasSimdImplementations;

	//typedef void (*SimdRunOnceRestOfLstm)( float* inputResult, const float* recurentResult, int inputHeight, int inputWidth,
	//	float* outputStateBackLink, float* outputMainBackLink, int ouputHeight, int outputWidth );
	typedef void (*SimdRunOnceRestOfLstm)( const float* inputStateBackLink, float* outputStateBackLink, float* outputMainBackLink );

	SimdRunOnceRestOfLstm simdRunOnceOfLstm;
};

} // namespace NeoML
