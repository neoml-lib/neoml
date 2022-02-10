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
#include <CpuMathEnginePrivate.h>
#include <MemoryHandleInternal.h>

namespace NeoML {

struct CMathEngineLstmDesc : public CLstmDesc {
	CMathEngineLstmDesc( const CFloatHandle& _inputFullyConnectedResult, const CFloatHandle& _reccurentFullyConnectedResult,
		int _hiddenSize, int _objectCount, int _objectSize, IMathEngine* _mathEngine, int _threadCount ) :
			inputFullyConnectedResult( _inputFullyConnectedResult ),
			reccurentFullyConnectedResult( _reccurentFullyConnectedResult ),
			hiddenSize( _hiddenSize ),
			objectCount( _objectCount ),
			objectSize( _objectSize ),
			mathEngine( _mathEngine ),
			threadCount( _threadCount )
	{}

	void Reset( const CFloatHandle& _inputFullyConnectedResult, const CFloatHandle& _reccurentFullyConnectedResult,
		int _hiddenSize, int _objectCount, int _objectSize, IMathEngine* _mathEngine, int _threadCount ) {
		inputFullyConnectedResult = _inputFullyConnectedResult;
		reccurentFullyConnectedResult = _reccurentFullyConnectedResult;
		hiddenSize = _hiddenSize;
		objectCount = _objectCount;
		objectSize = _objectSize;
		mathEngine = _mathEngine;
		threadCount = _threadCount;
	}

	static int constexpr GatesNum = 4;

	CFloatHandle inputFullyConnectedResult;
	CFloatHandle reccurentFullyConnectedResult;
	int hiddenSize; 
	int objectCount;
	int objectSize;
	IMathEngine* mathEngine;
	int threadCount;

	virtual void RunOnceRestOfLstm( const CConstFloatHandle& inputStateBackLink,
		const CFloatHandle& outputStateBackLink, const CFloatHandle& outputMainBackLink );
};

void CMathEngineLstmDesc::RunOnceRestOfLstm( const CConstFloatHandle& inputStateBackLink, 
	const CFloatHandle& outputStateBackLink, const CFloatHandle& outputMainBackLink )
{
	// Elementwise summ of fully connected layers' results (inplace)
	const int ResultMatrixWidth = CMathEngineLstmDesc::GatesNum * hiddenSize;

	const int curThreadCount = IsOmpRelevant( static_cast< int >( objectCount ) ) ? threadCount : 1;
	NEOML_OMP_NUM_THREADS( curThreadCount ) {
		int offt, count;
		if( OmpGetTaskIndexAndCount( static_cast< int >( objectCount ), offt, count ) ) {
			const int OffsetFullyConnectedResult = offt * ResultMatrixWidth;
			const int OffsetBackLink = offt * hiddenSize;
			const int CurDataSize = count * hiddenSize;
			const CFloatHandle& curInputFullyConnectedResult = inputFullyConnectedResult + OffsetFullyConnectedResult;
			const CFloatHandle& curReccurentFullyConnectedResult = reccurentFullyConnectedResult + OffsetFullyConnectedResult;
			
			const CFloatHandle& hiddenLayerSum = curInputFullyConnectedResult;
			mathEngine->VectorAdd( curInputFullyConnectedResult, curReccurentFullyConnectedResult,
				hiddenLayerSum, count * ResultMatrixWidth );

			// Rearrange sum
			const CFloatHandle& hiddenLayerSumRearranged = curReccurentFullyConnectedResult;
			CFloatHandle inputTanhData = hiddenLayerSumRearranged;
			CFloatHandle forgetData = inputTanhData + CurDataSize;
			CFloatHandle inputData = forgetData + CurDataSize;
			CFloatHandle outputData = inputData + CurDataSize;

			float* rawFrom = GetRaw( hiddenLayerSum );
			float* rawTo = GetRaw( hiddenLayerSumRearranged );
			for( int x = 0; x < count; x++ ) {
				const float* input = rawFrom + x * ResultMatrixWidth;
				for( int i = 0; i < CMathEngineLstmDesc::GatesNum; ++i ) {
					memcpy( ( rawTo + i * CurDataSize ) + x * hiddenSize, input, hiddenSize * sizeof( float ) );
					input += hiddenSize;
				}
			}

			// Apply activations
			NeoML::vectorTanh( GetRaw( inputTanhData ), GetRaw( inputTanhData ), CurDataSize );
			
			NeoML::vectorSigmoid( GetRaw( forgetData ), GetRaw( forgetData ), CurDataSize );
			NeoML::vectorSigmoid( GetRaw( inputData ), GetRaw( inputData ), CurDataSize );
			NeoML::vectorSigmoid( GetRaw( outputData ), GetRaw( outputData ), CurDataSize );
			
			// Multiply input gates
			NeoML::vectorEltwiseMultiply( GetRaw( inputData ), GetRaw( inputTanhData ), GetRaw( inputData ), CurDataSize );

			// Multiply state backlink with forget gate
			NeoML::vectorEltwiseMultiply( GetRaw( forgetData ), GetRaw( inputStateBackLink + OffsetBackLink ), GetRaw( forgetData ), CurDataSize );

			// Append input gate to state backlink
			NeoML::vectorAdd( GetRaw( forgetData ), GetRaw( inputData ), GetRaw( outputStateBackLink + OffsetBackLink ), CurDataSize );

			// Apply tanh to state baclink
			NeoML::vectorTanh( GetRaw( outputStateBackLink + OffsetBackLink ), GetRaw( inputData ), CurDataSize );

			// Multiply output gate with result of previous operation
			NeoML::vectorEltwiseMultiply( GetRaw( outputData ), GetRaw( inputData ), GetRaw( outputMainBackLink + OffsetBackLink ), CurDataSize );
		}
	}
}

} // namespace NeoML
