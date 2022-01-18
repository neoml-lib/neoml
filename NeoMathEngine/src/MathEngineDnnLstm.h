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
	CMathEngineLstmDesc( const CFloatHandle& _inputFullyConnectedResult, const CFloatHandle& _reccurentFullyConnectedResult,
		int _hiddenSize, int _objectCount, int _objectSize, IMathEngine* _mathEngine, int _threadCount ) :
		inputFullyConnectedResult( _inputFullyConnectedResult ), reccurentFullyConnectedResult( _reccurentFullyConnectedResult ),
		hiddenSize( _hiddenSize ), objectCount( _objectCount ), objectSize( _objectSize ),
		mathEngine( _mathEngine ), threadCount( _threadCount )
	{}

	static int constexpr GatesNum = 4;

	const CFloatHandle inputFullyConnectedResult;
	const CFloatHandle reccurentFullyConnectedResult;
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
	const int ResultMatrixHeight = objectCount;
	const int ResultMatrixWidth = CMathEngineLstmDesc::GatesNum * hiddenSize;
	const int DataSize = ResultMatrixHeight * hiddenSize;

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
			CFloatHandle forgetData = inputTanhData + DataSize;
			CFloatHandle inputData = forgetData + DataSize;
			CFloatHandle outputData = inputData + DataSize;

			int objectSize = hiddenSize;
			float* rawFrom = GetRaw( hiddenLayerSum );
			float* rawTo = GetRaw( hiddenLayerSumRearranged );
			for( int x = offt; x < offt + count; x++ ) {
				const float* input = rawFrom + x * ResultMatrixWidth;
				for( int i = 0; i < CMathEngineLstmDesc::GatesNum; ++i ) {
					memcpy( ( rawTo + i * DataSize ) + x * objectSize, input, objectSize * sizeof( float ) );
					input += objectSize;
				}
			}

			// Apply activations
			mathEngine->CalcTanh( inputTanhData, inputTanhData, CurDataSize, false );
			mathEngine->CalcSigmoid( forgetData, forgetData, CurDataSize, false );
			mathEngine->CalcSigmoid( inputData, inputData, CurDataSize, false );
			mathEngine->CalcSigmoid( outputData, outputData, CurDataSize, false );

			// Multiply input gates
			mathEngine->VectorEltwiseMultiply( inputData, inputTanhData, inputData, CurDataSize );

			// Multiply state backlink with forget gate
			mathEngine->VectorEltwiseMultiply( forgetData, inputStateBackLink + OffsetBackLink, forgetData, CurDataSize );

			// Append input gate to state backlink
			mathEngine->VectorAdd( forgetData, inputData, outputStateBackLink + OffsetBackLink, CurDataSize );

			// Apply tanh to state baclink
			mathEngine->CalcTanh( outputStateBackLink + OffsetBackLink, inputData, CurDataSize, false );

			// Multiply output gate with result of previous operation
			mathEngine->VectorEltwiseMultiply( outputData, inputData, outputMainBackLink + OffsetBackLink, CurDataSize );
		}
	}
}

} // namespace NeoML
