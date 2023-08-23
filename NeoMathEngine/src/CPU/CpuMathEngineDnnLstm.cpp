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

#include <common.h>
#pragma hdrstop

#include <algorithm>
#include <memory>

#include <CpuMathEngine.h>
#include <CpuMathEnginePrivate.h>
#include <CpuMathEngineDnnLstm.h>
#include <NeoMathEngine/NeoMathEngineException.h>
#include <NeoMathEngine/OpenMP.h>

namespace NeoML {

CMathEngineLstmDesc::~CMathEngineLstmDesc() = default;

CLstmDesc* CCpuMathEngine::InitLstm( CLstmDesc* currentDesc, int hiddenSize, int objectSize )
{
	if( currentDesc != nullptr ) {
		static_cast<CMathEngineLstmDesc*>( currentDesc )->Reset( hiddenSize, objectSize );
		return currentDesc;
	} else {
		return new CMathEngineLstmDesc( hiddenSize, objectSize );
	}
}

template<class T>
class CSequenceWrapper {
public:
	//CSequenceWrapper( float* buffer, int sequenceLength, int objectSize ) :
	//	buffer( buffer ), sequenceLength( sequenceLength ), objectSize( objectSize ) {}
	CSequenceWrapper( const CTypedMemoryHandle<T>& handle, int sequenceLength, int elemSize ) :
		buffer( GetRaw( handle ) ), sequenceLength( sequenceLength ), elemSize( elemSize ) {}
	CSequenceWrapper( const CSequenceWrapper& ) = delete;
	CSequenceWrapper( CSequenceWrapper&& ) = delete;
	CSequenceWrapper& operator=( const CSequenceWrapper& ) = delete;
	CSequenceWrapper& operator=( CSequenceWrapper&& ) = delete;
	~CSequenceWrapper() = default;

	int ElemSize() const { return elemSize; }
	int SequenceLength() const { return sequenceLength; }
	T* operator[]( int index ) { return buffer + ( index % sequenceLength ) * elemSize; }

private:
	T* buffer;
	int sequenceLength;
	int elemSize;
};

void CCpuMathEngine::Lstm( CLstmDesc& desc, bool reverse, int sequenceLength, int sequenceCount,
	const CConstFloatHandle& inputWeights, const CConstFloatHandle& inputFreeTerm,
	const CConstFloatHandle& recurrentWeights, const CConstFloatHandle& recurrentFreeTerm,
	const CConstFloatHandle& inputStateBackLink, const CConstFloatHandle& inputMainBackLink,
	const CConstFloatHandle& inputHandle, const CFloatHandle& outputStateBackLink,
	const CFloatHandle& outputMainBackLink )
{
	CMathEngineLstmDesc& lstmDesc = dynamic_cast<CMathEngineLstmDesc&>( desc );

	auto initializeBacklink = [&] ( const CConstFloatHandle& initialState, CSequenceWrapper<float>& wrapper )
	{
		const int firstElemIdx = reverse ? sequenceLength - 1 : 0;
		if( initialState.IsNull() ) {
			vectorFill0( wrapper[firstElemIdx], wrapper.ElemSize() );
		} else {
			dataCopy( wrapper[firstElemIdx], GetRaw( initialState ), wrapper.ElemSize() );
		}
	};

	// Write state data directly to output or create temporary blob for recurent
	std::unique_ptr<CFloatHandleStackVar> stateBackLinkVar;
	if( outputStateBackLink.IsNull() ) {
		stateBackLinkVar.reset( new CFloatHandleStackVar( *this, sequenceCount * lstmDesc.hiddenSize ) );
	}
	CSequenceWrapper<float> stateBackLink(
		outputStateBackLink.IsNull() ? stateBackLinkVar->GetHandle() : outputStateBackLink,
		outputStateBackLink.IsNull() ? 1 : sequenceLength,
		sequenceCount * lstmDesc.hiddenSize );
	initializeBacklink( inputStateBackLink, stateBackLink );

	// Create temporary blobs for result of fully connected layers
	const int fcLen = std::min( sequenceLength, ( 64 + sequenceCount - 1 ) / sequenceCount );
	CFloatHandleStackVar fullyConnectedResultVar( *this, fcLen * sequenceCount * 4 * lstmDesc.hiddenSize );
	CSequenceWrapper<float> fullyConnectedResult( fullyConnectedResultVar,
		fcLen, sequenceCount * 4 * lstmDesc.hiddenSize );

	// Emulate working of LSTM recurrent implementation
	CSequenceWrapper<float> mainBackLink( outputMainBackLink, sequenceLength, sequenceCount * lstmDesc.hiddenSize );
	initializeBacklink( inputMainBackLink, mainBackLink );

	CSequenceWrapper<const float> input( inputHandle, sequenceLength, sequenceCount * lstmDesc.objectSize );

	CConstFloatHandle freeTerm;
	std::unique_ptr<CFloatHandleStackVar> freeTermVar;
	if( !inputFreeTerm.IsNull() && !recurrentFreeTerm.IsNull() ) {
		freeTermVar.reset( new CFloatHandleStackVar( *this, 4 * lstmDesc.hiddenSize ) );
		freeTerm = freeTermVar->GetHandle();
		vectorAdd( GetRaw( inputFreeTerm ), GetRaw( recurrentFreeTerm ),
			GetRaw( freeTermVar->GetHandle() ), freeTermVar->Size() );
	} else if( !inputFreeTerm.IsNull() ) {
		freeTerm = inputFreeTerm;
	} else {
		freeTerm = recurrentFreeTerm;
	}

	// Iterate recurent net step by step
	int seqElemsInBuffer = 0;
	for( int i = 0; i < sequenceLength; i++ ) {
		int inputPos, outputPos;
		if( reverse ) {
			const int LastIdx = sequenceLength - 1;
			int iRev = LastIdx - i;
			inputPos = std::min( LastIdx, iRev + 1 );
			outputPos = iRev;
		} else {
			inputPos = std::max( 0, i - 1 );
			outputPos = i;
		}

		// Precalculate portion of input multiplied by weights and with free terms added
		if( seqElemsInBuffer == 0 ) {
			const int bufferIdx = outputPos / fullyConnectedResult.SequenceLength();
			seqElemsInBuffer = std::min( fullyConnectedResult.SequenceLength(),
				sequenceLength - bufferIdx * fullyConnectedResult.SequenceLength() );
			multiplyMatrixByTransposedMatrix( input[bufferIdx * fullyConnectedResult.SequenceLength()],
				seqElemsInBuffer * sequenceCount, lstmDesc.objectSize, lstmDesc.objectSize, GetRaw( inputWeights ),
				4 * lstmDesc.hiddenSize, lstmDesc.objectSize, fullyConnectedResult[0], 4 * lstmDesc.hiddenSize );
		}

		multiplyMatrixByTransposedMatrixAndAdd( mainBackLink[inputPos], sequenceCount, lstmDesc.hiddenSize,
			lstmDesc.hiddenSize, GetRaw( recurrentWeights ), 4 * lstmDesc.hiddenSize, lstmDesc.hiddenSize,
			fullyConnectedResult[outputPos], 4 * lstmDesc.hiddenSize );
		if( !freeTerm.IsNull() ) {
			addVectorToMatrixRows( fullyConnectedResult[outputPos], fullyConnectedResult[outputPos],
				sequenceCount, 4 * lstmDesc.hiddenSize, 4 * lstmDesc.hiddenSize, 4 * lstmDesc.hiddenSize,
				GetRaw( freeTerm ) );
		}

		// if outputMainBackLink != output then we are in compatibility mode
		if( simdMathEngine != nullptr ) {
			simdMathEngine->RunOnceRestOfLstm( &lstmDesc, sequenceCount, fullyConnectedResult[outputPos],
				stateBackLink[inputPos], stateBackLink[outputPos], mainBackLink[outputPos] );
		} else {
			lstmDesc.RunOnceRestOfLstm( sequenceCount, fullyConnectedResult[outputPos], stateBackLink[inputPos],
				stateBackLink[outputPos], mainBackLink[outputPos] );
		}
		--seqElemsInBuffer;
	}
}

} // namespace NeoML
