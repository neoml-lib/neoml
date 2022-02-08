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
#include <CpuMathEngineDnnLstm.h>
#include <NeoMathEngine/NeoMathEngineException.h>
#include <NeoMathEngine/OpenMP.h>

namespace NeoML {

CLstmDesc* CCpuMathEngine::InitLstm( CLstmDesc* currentDesc, const CFloatHandle& inputFullyConnectedResult, const CFloatHandle& reccurentFullyConnectedResult,
	int hiddenSize, int objectCount, int objectSize )
{
	if( currentDesc != nullptr ) {
		*currentDesc = CMathEngineLstmDesc( inputFullyConnectedResult, reccurentFullyConnectedResult,
			hiddenSize, objectCount, objectSize, this, threadCount );
		return currentDesc;
	} else {
		return new CMathEngineLstmDesc( inputFullyConnectedResult, reccurentFullyConnectedResult,
			hiddenSize, objectCount, objectSize, this, threadCount );
	}
}

void CCpuMathEngine::Lstm( CLstmDesc& desc, 
	const CFloatHandle& inputWeights, const CConstFloatHandle& inputFreeTerm,
	const CFloatHandle& recurrentWeights, const CConstFloatHandle& recurrentFreeTerm,
	const CConstFloatHandle& inputStateBackLink, const CConstFloatHandle& inputMainBackLink, const CConstFloatHandle& input,
	const CFloatHandle& outputStateBackLink, const CFloatHandle& outputMainBackLink )
{
	CMathEngineLstmDesc& lstmDesc = dynamic_cast< CMathEngineLstmDesc& >( desc );

	auto fullyConnectedRunOnce = [&]( const CConstFloatHandle& input, int inputHeight, int inputWidth,
		const CFloatHandle& weights, int weightHeight, int weightsWidth,
		const CFloatHandle& output, const CConstFloatHandle& freeTerm ) {
		MultiplyMatrixByTransposedMatrix( input, inputHeight, inputWidth, inputWidth,
			weights, weightsWidth, weightHeight,
			output, weightsWidth, inputHeight * weightsWidth );

		if( !freeTerm.IsNull() ) {
			AddVectorToMatrixRows( 1, output, output, inputHeight,
				weightsWidth, freeTerm );
		}
	};

	//-----------------------------------------------------------------------------------------------------------------
	// Apply fully connected layers
	fullyConnectedRunOnce( input, lstmDesc.objectCount, lstmDesc.objectSize,
		inputWeights, lstmDesc.objectSize, CMathEngineLstmDesc::GatesNum * lstmDesc.hiddenSize,
		lstmDesc.inputFullyConnectedResult, inputFreeTerm );

	fullyConnectedRunOnce( inputMainBackLink, lstmDesc.objectCount, lstmDesc.hiddenSize,
		recurrentWeights, lstmDesc.hiddenSize, CMathEngineLstmDesc::GatesNum * lstmDesc.hiddenSize,
		lstmDesc.reccurentFullyConnectedResult, recurrentFreeTerm );

	// if outputMainBackLink != output then we are in compatibility mode
	if( simdMathEngine != nullptr ) {
		simdMathEngine->RunOnceRestOfLstm( &lstmDesc, inputStateBackLink, outputStateBackLink, outputMainBackLink );
	} else {
		lstmDesc.RunOnceRestOfLstm( inputStateBackLink, outputStateBackLink, outputMainBackLink );
	}
}

} // namespace NeoML
