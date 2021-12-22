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

#include <MathEngineDnnDropout.h>
#include <CpuMathEngine.h>
#include <CpuExecutionScope.h>
#include <CpuRandom.h>

namespace NeoML {

void CCpuMathEngine::Dropout( const CDropoutDesc& dropoutDesc, const CFloatHandle& inputData, const CFloatHandle& outputData )
{
	CCpuExecutionScope scope;

	const CMathEngineDropoutDesc& desc = static_cast<const CMathEngineDropoutDesc&>( dropoutDesc );
	const CBlobDesc& input = desc.Input;
	const CBlobDesc& output = desc.Output;

	if( desc.ForwardRate == 1.f ) {
		VectorCopy( outputData, inputData, input.BlobSize() );
		return;
	}

	const int objectSize = desc.IsSpatial ? input.Channels() : input.ObjectSize();
	const int batchLength = desc.IsBatchwise ? input.ObjectCount() : input.BatchLength();
	const int batchWidth = input.ObjectCount() / batchLength;
	const int maskSize = batchWidth * objectSize;

	ASSERT_EXPR( desc.Mask.Size() == maskSize );

	if( !desc.IsSpatial ) {
		MultiplyMatrixByDiagMatrix( inputData, batchLength, maskSize, desc.Mask.GetHandle(), outputData,
			output.BlobSize() );
		return;
	}

	CFloatHandle currInput = inputData;
	CFloatHandle currOutput = outputData;
	for( int i = 0; i < input.ObjectCount(); ++i ) {
		mathEngine().MultiplyMatrixByDiagMatrix( currInput, input.ObjectSize() / objectSize, objectSize,
			desc.Mask.GetHandle() + ( i % batchWidth ) * objectSize, currOutput, input.ObjectSize() );
		currInput += input.ObjectSize();
		currOutput += input.ObjectSize();
	}
}

CDropoutDesc* CCpuMathEngine::InitDropout( float rate, bool isSpatial, bool isBatchwise,
	const CBlobDesc& input, const CBlobDesc& output, int seed )
{
	return new CMathEngineDropoutDesc( mathEngine(), rate, isSpatial, isBatchwise, input, output, seed );
}

} // namespace NeoML
