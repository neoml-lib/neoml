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
#include <MemoryHandleInternal.h>
#include <CpuMathEngine.h>
#include <CpuMathEnginePrivate.h>
#include <CpuExecutionScope.h>
#include <CpuRandom.h>
#include <CPUInfo.h>

namespace NeoML {

CDropoutDesc* CCpuMathEngine::InitDropout( float rate, bool isSpatial, bool isBatchwise,
	const CBlobDesc& input, const CBlobDesc& output, int seed )
{
	return new CSeedDropoutDesc(rate, isSpatial, isBatchwise, input, output, seed);
}

static constexpr int maskAlign = 4;

static void FillCpuDropoutMask( CCpuRandom& random, float* const curr, const CSeedDropoutDesc& desc, const int& index, const int& size )
{
	const unsigned threshold = (unsigned int)((double)desc.ForwardRate * UINT_MAX);
	const float value = 1.f / desc.ForwardRate;

	ASSERT_EXPR((index % maskAlign) == 0);
	const int currBlock = index / maskAlign;
	random.Skip(currBlock);
	int idx = 0;

	const int alignedSize = (size + ( maskAlign - 1) ) / maskAlign;
	for(int i = 0; i < alignedSize; ++i) {
		CIntArray<maskAlign> generated = random.Next();
		for (int j = 0; j < maskAlign && idx < size; ++j) {
			curr[idx++] = (generated[j] <= threshold) ? value : 0.f;
		}
	}
}

void CCpuMathEngine::Dropout( const CDropoutDesc& dropoutDesc, const CFloatHandle& inputData, const CFloatHandle& outputData )
{
	CCpuExecutionScope scope;

	const CSeedDropoutDesc& desc = static_cast<const CSeedDropoutDesc&>( dropoutDesc );
	const CBlobDesc& input = desc.Input;

	if( desc.ForwardRate == 1.f ) {
		VectorCopy( outputData, inputData, input.BlobSize() );
		return;
	}

	const int objectSize = desc.IsSpatial ? input.Channels() : input.ObjectSize();
	const int batchLength = desc.IsBatchwise ? input.ObjectCount() : input.BatchLength();
	const int batchWidth = input.ObjectCount() / batchLength;
	const int maskSize = batchWidth * objectSize;

	CFloatHandle currInput = inputData;
	CFloatHandle currOutput = outputData;

	if( !desc.IsSpatial ) {
		CCpuRandom random( desc.seed );
		// will construct the dropout mask by samples of L1 cache line size
		int cacheSize = static_cast<int>( std::min( CCPUInfo::GetCPUInfo().L1CacheSize, static_cast<size_t>(INT_MAX) ) );
		cacheSize = std::min( cacheSize, maskSize );
		// aligning and getting convertingn bytes to num of float elements
		cacheSize -= (cacheSize % maskAlign) / sizeof(float);
		
		CFloatHandleStackVar maskVar( mathEngine(), cacheSize );
		float* const mask = GetRaw( maskVar.GetHandle() );

		for( int i = 0; i < (maskSize + cacheSize - 1) / cacheSize; ++i ) {
			const int currSize = std::min( cacheSize, maskSize - i * cacheSize );
			FillCpuDropoutMask( random, mask, desc, 0, currSize );

			const float* first = GetRaw( currInput );
			const float* second = mask;
			float* result = GetRaw( currOutput );

			for( int b = 0; b < batchLength; ++b ) {
				NeoML::vectorEltwiseMultiply( first, second, result, currSize );

				first += maskSize;
				result += maskSize;
			}

			currInput += currSize;
			currOutput += currSize;
		}
	} else {
		const int inputObjectSize = input.ObjectSize();
		for(int i = 0; i < batchWidth; ++i) {
			const int index = i * objectSize;
			CCpuRandom random( desc.seed );
			CFloatHandleStackVar maskVar( mathEngine(), objectSize + index % maskAlign );

			float* mask = GetRaw( maskVar.GetHandle() );
			FillCpuDropoutMask( random, mask, desc, index - index % maskAlign, objectSize + index % maskAlign );
			mask += ( index % maskAlign );

			const float* first = GetRaw( currInput );
			float* result = GetRaw( currOutput );

			for(int j = 0; j < batchLength; ++j) {
				multiplyMatrixByDiagMatrix( first, input.ObjectSize() / objectSize, objectSize, mask, result );

				first += inputObjectSize * batchWidth;
				result += inputObjectSize * batchWidth;
			}

			currInput += inputObjectSize;
			currOutput += inputObjectSize;
		}
	}
}

} // namespace NeoML
