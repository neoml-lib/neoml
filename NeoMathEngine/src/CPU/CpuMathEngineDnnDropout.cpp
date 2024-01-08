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

CDropoutDesc* CCpuMathEngine::InitDropout(float rate, bool isSpatial, bool isBatchwise,
	const CBlobDesc& input, const CBlobDesc& output, int seed)
{
	return new CSeedDropoutDesc(rate, isSpatial, isBatchwise, input, output, seed);
}

static void FillCpuDropoutMask(CCpuRandom& random, float* const curr, const CSeedDropoutDesc& desc, const int& index, const int& size)
{
	const unsigned int threshold = (unsigned int)((double)desc.ForwardRate * UINT_MAX);
	const float value = 1.f / desc.ForwardRate;
	const int currBlock = index / 4;
	const int leftToFill = (currBlock + 1) * 4 - index;
	random.Skip(currBlock);
	int idx = 0;

	if( leftToFill % 4 ) {
		CIntArray<4> generated = random.Next();
		for(; idx < leftToFill && idx < size; ++idx) {
			curr[idx] = (generated[4 - leftToFill + idx] <= threshold) ? value : 0.f;
		}
	}

	const int num = (size - idx + 3) / 4;
	for(int t = 0; t < num; ++t) {
		CIntArray<4> generated = random.Next();
		for (int j = 0; j < 4 && idx < size; ++j) {
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
		CCpuRandom random(desc.seed);
		// will construct the dropout mask by samples of L1 cache line size
		int cacheSize = static_cast<int>(std::min(CCPUInfo::GetCPUInfo().L1CacheSize, static_cast<size_t>(INT_MAX)));
		cacheSize = std::min(cacheSize, maskSize);
		// make it divisible by 4 to simplify random numbers generation
		cacheSize -= (cacheSize % 4);

		CFloatHandleVar mask( mathEngine(), cacheSize );

		for(int i = 0; i < (maskSize + cacheSize - 1) / cacheSize; ++i) {
			const int currSize = std::min<int>(cacheSize, maskSize - i * cacheSize);
			FillCpuDropoutMask(random, GetRaw(mask.GetHandle()), desc, 0, currSize);

			const float* first = GetRaw(currInput);
			const float* second = GetRaw(mask.GetHandle());
			float* result = GetRaw(currOutput);

			for (int b = 0; b < batchLength; ++b) {
				multiplyMatrixByDiagMatrix(first, 1, currSize, second, result);
				first += maskSize;
				result += maskSize;
			}

			currInput += currSize;
			currOutput += currSize;
		}
		return;
	}

	for( int i = 0; i < input.ObjectCount(); ++i ) {
		CCpuRandom random(desc.seed);
		CFloatHandleVar mask(mathEngine(), objectSize);
		const int index = (i % batchWidth) * objectSize;
		FillCpuDropoutMask(random, GetRaw(mask.GetHandle()), desc, index, objectSize);

		MultiplyMatrixByDiagMatrix(currInput, input.ObjectSize() / objectSize, objectSize,
			mask, currOutput, input.ObjectSize());

		currInput += input.ObjectSize();
		currOutput += input.ObjectSize();
	}
}

} // namespace NeoML
