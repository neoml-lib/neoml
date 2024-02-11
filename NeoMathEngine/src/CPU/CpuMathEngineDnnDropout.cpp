/* Copyright © 2017-2024 ABBYY Production LLC

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

namespace NeoML {

static constexpr int maskAlign = 4;

CDropoutDesc* CCpuMathEngine::InitDropout(float rate, bool isSpatial, bool isBatchwise,
	const CBlobDesc& input, const CBlobDesc& output, int seed)
{
	return new CSeedDropoutDesc(mathEngine(), rate, isSpatial, isBatchwise, input, output, seed);
}

void CCpuMathEngine::Dropout(const CDropoutDesc& dropoutDesc, const CFloatHandle& inputData, const CFloatHandle& outputData)
{
	CCpuExecutionScope scope;

	const CSeedDropoutDesc& desc = static_cast<const CSeedDropoutDesc&>(dropoutDesc);
	const CBlobDesc& input = desc.Input;

	if( desc.ForwardRate == 1.f ) {
		VectorCopy( outputData, inputData, input.BlobSize() );
		return;
	}

	const int objectSize = desc.IsSpatial ? input.Channels() : input.ObjectSize();
	const int batchLength = desc.IsBatchwise ? input.ObjectCount() : input.BatchLength();
	const int batchWidth = input.ObjectCount() / batchLength;
	const int maskSize = batchWidth * objectSize;
	
	CCpuRandom random(desc.seed);
	CIntArray<maskAlign> generated;

	const int inputObjectSize = input.ObjectSize();
	const float* inputPointer = GetRaw(inputData);
	float* outputPointer = GetRaw(outputData);
	float* mask = GetRaw(desc.Mask.GetHandle());
	const int cacheSize = desc.Mask.Size();

	if(!desc.IsSpatial) {
		const int numOfIter = (maskSize + cacheSize - 1) / cacheSize;
		int currSize = cacheSize;

		for(int i = 0; i < numOfIter; ++i) {
			if (i == numOfIter - 1) {
				currSize = maskSize - i * cacheSize;
			}

			int idx = 0;
			const float* first = inputPointer;
			float* result = outputPointer;

			const int alignedSize = (currSize + (maskAlign - 1)) / maskAlign;
			for(int i = 0; i < alignedSize; ++i) {
				generated = random.Next();
				for (int j = 0; j < maskAlign && idx < currSize; ++j) {
					mask[idx++] = (generated[j] <= desc.threshold) ? desc.value : 0.f;
				}
			}

			for(int b = 0; b < batchLength; ++b) {
				vectorEltwiseMultiply(first, mask, result, currSize);

				first += maskSize;
				result += maskSize;
			}

			inputPointer += currSize;
			outputPointer += currSize;
		}
	}
	else {
		const int numOfIter = (objectSize + cacheSize - 1) / cacheSize;
		const int alignedSize = (objectSize + (maskAlign - 1)) / maskAlign;

		for (int i = 0; i < batchWidth; ++i) {
			const float* first = inputPointer;
			float* result = outputPointer;
			int currSize = cacheSize;
			for (int j = 0; j < alignedSize; ++j) {
				int idx = 0;
				if (j == numOfIter - 1) {
					currSize = objectSize - j * cacheSize;
				}
				generated = random.Next();
				for (int k = 0; k < maskAlign && idx < currSize; ++k) {
					mask[idx++] = (generated[k] <= desc.threshold) ? desc.value : 0.f;
				}

				for (int j = 0; j < batchLength; ++j) {
					const float* localFirst = first;
					float* localResult = result;
					for (int k = 0; k < inputObjectSize / objectSize; ++k) {
						vectorEltwiseMultiply(localFirst, mask, localResult, currSize);
						localFirst += objectSize;
						localResult += objectSize;
					}
					first += batchWidth * inputObjectSize;
					result += batchWidth * inputObjectSize;
				}
			}
			inputPointer += inputObjectSize;
			outputPointer += inputObjectSize;
		}
	}
}

} // namespace NeoML
