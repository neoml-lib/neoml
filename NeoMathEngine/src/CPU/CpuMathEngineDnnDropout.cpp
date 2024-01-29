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

	const float* first;
	float* result;

	if(!desc.IsSpatial) {

		const float* first;
		float* result;
		int currSize;

		for(int i = 0; i < (maskSize + cacheSize - 1) / cacheSize; ++i) {
			currSize = std::min(cacheSize, maskSize - i * cacheSize);
			int idx = 0;

			first = inputPointer;
			result = outputPointer;

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
	} else {
		const int reloadIter = cacheSize / objectSize;
		float* curr = mask;

		for(int i = 0; i < batchWidth; ++i) {
			if( !(i % reloadIter) ) {
				int idx = 0;
				for(int i = 0; i < (desc.Mask.Size() + 3) / 4; ++i) {
					generated = random.Next();
					for(int j = 0; j < maskAlign && idx < desc.Mask.Size(); ++j) {
						mask[idx++] = (generated[j] <= desc.threshold) ? desc.value : 0.f;
					}
				}
				curr = mask;
			}

			first = inputPointer;
			result = outputPointer;

			for(int j = 0; j < batchLength; ++j) {
				multiplyMatrixByDiagMatrix(first, input.ObjectSize() / objectSize, objectSize, curr, result);

				first += inputObjectSize * batchWidth;
				result += inputObjectSize * batchWidth;
			}

			inputPointer += inputObjectSize;
			outputPointer += inputObjectSize;
			curr += objectSize;
		}
	}
}

} // namespace NeoML
