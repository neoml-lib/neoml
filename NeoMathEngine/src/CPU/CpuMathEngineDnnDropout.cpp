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
	return new CSeedDropoutDesc(mathEngine(), rate, isSpatial, isBatchwise, input, output, seed);
}

static constexpr int maskAlign = 4;

static void FillCpuDropoutMask(CCpuRandom& random, float* const curr, const CSeedDropoutDesc& desc, const int& index, const int& size)
{
	ASSERT_EXPR((index % maskAlign) == 0);
	const int currBlock = index / maskAlign;
	random.Skip(currBlock);
	int idx = 0;
	CIntArray<maskAlign> generated;

	const int alignedSize = (size + (maskAlign - 1)) / maskAlign;
	for (int i = 0; i < alignedSize; ++i) {
		generated = random.Next();
		for (int j = 0; j < maskAlign && idx < size; ++j) {
			curr[idx++] = (generated[j] <= desc.threshold) ? desc.value : 0.f;
		}
	}
}

void CCpuMathEngine::Dropout(const CDropoutDesc& dropoutDesc, const CFloatHandle& inputData, const CFloatHandle& outputData)
{
	CCpuExecutionScope scope;

	const CSeedDropoutDesc& desc = static_cast<const CSeedDropoutDesc&>(dropoutDesc);
	const CBlobDesc& input = desc.Input;

	if (desc.ForwardRate == 1.f) {
		VectorCopy(outputData, inputData, input.BlobSize());
		return;
	}

	const int objectSize = desc.IsSpatial ? input.Channels() : input.ObjectSize();
	const int batchLength = desc.IsBatchwise ? input.ObjectCount() : input.BatchLength();
	const int batchWidth = input.ObjectCount() / batchLength;
	const int maskSize = batchWidth * objectSize;

	CFloatHandle currInput = inputData;
	CFloatHandle currOutput = outputData;

	if(!desc.IsSpatial) {
		CCpuRandom random(desc.seed);

		const float* inputPointer = GetRaw(inputData);
		float* outputPointer = GetRaw(outputData);
		float* mask = GetRaw(desc.Mask.GetHandle());
		CIntArray<maskAlign> generated;

		const float* first;
		float* result;
		int currSize;
		const int cacheSize = desc.Mask.Size();

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
				NeoML::vectorEltwiseMultiply(first, mask, result, currSize);

				first += maskSize;
				result += maskSize;
			}

			inputPointer += currSize;
			outputPointer += currSize;
		}
	}
	else {
		CIntArray<maskAlign> generated;
		CCpuRandom random(desc.seed);
		int currSize;

		const int inputObjectSize = input.ObjectSize();
		const float* inputPointer = GetRaw(inputData);
		float* outputPointer = GetRaw(outputData);
		float* mask = GetRaw(desc.Mask.GetHandle());
		const int cacheSize = desc.Mask.Size() - desc.Mask.Size() % objectSize;

		//for (int i = 0; i < batchWidth; ++i) {
		//	const int index = i * objectSize;
		//	CCpuRandom random(desc.seed);
		//	//CFloatHandleStackVar maskVar(mathEngine(), objectSize + index % maskAlign);

		//	float* mask = GetRaw(desc.Mask.GetHandle());
		//	FillCpuDropoutMask(random, mask, desc, index - index % maskAlign, objectSize + index % maskAlign);
		//	mask += (index % maskAlign);

		//	const float* first = GetRaw(currInput);
		//	float* result = GetRaw(currOutput);

		//	for(int j = 0; j < batchLength; ++j) {
		//		multiplyMatrixByDiagMatrix(first, input.ObjectSize() / objectSize, objectSize, mask, result);

		//		first += inputObjectSize * batchWidth;
		//		result += inputObjectSize * batchWidth;
		//	}

		//	currInput += inputObjectSize;
		//	currOutput += inputObjectSize;
		//}

		for (int i = 0; i < (batchWidth * objectSize + desc.Mask.Size() - 1) / desc.Mask.Size(); ++i) {
			currSize = std::min(desc.Mask.Size(), batchWidth * objectSize - i * desc.Mask.Size());
			int idx = 0;

			const int alignedSize = (currSize + (maskAlign - 1)) / maskAlign;
			for (int i = 0; i < alignedSize; ++i) {
				generated = random.Next();
				for (int j = 0; j < maskAlign && idx < currSize; ++j) {
					mask[idx++] = (generated[j] <= desc.threshold) ? desc.value : 0.f;
				}
			}

			const float* first = inputPointer;
			float* result = outputPointer;

			for (int j = 0; j < batchLength; ++j) {
				multiplyMatrixByDiagMatrix(first, input.ObjectSize() / objectSize, objectSize, mask, result);

				first += inputObjectSize * batchWidth;
				result += inputObjectSize * batchWidth;
			}

			inputPointer += inputObjectSize;
			outputPointer += inputObjectSize;
		}
	}
}

} // namespace NeoML
