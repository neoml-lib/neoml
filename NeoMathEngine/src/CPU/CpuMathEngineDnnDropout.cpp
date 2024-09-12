/* Copyright © 2017-2024 ABBYY

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

CDropoutDesc* CCpuMathEngine::InitDropout(float rate, bool isSpatial, bool isBatchwise,
	const CBlobDesc& input, const CBlobDesc& output, int seed)
{
	return new CSeedDropoutDesc(rate, isSpatial, isBatchwise, input, output, seed);;
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

	CCpuRandom random(desc.Seed);
	CCpuRandom::CCounter generated{};

	const int inputObjectSize = input.ObjectSize();
	const unsigned int threshold = desc.Threshold;
	const float value = desc.Value;
	constexpr int cacheSize = CSeedDropoutDesc::CacheSize;
	constexpr int maskAlign = CSeedDropoutDesc::MaskAlign;

	const float* inputPointer = GetRaw(inputData);
	float* outputPointer = GetRaw(outputData);
	float mask[cacheSize];

	const int unitSize = desc.IsSpatial ? objectSize : maskSize;
	const int numOfIter = (unitSize + cacheSize - 1) / cacheSize;
	const int channelsIterations = desc.IsSpatial ? (inputObjectSize / objectSize) : 1;
	const int batchIterations = desc.IsSpatial ? batchWidth : 1;
	const int batchWiseSize = desc.IsSpatial ? inputObjectSize : 0;
	const int nextBatchStep = desc.IsSpatial ? batchWidth * inputObjectSize : unitSize;

	for (int i = 0; i < batchIterations; ++i) {
		for (int j = 0; j < numOfIter; ++j) {
			int currSize = std::min(cacheSize, unitSize - j * cacheSize);
			const int numOfGenerations = (currSize + (maskAlign - 1)) / maskAlign;
			int idx = 0;
			for (int g = 0; g < numOfGenerations; ++g) {
				random.Next( generated );
				for (int k = 0; k < maskAlign; ++k) {
					mask[idx++] = (generated.Data[k] <= threshold) ? value : 0.f;
				}
			}

			const float* first = inputPointer + j * cacheSize;
			float* result = outputPointer + j * cacheSize;
			for (int b = 0; b < batchLength; ++b) {
				const float* localFirst = first;
				float* localResult = result;
				for (int k = 0; k < channelsIterations; ++k) {
					vectorEltwiseMultiply(localFirst, mask, localResult, currSize);

					localFirst += objectSize;
					localResult += objectSize;
				}
				first += nextBatchStep;
				result += nextBatchStep;
			}
		}

		inputPointer += batchWiseSize;
		outputPointer += batchWiseSize;
	}
}

} // namespace NeoML
