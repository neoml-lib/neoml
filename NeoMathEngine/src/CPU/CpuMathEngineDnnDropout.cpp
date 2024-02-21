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

CDropoutDesc* CCpuMathEngine::InitDropout(float rate, bool isSpatial, bool isBatchwise)
{
	ASSERT_EXPR(rate >= 0.f && rate < 1.f);
	auto seedDesc = new CSeedDropoutDesc(mathEngine(), true);
	seedDesc->ForwardRate = 1.f - rate;
	seedDesc->IsSpatial = isSpatial;
	seedDesc->IsBatchwise = isBatchwise;
	seedDesc->isValid = false;
	seedDesc->value = 1.f / seedDesc->ForwardRate;
	seedDesc->threshold = (unsigned int)(seedDesc->ForwardRate * UINT_MAX);
	return seedDesc;
}

void CCpuMathEngine::UpdateDropout(CDropoutDesc* dropoutDesc, const CBlobDesc& input,
	const CBlobDesc& output, int seed, bool valid) 
{
	auto seedDesc = dynamic_cast<CSeedDropoutDesc*>(dropoutDesc);
	seedDesc->isValid = valid;;
	if (valid) {
		seedDesc->seed = seed;
		seedDesc->Input = input;
		seedDesc->Output = output;
	}
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
	CIntArray<CSeedDropoutDesc::maskAlign> generated;

	const int inputObjectSize = input.ObjectSize();
	const float* inputPointer = GetRaw(inputData);
	float* outputPointer = GetRaw(outputData);
	float* mask = GetRaw(desc.Mask->GetHandle());
	const unsigned int threshold = desc.threshold;
	const float value = desc.value;
	constexpr int cacheSize = CSeedDropoutDesc::cacheSize;
	constexpr int maskAlign = CSeedDropoutDesc::maskAlign;
	constexpr int numOfGenerations = CSeedDropoutDesc::numOfGenerations;

	if(!desc.IsSpatial) {
		const int numOfIter = (maskSize + cacheSize - 1) / cacheSize;
		const int currSize = maskSize - (numOfIter - 1) * cacheSize;
		const int lastGenerations = (currSize + (maskAlign - 1)) / maskAlign;

		for(int i = 0; i < numOfIter - 1; ++i) {
			const float* first = inputPointer;
			float* result = outputPointer;

			int idx = 0;
			for(int i = 0; i < numOfGenerations; ++i) {
				generated = random.Next();
				for (int j = 0; j < maskAlign && idx < cacheSize; ++j) {
					mask[idx++] = (generated[j] <= threshold) ? value : 0.f;
				}
			}

			for(int b = 0; b < batchLength; ++b) {
				vectorEltwiseMultiply(first, mask, result, cacheSize);

				first += maskSize;
				result += maskSize;
			}

			inputPointer += cacheSize;
			outputPointer += cacheSize;
		}

		// last generation
		const float* first = inputPointer;
		float* result = outputPointer;

		int idx = 0;
		for (int i = 0; i < lastGenerations; ++i) {
			generated = random.Next();
			for (int j = 0; j < maskAlign && idx < currSize; ++j) {
				mask[idx++] = (generated[j] <= threshold) ? value : 0.f;
			}
		}

		for (int b = 0; b < batchLength; ++b) {
			vectorEltwiseMultiply(first, mask, result, currSize);

			first += maskSize;
			result += maskSize;
		}
	} else {
		const int numOfIter = (objectSize + cacheSize - 1) / cacheSize;
		const int currSize = objectSize - (numOfIter - 1) * cacheSize;
		const int lastGenerations = (currSize + (maskAlign - 1)) / maskAlign;
		const int channelIter = (inputObjectSize / objectSize);

		for( int i = 0; i < batchWidth; ++i ) {
			const float* first = inputPointer;
			float* result = outputPointer;

			for( int j = 0; j < numOfIter - 1; ++j ) {
				int idx = 0;
				for( int g = 0; g < numOfGenerations; ++g ) {
					generated = random.Next();
					for (int k = 0; k < maskAlign && idx < cacheSize; ++k) {
						mask[idx++] = (generated[k] <= threshold) ? value : 0.f;
					}
				}

				first = inputPointer + j * cacheSize;
				result = outputPointer + j * cacheSize;
				for( int b = 0; b < batchLength; ++b ) {
					const float* localFirst = first;
					float* localResult = result;
					for( int k = 0; k < channelIter; ++k ) {
						vectorEltwiseMultiply(localFirst, mask, localResult, cacheSize);
						localFirst += objectSize;
						localResult += objectSize;
					}
					first += batchWidth * inputObjectSize;
					result += batchWidth * inputObjectSize;
				}
			}

			// last generation
			int idx = 0;
			for (int g = 0; g < lastGenerations; ++g) {
				generated = random.Next();
				for (int k = 0; k < desc.maskAlign && idx < currSize; ++k) {
					mask[idx++] = (generated[k] <= threshold) ? value : 0.f;
				}
			}

			first = inputPointer + (numOfIter - 1) * desc.cacheSize;
			result = outputPointer + (numOfIter - 1) * desc.cacheSize;
			for (int b = 0; b < batchLength; ++b) {
				const float* localFirst = first;
				float* localResult = result;
				for (int k = 0; k < channelIter; ++k) {
					vectorEltwiseMultiply(localFirst, mask, localResult, currSize);
					localFirst += objectSize;
					localResult += objectSize;
				}
				first += batchWidth * inputObjectSize;
				result += batchWidth * inputObjectSize;
			}

			inputPointer += inputObjectSize;
			outputPointer += inputObjectSize;
		}
	}
}

} // namespace NeoML
