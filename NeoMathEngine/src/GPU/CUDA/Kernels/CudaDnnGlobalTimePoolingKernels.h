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

#pragma once

#include <CudaMathEngineDnnPoolings.h>
#include <Kernels/CudaGrid.h>
#include <Kernels/CudaReduce.h>
#include <cfloat>

namespace NeoML {

__global__ void BlobGlobalMaxOverTimePoolingWithIndexKernel( const CCudaGlobalMaxOverTimePoolingDescInternal desc,
	const float* sourceData, int* maxIndicesData, float* resultData )
{
	const CCudaBlobDesc& source = desc.Source;

	int objectCount = source.BatchLength();
	int objectSize = source.BlobSize() / objectCount;

	int objectNum;
	if(!GetCudaTaskIndex(objectSize, objectNum)) {
		return;
	}

	int curIndex = objectNum;
	int maxIndex = 0;
	float maxVal = __ldg(sourceData + curIndex);

	for(int i = 1; i < objectCount; ++i) {
		curIndex += objectSize;
		float candidate = __ldg(sourceData + curIndex);
		if(candidate > maxVal) {
			maxVal = candidate;
			maxIndex = i;
		}
	}

	resultData[objectNum] = maxVal;
	maxIndicesData[objectNum] = maxIndex;
}

__global__ void BlobGlobalMaxOverTimePoolingKernel( const CCudaGlobalMaxOverTimePoolingDescInternal desc,
	const float* sourceData, float* resultData )
{
	const CCudaBlobDesc& source = desc.Source;

	int objectCount = source.BatchLength();
	int objectSize = source.BlobSize() / objectCount;

	int objectNum;
	if(!GetCudaTaskIndex(objectSize, objectNum)) {
		return;
	}

	int curIndex = objectNum;
	float maxVal = -FLT_MAX;

	for(int i = 0; i < objectCount; ++i) {
		float candidate = __ldg(sourceData + curIndex);
		if(candidate > maxVal) {
			maxVal = candidate;
		}
		curIndex += objectSize;
	}

	resultData[objectNum] = maxVal;
}

__global__ void BlobGlobalMaxOverTimePoolingBackwardKernel( const CCudaGlobalMaxOverTimePoolingDescInternal desc,
	const float* __restrict__ sourceData, const int* __restrict__ maxIndicesData, float* resultData )
{
	const CCudaBlobDesc& outputDiff = desc.Result;
	int pos;
	if(!GetCudaTaskIndex(outputDiff.BlobSize(), pos)) {
		return;
	}

	resultData[__ldg(maxIndicesData + pos) * outputDiff.BlobSize() + pos] = __ldg(sourceData + pos);
}

} // namespace NeoML
