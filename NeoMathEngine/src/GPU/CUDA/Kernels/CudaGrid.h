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

namespace NeoML {

// Returns the number of operations in the kernel, the initial index and stride
inline __device__ int GetCudaTaskCountAndIndexX(int taskCount, int combineCount, int& index, int& step)
{
	index = blockIdx.x * combineCount * blockDim.x + threadIdx.x;
	step = blockDim.x;

	return min(combineCount, (taskCount - index + step - 1) / step);
}

inline __device__ int GetCudaTaskCountAndIndexY(int taskCount, int combineCount, int& index, int& step)
{
	index = blockIdx.y * combineCount * blockDim.y + threadIdx.y;
	step = blockDim.y;

	return min(combineCount, (taskCount - index + step - 1) / step);
}

inline __device__ int GetCudaTaskCountAndIndex(int taskCount, int combineCount, int& index, int& step)
{
	return GetCudaTaskCountAndIndexX(taskCount, combineCount, index, step);
}

// The variation for combineCount = 1.
// Returns false if the task does not need to be completed on the current thread
inline __device__ bool GetCudaTaskIndex(int taskCount, int& index)
{
	index = blockIdx.x * blockDim.x + threadIdx.x;

	return index < taskCount;
}

inline __device__ bool GetCudaTaskIndex2D(int height, int width, int& j, int& i)
{
	j = blockIdx.y * blockDim.y + threadIdx.y;
	i = blockIdx.x * blockDim.x + threadIdx.x;

	return j < height && i < width;
}

inline __device__ bool GetCudaTaskIndex3D(int batchSize, int height, int width, int &num, int& j, int& i)
{
	num = blockIdx.z * blockDim.z + threadIdx.z;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	i = blockIdx.x * blockDim.x + threadIdx.x;

	return num < batchSize && j < height && i < width;
}

} // namespace NeoML
