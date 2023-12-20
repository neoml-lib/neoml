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

#include <Kernels/CudaGrid.h>
#include <Kernels/CudaRandom.h>

namespace NeoML {

__global__ void RandomMatrixDropout( const float* __restrict__ first, int firstHeight,
	int firstWidth, float* res, int seed, float forwardRate )
{
	const unsigned int threshold = forwardRate * UINT_MAX;
	int row;
	int col;
	if( GetCudaTaskIndex2D( firstHeight, ( firstWidth + 3 ) / 4, row, col ) ) {
		CCudaRandom random(seed);
		random.Skip(col);
		col *= 4;
		const int index = row * firstWidth + col;

		CIntArray<4> generated = random.Next();
		for(int j = 0; j < 4 && col + j < firstWidth; ++j) {
			res[index + j] = (generated[j] <= threshold) ? (first[index + j] / forwardRate) : 0.f;
		}
	}
}

__global__ void RandomSpatialDropout( const float* __restrict__ input, float* res, int inputObjectCount,
	int inputObjectSize, int maskObjectCount, int maskObjectSize, int seed, float forwardRate )
{
	const unsigned int threshold = forwardRate * UINT_MAX;
	int obj;
	int row;
	int col;
	if( GetCudaTaskIndex3D( inputObjectCount, inputObjectSize / maskObjectSize, maskObjectSize, obj, row, col ) ) {
		int pack = obj % maskObjectCount;
		int index = obj * inputObjectSize + row * maskObjectSize + col;
		int numBlock = ( pack * maskObjectSize + col ) / 4;
		int numLeft = ( pack * maskObjectSize + col ) % 4;
		CCudaRandom random(seed);
		random.Skip(numBlock);

		CIntArray<4> generated = random.Next();
		res[index] = (generated[numLeft] <= threshold) ? (input[index] / forwardRate) : 0.f;
	}
}

} // namespace NeoML
