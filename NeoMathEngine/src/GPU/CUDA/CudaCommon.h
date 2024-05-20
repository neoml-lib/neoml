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

#pragma once

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_CUDA

#include <NeoMathEngine/CrtAllocatedObject.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

namespace NeoML {

struct CCudaVectorArray final {
	static const int MaxSize = 16;
	float* Vectors[MaxSize]{};
	int VectorCount = 0;
};

struct CCudaConstVectorArray final {
	static const int MaxSize = 16;
	const float* Vectors[MaxSize]{};
	int VectorCount = 0;
};

//------------------------------------------------------------------------------------------------------------

// define for logarithms FLT_MIN/MAX. define used to avoid problems with CUDA
#define FLT_MIN_LOG -87.33654474f
#define FLT_MAX_LOG 88.f

// The exponent with limitations to avoid NaN
inline __device__ float ExponentFunc(float f)
{
	if(f < FLT_MIN_LOG) {
		return 0;
	} else if(f > FLT_MAX_LOG) {
		return FLT_MAX;
	} else {
		return expf(f);
	}
}

// LogSumExp for two numbers
inline __device__ float LogSumExpFunc(float f, float s)
{
	if(f >= s) {
		return f + log1pf( ExponentFunc( s - f ) );
	} else {
		return s + log1pf( ExponentFunc( f - s ) );
	}
}

//------------------------------------------------------------------------------------------------------------

// RLE image
struct CCudaRleStroke final {
	short Start = 0; // stroke start
	short End = 0;   // stroke end (first position after it ends)
};

struct CCudaRleImage final {
	int StrokesCount = 0;
	int Height = 0;
	int Width = 0;
	CCudaRleStroke Stub{};
	CCudaRleStroke Lines[1]{};
};

//------------------------------------------------------------------------------------------------------------

// Setting device
void SetCudaDevice( int deviceNum );

} // namespace NeoML

#endif // NEOML_USE_CUDA
