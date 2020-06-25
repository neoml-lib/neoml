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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_CUDA

#include <NeoMathEngine/CrtAllocatedObject.h>
#include <cuda_runtime.h>
#include <cfloat>

namespace NeoML {

// Constants
class CCudaConst : public CCrtAllocatedObject {
public:
	static const float* Zero;
	static const float* One;
};

//------------------------------------------------------------------------------------------------------------

struct CCudaVectorArray {
	static const int MaxSize = 16;
	float* Vectors[MaxSize];
	int VectorCount;
};

struct CCudaConstVectorArray {
	static const int MaxSize = 16;
	const float* Vectors[MaxSize];
	int VectorCount;
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
		return f + log1pf(expf(s - f));
	} else {
		return s + log1pf(expf(f - s));
	}
}

//------------------------------------------------------------------------------------------------------------

// RLE image
struct CCudaRleStroke {
	short Start;	// stroke start
	short End;		// stroke end (first position after it ends)
};

struct CCudaRleImage {
	int StrokesCount;
	int Height;
	int Width;
	CCudaRleStroke Stub;
	CCudaRleStroke Lines[1];
};

} // namespace NeoML

#endif // NEOML_USE_CUDA
