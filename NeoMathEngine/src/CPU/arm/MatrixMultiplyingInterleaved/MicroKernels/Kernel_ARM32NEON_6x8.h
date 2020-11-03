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

#include "MatrixMultiplyingInterleavedCommon/MicroKernels/MicroKernelBase.h"

struct CMicroKernel6x8 : public CMicroKernelBase<6, 8> {
	static void Calculate(const float* aPtr, const float* bPtr, float* cPtr, size_t ldc, size_t k)
	{
		constexpr size_t aRowSize = height * sizeof(float);
		constexpr size_t bRowSize = width * sizeof(float);
		size_t cRowSize = ldc * sizeof(float);
		float* cTmp = cPtr;
		__asm __volatile(
			"vld1.32 {q4,q5},	[%[cTmp]]\n"
			"add     %[cTmp],    %[cTmp],    %[cRowSize]\n"
			"vld1.32 {q6,q7},    [%[cTmp]]\n"
			"add     %[cTmp],    %[cTmp],    %[cRowSize]\n"
			"vld1.32 {q8,q9},    [%[cTmp]]\n"
			"add     %[cTmp],    %[cTmp],    %[cRowSize]\n"
			"vld1.32 {q10,q11},  [%[cTmp]]\n"
			"add     %[cTmp],    %[cTmp],    %[cRowSize]\n"
			"vld1.32 {q12,q13},  [%[cTmp]]\n"
			"add     %[cTmp],    %[cTmp],    %[cRowSize]\n"
			"vld1.32 {q14,q15},  [%[cTmp]]\n"
			"add     %[cTmp],    %[cTmp],	 %[cRowSize]\n"

		"1:\n"
			"vld1.32 {d4,d5,d6}, [%[aPtr]]\n"
			"vld1.32 {q0,q1},    [%[bPtr]]\n"

			"add     %[aPtr],   %[aPtr],    %[aRowSize]\n"
			"add     %[bPtr],   %[bPtr],    %[bRowSize]\n"

			"vmla.f32    q4,    q0,  d4[0]\n"
			"vmla.f32    q5,    q1,  d4[0]\n"

			"vmla.f32    q6,    q0,  d4[1]\n"
			"vmla.f32    q7,    q1,  d4[1]\n"

			"vmla.f32    q8,    q0,  d5[0]\n"
			"vmla.f32    q9,    q1,  d5[0]\n"

			"vmla.f32    q10,   q0,  d5[1]\n"
			"vmla.f32    q11,   q1,  d5[1]\n"

			"vmla.f32    q12,   q0,  d6[0]\n"
			"vmla.f32    q13,   q1,  d6[0]\n"

			"vmla.f32    q14,   q0,  d6[1]\n"
			"vmla.f32    q15,   q1,  d6[1]\n"

			"subs    %[k],   %[k],   #1\n"
			"bne     1b\n"

			"vst1.32 {q4,q5},    [%[cPtr]]\n"
			"add     %[cPtr],    %[cPtr],    %[cRowSize]\n"
			"vst1.32 {q6,q7},    [%[cPtr]]\n"
			"add     %[cPtr],    %[cPtr],    %[cRowSize]\n"
			"vst1.32 {q8,q9},    [%[cPtr]]\n"
			"add     %[cPtr],    %[cPtr],    %[cRowSize]\n"
			"vst1.32 {q10,q11},  [%[cPtr]]\n"
			"add     %[cPtr],    %[cPtr],    %[cRowSize]\n"
			"vst1.32 {q12,q13},  [%[cPtr]]\n"
			"add     %[cPtr],    %[cPtr],    %[cRowSize]\n"
			"vst1.32 {q14,q15},  [%[cPtr]]\n"
			"add     %[cPtr],    %[cPtr],	%[cRowSize]\n"
		: [aPtr] "+r" (aPtr),
		  [bPtr] "+r" (bPtr),
		  [cPtr] "+r" (cPtr), [cRowSize] "+r" (cRowSize),
		  [cTmp] "+r" (cTmp), [k] "+r" (k)
		: [aRowSize] "i" (aRowSize), [bRowSize] "i" (bRowSize)
		: "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
		  "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15",
		  "cc", "memory"
		);
	}
};