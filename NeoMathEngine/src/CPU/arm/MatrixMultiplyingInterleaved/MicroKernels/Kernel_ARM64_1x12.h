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

// The micro-kernel size is 1*12. The temporary variables are stored in q7 registers
// The B input matrix is stored row-by-row
// When iterating by K, the values from B are written into q1-q4
// When iterating by K, four values at once are written into q0
// The main cycle processes 4 rows at a time
// There are also blocks that process 2 rows and 1 row (combine to process three)
//             <---q1----> <---q2----> <---q3---->
//             <---q4----> <---q1----> <---q2---->
//             <---q3----> <---q4----> <---q1----> 
//             <---q2----> <---q3----> <---q4---->
//
// <---q0----> <---q7----> <---q8----> <---q9---->
struct CMicroKernel1x12 : public CMicroKernelBase<1, 12> {
	static void Calculate(const float* aPtr, const float* bPtr, float* cPtr, size_t /*cRowSize*/, size_t k)
	{
		__asm __volatile(
			"ldr    q7,     [%[cPtr], #0x0]\n"
			"ldr	q8,     [%[cPtr], #0x10]\n"
			"ldr	q9,     [%[cPtr], #0x20]\n"

			"subs	%[k],   %[k], #0x3\n"
			"bls 2f\n"
			// The main cycle
		"1:\n"
			"ldr    q0,     [%[aPtr]], #0x10\n"

			"ldr    q1,     [%[bPtr], #0x0]\n"
			"ldr    q2,     [%[bPtr], #0x10]\n"
			"ldr    q3,     [%[bPtr], #0x20]\n"
			"ldr    q4,     [%[bPtr], #0x30]\n"

			"fmla   v7.4s,  v1.4s, v0.s[0]\n"
			"ldr    q1,     [%[bPtr], #0x40]\n"
			"fmla   v8.4s,  v2.4s, v0.s[0]\n"
			"ldr    q2,     [%[bPtr], #0x50]\n"
			"fmla   v9.4s,  v3.4s, v0.s[0]\n"
			"ldr    q3,     [%[bPtr], #0x60]\n"

			"fmla   v7.4s,  v4.4s, v0.s[1]\n"
			"ldr    q4,     [%[bPtr], #0x70]\n"
			"fmla   v8.4s,  v1.4s, v0.s[1]\n"
			"ldr    q1,     [%[bPtr], #0x80]\n"
			"fmla   v9.4s,  v2.4s, v0.s[1]\n"
			"ldr    q2,     [%[bPtr], #0x90]\n"

			"fmla   v7.4s,  v3.4s, v0.s[2]\n"
			"ldr    q3,     [%[bPtr], #0xA0]\n"
			"fmla   v8.4s,  v4.4s, v0.s[2]\n"
			"ldr    q4,     [%[bPtr], #0xB0]\n"
			"fmla   v9.4s,  v1.4s, v0.s[2]\n"

			"fmla   v7.4s,  v2.4s, v0.s[3]\n"
			"fmla   v8.4s,  v3.4s, v0.s[3]\n"
			"fmla   v9.4s,  v4.4s, v0.s[3]\n"

			"add    %[bPtr],%[bPtr], #0xC0\n"
			"subs	%[k],   %[k], #0x4\n"
			"bhi 1b\n"
		"2:\n"

			"adds   %[k],   %[k], #0x3\n"
			"beq 4f\n"

			"subs	%[k],   %[k], #0x1\n"
			"beq 3f\n"

			// Process 2 rows (over K)
			"ldr    d0,     [%[aPtr]], #0x08\n"

			"ldr    q1,     [%[bPtr], #0x0]\n"
			"ldr    q2,     [%[bPtr], #0x10]\n"
			"ldr    q3,     [%[bPtr], #0x20]\n"
			"ldr    q4,     [%[bPtr], #0x30]\n"

			"fmla   v7.4s,  v1.4s, v0.s[0]\n"
			"ldr    q1,     [%[bPtr], #0x40]\n"
			"fmla   v8.4s,  v2.4s, v0.s[0]\n"
			"ldr    q2,     [%[bPtr], #0x50]\n"
			"fmla   v9.4s,  v3.4s, v0.s[0]\n"

			"fmla   v7.4s,  v4.4s, v0.s[1]\n"
			"fmla   v8.4s,  v1.4s, v0.s[1]\n"
			"fmla   v9.4s,  v2.4s, v0.s[1]\n"

			"add    %[bPtr],%[bPtr], #0x60\n"
			"subs	%[k],   %[k], #0x1\n"
			"beq 4f\n"

			// Process one row (over K)
		"3:\n"
			"ldr    s0,     [%[aPtr], #0x0]\n"

			"ldr    q1,     [%[bPtr], #0x0]\n"
			"ldr    q2,     [%[bPtr], #0x10]\n"
			"ldr    q3,     [%[bPtr], #0x20]\n"

			"fmla   v7.4s,  v1.4s, v0.s[0]\n"
			"fmla   v8.4s,  v2.4s, v0.s[0]\n"
			"fmla   v9.4s,  v3.4s, v0.s[0]\n"

			// Save C
		"4:\n"
			"str	q7,     [%[cPtr], #0x0]\n"
			"str	q8,     [%[cPtr], #0x10]\n"
			"str	q9,     [%[cPtr], #0x20]\n"
		: [aPtr]"+r"(aPtr), [bPtr]"+r"(bPtr), [cPtr]"+r"(cPtr), [k]"+r"(k)
		:
		: "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "cc"
		);
	}
};