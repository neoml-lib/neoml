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

// The micro-kernel size is 4*1. The temporary variables are stored in q3 and q4 registers
// The A input matrix is stored column-by-column
// When iterating by K, the values from B are written into q5 four at once
// When iterating by K, each column from A is written into q0 and q1
// The main cycle processes 4 rows at a time
// There are also blocks that process 2 rows and 1 row (combine to process three)
//      ^ 
//      q0
//      | 
//      v 
//
// ^    ^ 
// q1   q3
// |    | 
// v    v 
struct CMicroKernel4x1 : public CMicroKernelBase<4, 1> {
	static void Calculate(const float* aPtr, const float* bPtr, float* cPtr, size_t ldc, size_t k)
	{
		// Passing through the C matrix twice: first load, next save
		// Two pointers are used because the pointer changes after the pass
		float* cLdr = cPtr;
		ldc *= sizeof(float);
		__asm __volatile(
			// Load the C matrix
			// Loading values one by one, 
			// after which they are moved to one vector register
			"ldr	s3,    [%[cLdr], #0x0]\n"
			"add    %[cLdr],%[cLdr], %[ldc]\n"
			"ldr	s2,    [%[cLdr], #0x0]\n"
			"add    %[cLdr],%[cLdr], %[ldc]\n"
			"ldr	s1,    [%[cLdr], #0x0]\n"
			"add    %[cLdr],%[cLdr], %[ldc]\n"
			"ldr	s0,    [%[cLdr], #0x0]\n"
			"add    %[cLdr],%[cLdr], %[ldc]\n"
			"ins    v3.s[1], v2.s[0]\n"
			"ins    v3.s[2], v1.s[0]\n"
			"ins    v3.s[3], v0.s[0]\n"

			"subs	%[k],   %[k], #0x3\n"
			// Skip the main cycle for small values of K
			"bls 2f\n"
			// Main cycle over K (4 at a time)
		"1:\n"
			"ldr    q0,     [%[bPtr], #0x0]\n"
			"ldr    q1,     [%[aPtr], #0x0]\n"

			"fmla   v3.4s,  v1.4s, v0.s[0]\n"
			"ldr    q1,     [%[aPtr], #0x10]\n"

			"fmla   v3.4s,  v1.4s, v0.s[1]\n"
			"ldr    q1,     [%[aPtr], #0x20]\n"

			"fmla   v3.4s,  v1.4s, v0.s[2]\n"
			"ldr    q1,     [%[aPtr], #0x30]\n"

			"fmla   v3.4s,  v1.4s, v0.s[3]\n"

			"add    %[bPtr],%[bPtr], #0x10\n"
			"add    %[aPtr],%[aPtr], #0x40\n"

			"subs	%[k],   %[k], #0x4\n"
			"bhi 1b\n"
		"2:\n"

			"adds   %[k],   %[k], #0x3\n"
			"beq 4f\n"

			"subs	%[k],   %[k], #0x1\n"
			"beq 3f\n"
			// Process two rows (over K)
			"ldr    d0,     [%[bPtr], #0x0]\n"
			"ldr    q1,     [%[aPtr], #0x0]\n"

			"fmla   v3.4s,  v1.4s, v0.s[0]\n"
			"ldr    q1,     [%[aPtr], #0x10]\n"

			"fmla   v3.4s,  v1.4s, v0.s[1]\n"

			"subs	%[k],   %[k], #0x1\n"
			"beq 4f\n"

			"add    %[bPtr],%[bPtr], #0x08\n"
			"add    %[aPtr],%[aPtr], #0x20\n"
			// Process one row (over K)
		"3:\n"
			"ldr    s0,     [%[bPtr], #0x0]\n"
			"ldr    q1,     [%[aPtr], #0x0]\n"

			"fmla   v3.4s,  v1.4s, v0.s[0]\n"

			// Save the C matrix. The data from the vector register 
			// is split into several registers and then saved
		"4:\n"
			"dup    s2,      v3.s[1]\n"
			"dup    s1,      v3.s[2]\n"
			"dup    s0,      v3.s[3]\n"
			"str	s3,     [%[cPtr], #0x0]\n"
			"add    %[cPtr],%[cPtr], %[ldc]\n"
			"str	s2,     [%[cPtr], #0x0]\n"
			"add    %[cPtr],%[cPtr], %[ldc]\n"
			"str	s1,     [%[cPtr], #0x0]\n"
			"add    %[cPtr],%[cPtr], %[ldc]\n"
			"str	s0,     [%[cPtr], #0x0]\n"
			"add    %[cPtr],%[cPtr], %[ldc]\n"
		: [aPtr]"+r"(aPtr), [bPtr]"+r"(bPtr), [cPtr]"+r"(cPtr), [cLdr]"+r"(cLdr), [ldc]"+r"(ldc), [k]"+r"(k)
		:
		: "v0", "v1", "v3",
		  "cc", "memory"
		);
	}
};