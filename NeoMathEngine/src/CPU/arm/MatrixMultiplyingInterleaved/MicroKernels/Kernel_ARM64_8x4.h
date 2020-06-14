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

// The micro-kernel size is 8*4. The temporary variables are stored in q8-q15 registers
// The A input matrix is stored column-by-column
// The B input matrix is stored row-by-row
// When iterating by K, each row from B is written into q2 and q3 in turns
// When iterating by K, each column from A is written into q0-q1 and q5-q6 pairs in turns
// The main cycle processes 2 rows at a time. Before the main cycle, load the registers; 
// at the end of the cycle load the registers for the next iteration, process the remainder over K for odd and even values
//         <---q2---->
//         <---q3---->
//
// ^  ^    <---q8---->
// q0 q5   <---q9---->
// |  |    <---q10--->
// v  v    <---q11--->
// ^  ^    <---q12--->
// q1 q6   <---q13--->
// |  |    <---q14--->
// v  v    <---q15--->
struct CMicroKernel8x4 : public CMicroKernelBase<8, 4> {
	static void Calculate(const float* aPtr, const float* bPtr, float* cPtr, size_t ldc, size_t k)
	{
		// Passing through the C matrix twice: first load, next save
		// Two pointers are used because the pointer changes after the pass
		float* cLdr = cPtr;
		ldc *= sizeof(float);
		__asm __volatile(
			// Load the C matrix into the registers, also two registers from A and B each
			// Pre-fetch A and B matrices
			"ldr    q0,     [%[aPtr], #0x0]\n"
			"ldr    q2,     [%[bPtr], #0x0]\n"
			"prfm pldl1keep,[%[bPtr], #0x40]\n"

			"ldr    q1,     [%[aPtr], #0x10]\n"
			"prfm pldl1keep,[%[aPtr], #0x40]\n"

			"ldr    q8,     [%[cLdr], #0x0]\n"
			"add    %[cLdr],%[cLdr], %[ldc]\n"
			"prfm pldl1keep,[%[bPtr], #0x80]\n"
			"ldr    q9,     [%[cLdr], #0x0]\n"
			"add    %[cLdr],%[cLdr], %[ldc]\n"
			"prfm pldl1keep,[%[aPtr], #0x80]\n"
			"ldr    q10,    [%[cLdr], #0x0]\n"
			"add    %[cLdr],%[cLdr], %[ldc]\n"
			"ldr    q11,    [%[cLdr], #0x0]\n"
			"add    %[cLdr],%[cLdr], %[ldc]\n"
			"ldr    q12,    [%[cLdr], #0x0]\n"
			"add    %[cLdr],%[cLdr], %[ldc]\n"
			"prfm pldl1keep,[%[aPtr], #0xC0]\n"
			"ldr    q13,    [%[cLdr], #0x0]\n"
			"ldr    q13,    [%[cLdr], #0x0]\n"
			"add    %[cLdr],%[cLdr], %[ldc]\n"
			"ldr    q14,    [%[cLdr], #0x0]\n"
			"add    %[cLdr],%[cLdr], %[ldc]\n"
			"prfm pldl1keep,[%[aPtr], #0x100]\n"
			"ldr    q15,    [%[cLdr], #0x0]\n"
			// Skip the main cycle for small values of K
			"subs   %[k],    %[k], #0x2\n"
			"bls 2f\n"

			// The main cycle over K
		"1:\n"
			"ldr    q3,     [%[bPtr], #0x10]\n"

			"fmla   v8.4s,  v2.4s, v0.s[0]\n"
			"fmla   v9.4s,  v2.4s, v0.s[1]\n"
			"fmla   v10.4s, v2.4s, v0.s[2]\n"
			"fmla   v11.4s, v2.4s, v0.s[3]\n"
			"ldr    q5,	    [%[aPtr], #0x20]\n"
			"fmla   v12.4s, v2.4s, v1.s[0]\n"
			"fmla   v13.4s, v2.4s, v1.s[1]\n"
			"ldr    q6,     [%[aPtr], #0x30]\n"
			"fmla   v14.4s, v2.4s, v1.s[2]\n"
			"fmla   v15.4s, v2.4s, v1.s[3]\n"
			"ldr    q2,     [%[bPtr], #0x20]\n"

			"prfm pldl1keep,[%[aPtr], #0x140]\n"
			"prfm pldl1keep,[%[bPtr], #0xC0]\n"

			"fmla   v8.4s,  v3.4s, v5.s[0]\n"
			"fmla   v9.4s,  v3.4s, v5.s[1]\n"
			"ldr    q0,     [%[aPtr], #0x40]\n"
			"fmla   v10.4s, v3.4s, v5.s[2]\n"
			"fmla   v11.4s, v3.4s, v5.s[3]\n"
			"fmla   v12.4s, v3.4s, v6.s[0]\n"
			"ldr    q1,     [%[aPtr], #0x50]\n"
			"fmla   v13.4s, v3.4s, v6.s[1]\n"
			"fmla   v14.4s, v3.4s, v6.s[2]\n"
			"fmla   v15.4s, v3.4s, v6.s[3]\n"
			"add    %[bPtr],%[bPtr], #0x20\n"
			"add    %[aPtr],%[aPtr], #0x40\n"
			"subs   %[k],    %[k], #0x2\n"
			"bhi    1b\n"

			// Main cycle ends
		"2:\n"

			// For odd values of K
			"bne 3f\n"

			// Process the remainder for even values of K
			"fmla   v8.4s,  v2.4s, v0.s[0]\n"
			"fmla   v9.4s,  v2.4s, v0.s[1]\n"
			"fmla   v10.4s, v2.4s, v0.s[2]\n"
			"fmla   v11.4s, v2.4s, v0.s[3]\n"
			"fmla   v12.4s, v2.4s, v1.s[0]\n"
			"fmla   v13.4s, v2.4s, v1.s[1]\n"
			"fmla   v14.4s, v2.4s, v1.s[2]\n"
			"fmla   v15.4s, v2.4s, v1.s[3]\n"

			"ldr    q2,     [%[bPtr], #0x10]\n"
			"ldr    q0,     [%[aPtr], #0x20]\n"
			"ldr    q1,     [%[aPtr], #0x30]\n"
			"add    %[aPtr],%[aPtr], #0x20\n"

			// Process the remainder for odd values of K
		"3:\n"
			"fmla   v8.4s,  v2.4s, v0.s[0]\n"
			"fmla   v9.4s,  v2.4s, v0.s[1]\n"
			"str    q8,     [%[cPtr], #0x0]\n"
			"add    %[cPtr],%[cPtr], %[ldc]\n"
			"str    q9,     [%[cPtr], #0x0]\n"

			"fmla   v10.4s, v2.4s, v0.s[2]\n"
			"add    %[cPtr],%[cPtr], %[ldc]\n"
			"str    q10,    [%[cPtr], #0x0]\n"

			"fmla   v11.4s, v2.4s, v0.s[3]\n"
			"add    %[cPtr],%[cPtr], %[ldc]\n"
			"str    q11,    [%[cPtr], #0x0]\n"

			"fmla   v12.4s, v2.4s, v1.s[0]\n"
			"add    %[cPtr],%[cPtr], %[ldc]\n"
			"str    q12,    [%[cPtr], #0x0]\n"

			"fmla   v13.4s, v2.4s, v1.s[1]\n"
			"add    %[cPtr],%[cPtr], %[ldc]\n"
			"str    q13,    [%[cPtr], #0x0]\n"

			"fmla   v14.4s, v2.4s, v1.s[2]\n"
			"add    %[cPtr],%[cPtr], %[ldc]\n"
			"str    q14,    [%[cPtr], #0x0]\n"

			"fmla   v15.4s, v2.4s, v1.s[3]\n"
			"add    %[cPtr],%[cPtr], %[ldc]\n"
			"str    q15,    [%[cPtr], #0x0]\n"
		: [aPtr]"+r"(aPtr), [bPtr]"+r"(bPtr), [cPtr]"+r"(cPtr), [cLdr]"+r"(cLdr),
		  [ldc]"+r"(ldc), [k]"+r"(k)
		:
		: "v0", "v1", "v2", "v3", "v5", "v6",
		  "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
		  "cc", "memory"
		);
	}
};
