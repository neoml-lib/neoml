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

#include "MatrixMultiplyingInterleavedCommon/Interleavers/InterleaverBase.h"

template <bool Transpose, size_t Len>
struct CInterleaver : public CInterleaverBase<Transpose, Len> {};

template <>
struct CInterleaver<false, 12> : public CInterleaverBase<false, 12> {
public:
	static void Prepare(float* out, const float* in, size_t stride, size_t height, size_t width)
	{
		size_t ldout = height * 12;
		for( ; height > 3; height -= 4, out += 12 * 4 ) {
			size_t len = width;
			float* out0 = out;
			const float* in0 = in;
			const float* in1 = in0 + stride;
			const float* in2 = in1 + stride;
			const float* in3 = in2 + stride;
			in = in3 + stride;

			prefetch(in0);
			prefetch(in1);
			prefetch(in2);
			prefetch(in3);

			for( ; len >= 12; len -= 12 ) {
				move4(out0, in0, in1, in2, in3);
				out0 += ldout;
			}
			if( len > 0 ) {
				movepart(out0, in0, len);
				movepart(out0, in1, len);
				movepart(out0, in2, len);
				movepart(out0, in3, len);
			}
		}

		if( height >= 2 ) {
			float* out0 = out;
			out += 12 * 2;
			const float* in0 = in;
			const float* in1 = in0 + stride;
			in = in1 + stride;
			prefetch(in0);
			prefetch(in1);
			height -= 2;
			size_t len = width;
			for( ; len >= 12; len -= 12 ) {
				move2(out0, in0, in1);
				out0 += ldout;
			}
			if( len > 0 ) {
				movepart(out0, in0, len);
				movepart(out0, in1, len);
			}
		}

		if( height >= 1 ) {
			prefetch(in);
			size_t len = width;
			for( ; len >= 12; len -= 12 ) {
				move1(out, in);
				out += ldout;
			}
			if( len > 0 ) {
				movepart(out, in, len);
			}
		}
	}
private:
	static inline void prefetch(const float* ptr)
	{
		__asm __volatile(
			"prfm pldl1keep,[%[ptr], #0x00]\n"
			"prfm pldl1keep,[%[ptr], #0x40]\n"
			"prfm pldl1keep,[%[ptr], #0x80]\n"
		:
		: [ptr]"r"(ptr)
		: "memory"
		);
	}
	static inline void move1(float* out, const float*& in0)
	{
		__asm __volatile(
			"ldp    q0, q1, [%[in0]], #0x20\n"
			"stp    q0, q1, [%[out]]\n"
			"prfm pldl1keep,[%[in0], #0xC0]\n"
			"ldr    q2, [%[in0]], #0x10\n"
			"str    q2, [%[out], #0x20]\n"
		: [in0]"+r"(in0)
		: [out]"r"(out)
		: "v0", "v1", "v2", "memory"
		);
	}
	static inline void move2(float* out, const float*& in0, const float*& in1)
	{
		__asm __volatile(
			"ldp    q0, q1, [%[in0]], #0x20\n"
			"stp    q0, q1, [%[out]]\n"
			"prfm pldl1keep,[%[in0], #0xC0]\n"
			"ldr    q2, [%[in0]], #0x10\n"
			"ldp	q3, q4, [%[in1]], #0x20\n"
			"stp    q2, q3, [%[out], #0x20]\n"
			"prfm pldl1keep,[%[in1], #0xC0]\n"
			"ldr	q5, [%[in1]], #0x10\n"
			"stp    q4, q5, [%[out], #0x40]\n"
		: [in0]"+r"(in0), [in1]"+r"(in1)
		: [out]"r"(out)
		: "v0", "v1", "v2", "v3", "v4", "v5", "memory"
		);
	}
	static inline void move4(float* out, const float*& in0, const float*& in1, const float*& in2, const float*& in3)
	{
		__asm __volatile(
			"ldp    q0, q1, [%[in0]], #0x20\n"
			"stp    q0, q1, [%[out]]\n"
			"ldr    q2,     [%[in0]], #0x10\n"
			"prfm pldl1keep,[%[in0], #0xC0]\n"
			"ldp	q3, q4, [%[in1]], #0x20\n"
			"stp    q2, q3, [%[out], #0x20]\n"
			"ldr	q5,     [%[in1]], #0x10\n"
			"prfm pldl1keep,[%[in1], #0xC0]\n"
			"stp    q4, q5, [%[out], #0x40]\n"
			"ldp	q6, q7, [%[in2]], #0x20\n"
			"stp    q6, q7, [%[out], #0x60]\n"
			"ldr	q8,     [%[in2]], #0x10\n"
			"prfm pldl1keep,[%[in2], #0xC0]\n"
			"ldp	q9, q10,[%[in3]], #0x20\n"
			"stp    q8, q9, [%[out], #0x80]\n"
			"ldr	q11,    [%[in3]], #0x10\n"
			"stp    q10, q11,[%[out], #0xA0]\n"
			"prfm pldl1keep,[%[in3], #0xC0]\n"
		: [in0]"+r"(in0), [in1]"+r"(in1), [in2]"+r"(in2), [in3]"+r"(in3)
		: [out]"r"(out)
		: "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "memory"
		);
	}
	static inline void movepart(float*& out, const float*& in, size_t len)
	{
		size_t i = 0;
		for( ; i < len; ++i ) {
			*out++ = *in++;
		}
		for( ; i < 12; ++i ) {
			*out++ = 0.f;
		}
	}
};

// Transposing in 4*4 blocks 
// Each block is loaded into four registers
// Either ldr or ldp is used when loading two blocks
// For each block zip1 and zip2 are performed sequentially
// An illustration of the process:
// q0 :                 A0, A1, A2, A3
// q1 :                 B0, B1, B2, B3
// q2 :                 C0, C1, C2, C3
// q3 :                 D0, D1, D2, D3
//
// zip1 (q0, q2)->q4 :  A0, C0, A1, C1
// zip2 (q0, q2)->q5 :  A2, C2, A3, C3
// zip1 (q1, q3)->q6 :  B0, D0, B1, D1
// zip2 (q1, q3)->q7 :  B2, D2, B3, D3
//
// zip1 (q4, q6)->q8 :  A0, B0, C0, D0
// zip2 (q4, q6)->q9 :  A1, B1, C1, D1
// zip1 (q5, q7)->q10 : A2, B2, C2, D2
// zip2 (q5, q7)->q11 : A3, B3, C3, D3
//
// After this sequence of 8 commands the block is transposed
// The commands may be executed in another order or reuse registers,
// but the basic principle stays the same


template<>
struct CInterleaver<true, 12> : public CInterleaverBase<true, 12> {
	static void Prepare(float* out, const float* in, size_t stride, size_t width, size_t height)
	{
		if( height == 0 ) {
			return;
		}
		float zerobuff[16];
		memset(zerobuff, 0, sizeof(zerobuff));
		while( true ) {
			size_t len = width;
			const float* in0 = in;
			const float* in1 = in0 + stride;
			const float* in2 = in1 + stride;
			const float* in3 = in2 + stride;
			const float* in4 = in3 + stride;
			const float* in5 = in4 + stride;
			const float* in6 = in5 + stride;
			const float* in7 = in6 + stride;
			const float* in8 = in7 + stride;
			const float* in9 = in8 + stride;
			const float* in10 = in9 + stride;
			const float* in11 = in10 + stride;
			in = in11 + stride;
			__asm __volatile(
				"prfm pldl1keep,[%[in0], #0x0]\n"
				"prfm pldl1keep,[%[in0], #0x40]\n"
				"prfm pldl1keep,[%[in1], #0x0]\n"
				"prfm pldl1keep,[%[in1], #0x40]\n"
				"prfm pldl1keep,[%[in2], #0x0]\n"
				"prfm pldl1keep,[%[in2], #0x40]\n"
				"prfm pldl1keep,[%[in3], #0x0]\n"
				"prfm pldl1keep,[%[in3], #0x40]\n"
				"prfm pldl1keep,[%[in4], #0x0]\n"
				"prfm pldl1keep,[%[in4], #0x40]\n"
				"prfm pldl1keep,[%[in5], #0x0]\n"
				"prfm pldl1keep,[%[in5], #0x40]\n"
				"prfm pldl1keep,[%[in6], #0x0]\n"
				"prfm pldl1keep,[%[in6], #0x40]\n"
				"prfm pldl1keep,[%[in7], #0x0]\n"
				"prfm pldl1keep,[%[in7], #0x40]\n"
				"prfm pldl1keep,[%[in8], #0x0]\n"
				"prfm pldl1keep,[%[in8], #0x40]\n"
				"prfm pldl1keep,[%[in9], #0x0]\n"
				"prfm pldl1keep,[%[in9], #0x40]\n"
				"prfm pldl1keep,[%[in10], #0x0]\n"
				"prfm pldl1keep,[%[in10], #0x40]\n"
				"prfm pldl1keep,[%[in11], #0x0]\n"
				"prfm pldl1keep,[%[in11], #0x40]\n"
			:
			: [in0] "r"(in0), [in1] "r"(in1), [in2] "r"(in2), [in3] "r"(in3),
			  [in4] "r"(in4), [in5] "r"(in5), [in6] "r"(in6), [in7] "r"(in7),
			  [in8] "r"(in8), [in9] "r"(in9), [in10] "r"(in10), [in11] "r"(in11)
			: "memory"
			);
			for( ; len >= 8; len -= 8 ) {
				switch (height) {
				case 1:
					in1 = zerobuff;
					// nobreak
				case 2:
					in2 = zerobuff;
					// nobreak;
				case 3:
					in3 = zerobuff;
					// nobreak;
				case 4:
					in4 = zerobuff;
					// nobreak;
				case 5:
					in5 = zerobuff;
					// nobreak;
				case 6:
					in6 = zerobuff;
					// nobreak;
				case 7:
					in7 = zerobuff;
					// nobreak;
				case 8:
					in8 = zerobuff;
					// nobreak;
				case 9:
					in9 = zerobuff;
					// nobreak;
				case 10:
					in10 = zerobuff;
					// nobreak;
				case 11:
					in11 = zerobuff;
					// nobreak;
				default:
					break;
				}
				__asm __volatile(
					"ldp    q0, q1, [%[in0]], 0x20\n"
					"ldp    q2, q3, [%[in1]], 0x20\n"
					"ldp    q4, q5, [%[in2]], 0x20\n"
					"ldp    q6, q7, [%[in3]], 0x20\n"
					"prfm pldl1keep,[%[in0], #0x80]\n"

					// A0
					"zip1   v16.4s, v0.4s, v4.4s\n"
					"zip1   v17.4s, v2.4s, v6.4s\n"
					"zip2   v18.4s, v0.4s, v4.4s\n"
					"zip2   v19.4s, v2.4s, v6.4s\n"
					"prfm pldl1keep,[%[in1], #0x80]\n"

					"zip1   v20.4s, v16.4s,v17.4s\n"
					"zip2   v23.4s, v16.4s,v17.4s\n"
					"zip1   v26.4s, v18.4s,v19.4s\n"
					"zip2   v29.4s, v18.4s,v19.4s\n"
					"prfm pldl1keep,[%[in2], #0x80]\n"

					"ldp    q8, q9, [%[in4]], 0x20\n"
					"ldp    q10,q11,[%[in5]], 0x20\n"
					"ldp    q12,q13,[%[in6]], 0x20\n"
					"ldp    q14,q15,[%[in7]], 0x20\n"
					"prfm pldl1keep,[%[in3], #0x80]\n"

					// A1 / 2
					"zip1   v16.4s, v1.4s, v5.4s\n"
					"zip1   v17.4s, v3.4s, v7.4s\n"
					"zip2   v18.4s, v1.4s, v5.4s\n"
					"zip2   v19.4s, v3.4s, v7.4s\n"
					"prfm pldl1keep,[%[in4], #0x80]\n"

					"ldp    q0, q1, [%[in8]], 0x20\n"
					"ldp    q2, q3, [%[in9]], 0x20\n"
					"ldp    q4, q5, [%[in10]],0x20\n"
					"ldp    q6, q7, [%[in11]],0x20\n"
					"prfm pldl1keep,[%[in5], #0x80]\n"

					// B0
					"zip1   v22.4s, v8.4s, v12.4s\n"
					"zip1   v25.4s, v10.4s,v14.4s\n"
					"zip2   v28.4s, v8.4s, v12.4s\n"
					"zip2   v31.4s, v10.4s,v14.4s\n"
					"prfm pldl1keep,[%[in6], #0x80]\n"

					"zip1   v21.4s, v22.4s,v25.4s\n"
					"zip2   v24.4s, v22.4s,v25.4s\n"
					"zip1   v27.4s, v28.4s,v31.4s\n"
					"zip2   v30.4s, v28.4s,v31.4s\n"
					"prfm pldl1keep,[%[in7], #0x80]\n"

					// C0
					"zip1   v8.4s,  v0.4s, v4.4s\n"
					"zip1   v10.4s, v2.4s, v6.4s\n"
					"zip2   v12.4s, v0.4s, v4.4s\n"
					"zip2   v14.4s, v2.4s, v6.4s\n"
					"prfm pldl1keep,[%[in8], #0x80]\n"

					"zip1   v22.4s, v8.4s, v10.4s\n"
					"zip2   v25.4s, v8.4s, v10.4s\n"
					"zip1   v28.4s, v12.4s,v14.4s\n"
					"zip2   v31.4s, v12.4s,v14.4s\n"
					"prfm pldl1keep,[%[in9], #0x80]\n"

					"stp    q20,q21, [%[out]], 0x20\n"
					"stp    q22,q23, [%[out]], 0x20\n"
					"stp    q24,q25, [%[out]], 0x20\n"
					"stp    q26,q27, [%[out]], 0x20\n"
					"stp    q28,q29, [%[out]], 0x20\n"
					"stp    q30,q31, [%[out]], 0x20\n"

					// A1 end
					"zip1   v20.4s, v16.4s,v17.4s\n"
					"zip2   v23.4s, v16.4s,v17.4s\n"
					"zip1   v26.4s, v18.4s,v19.4s\n"
					"zip2   v29.4s, v18.4s,v19.4s\n"
					"prfm pldl1keep,[%[in10], #0x80]\n"

					// C1
					"zip1   v16.4s, v1.4s, v5.4s\n"
					"zip1   v17.4s, v3.4s, v7.4s\n"
					"zip2   v18.4s, v1.4s, v5.4s\n"
					"zip2   v19.4s, v3.4s, v7.4s\n"
					"prfm pldl1keep,[%[in11], #0x80]\n"

					"zip1   v22.4s, v16.4s,v17.4s\n"
					"zip2   v25.4s, v16.4s,v17.4s\n"
					"zip1   v28.4s, v18.4s,v19.4s\n"
					"zip2   v31.4s, v18.4s,v19.4s\n"

					// B1
					"zip1   v16.4s, v9.4s, v13.4s\n"
					"zip1   v17.4s, v11.4s,v15.4s\n"
					"zip2   v18.4s, v9.4s, v13.4s\n"
					"zip2   v19.4s, v11.4s,v15.4s\n"

					"zip1   v21.4s, v16.4s,v17.4s\n"
					"zip2   v24.4s, v16.4s,v17.4s\n"
					"zip1   v27.4s, v18.4s,v19.4s\n"
					"zip2   v30.4s, v18.4s,v19.4s\n"

					"stp    q20,q21, [%[out]], 0x20\n"
					"stp    q22,q23, [%[out]], 0x20\n"
					"stp    q24,q25, [%[out]], 0x20\n"
					"stp    q26,q27, [%[out]], 0x20\n"
					"stp    q28,q29, [%[out]], 0x20\n"
					"stp    q30,q31, [%[out]], 0x20\n"
				: [in0]"+r"(in0), [in1]"+r"(in1), [in2]"+r"(in2), [in3]"+r"(in3),
				  [in4]"+r"(in4), [in5]"+r"(in5), [in6]"+r"(in6), [in7]"+r"(in7),
				  [in8]"+r"(in8), [in9]"+r"(in9), [in10]"+r"(in10), [in11]"+r"(in11),
				  [out]"+r"(out)
				:
				: "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
				  "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
				  "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
				  "v30", "v31", "memory"
				);
			}

			for( ; len > 0; len-- ) {
				*out++ = *in0++;
				*out++ = *in1++;
				*out++ = *in2++;
				*out++ = *in3++;
				*out++ = *in4++;
				*out++ = *in5++;
				*out++ = *in6++;
				*out++ = *in7++;
				*out++ = *in8++;
				*out++ = *in9++;
				*out++ = *in10++;
				*out++ = *in11++;
			}

			if( height > 12 ) {
				height -= 12;
			} else {
				break;
			}
		}
	}
};

template<>
struct CInterleaver<true, 8> : public CInterleaverBase<true, 8> {
	static void Prepare(float* out, const float* in, size_t stride, size_t width, size_t height)
	{
		if( height == 0 ) {
			return;
		}
		float zerobuff[16];
		memset(zerobuff, 0, sizeof(zerobuff));

		while( true ) {
			size_t len = width;
			const float* in0 = in;
			const float* in1 = in0 + stride;
			const float* in2 = in1 + stride;
			const float* in3 = in2 + stride;
			const float* in4 = in3 + stride;
			const float* in5 = in4 + stride;
			const float* in6 = in5 + stride;
			const float* in7 = in6 + stride;
			in = in7 + stride;

			__asm __volatile(
				"prfm pldl1keep,[%[in0], #0x0]\n"
				"prfm pldl1keep,[%[in0], #0x40]\n"
				"prfm pldl1keep,[%[in1], #0x0]\n"
				"prfm pldl1keep,[%[in1], #0x40]\n"
				"prfm pldl1keep,[%[in2], #0x0]\n"
				"prfm pldl1keep,[%[in2], #0x40]\n"
				"prfm pldl1keep,[%[in3], #0x0]\n"
				"prfm pldl1keep,[%[in3], #0x40]\n"
				"prfm pldl1keep,[%[in4], #0x0]\n"
				"prfm pldl1keep,[%[in4], #0x40]\n"
				"prfm pldl1keep,[%[in5], #0x0]\n"
				"prfm pldl1keep,[%[in5], #0x40]\n"
				"prfm pldl1keep,[%[in6], #0x0]\n"
				"prfm pldl1keep,[%[in6], #0x40]\n"
				"prfm pldl1keep,[%[in7], #0x0]\n"
				"prfm pldl1keep,[%[in7], #0x40]\n"
				:
			: [in0] "r"(in0), [in1] "r"(in1), [in2] "r"(in2), [in3] "r"(in3),
			  [in4] "r"(in4), [in5] "r"(in5), [in6] "r"(in6), [in7] "r"(in7)
			: "memory"
			);
			for( ; len > 7; len -= 8 ) {
				switch (height) {
				case 1:
					in1 = zerobuff;
					// nobreak
				case 2:
					in2 = zerobuff;
					// nobreak;
				case 3:
					in3 = zerobuff;
					// nobreak;
				case 4:
					in4 = zerobuff;
					// nobreak;
				case 5:
					in5 = zerobuff;
					// nobreak;
				case 6:
					in6 = zerobuff;
					// nobreak;
				case 7:
					in7 = zerobuff;
					// nobreak;
				default:
					break;
				}
				__asm __volatile(
					"ldp    q0,  q1,  [%[in0]], #0x20\n"
					"ldp    q2,  q3,  [%[in1]], #0x20\n"
					"ldp    q4,  q5,  [%[in2]], #0x20\n"
					"zip1   v16.4s, v0.4s, v4.4s\n"
					"prfm pldl1keep,[%[in0], #0x80]\n"

					"ldp    q6,  q7,  [%[in3]], #0x20\n"
					"zip1   v17.4s, v2.4s, v6.4s\n"
					"ldp    q8,  q9,  [%[in4]], #0x20\n"
					"ldp    q10, q11, [%[in5]], #0x20\n"
					"ldp    q12, q13, [%[in6]], #0x20\n"
					"zip1   v18.4s, v8.4s, v12.4s\n"
					"prfm pldl1keep,[%[in1], #0x80]\n"
					"ldp    q14, q15, [%[in7]], #0x20\n"
					"zip1   v19.4s, v10.4s, v14.4s\n"

					"zip1   v20.4s, v16.4s, v17.4s\n"
					"prfm pldl1keep,[%[in2], #0x80]\n"
					"zip1   v21.4s, v18.4s, v19.4s\n"
					"zip2   v22.4s, v16.4s, v17.4s\n"
					"zip2   v23.4s, v18.4s, v19.4s\n"

					"zip2   v16.4s, v0.4s, v4.4s\n"
					"prfm pldl1keep,[%[in3], #0x80]\n"
					"zip2   v17.4s, v2.4s, v6.4s\n"
					"stp    q20, q21, [%[out]], #0x20\n"

					"zip2   v18.4s, v8.4s, v12.4s\n"
					"zip2   v19.4s, v10.4s, v14.4s\n"
					"stp    q22, q23, [%[out]], #0x20\n"

					"zip1   v20.4s, v16.4s, v17.4s\n"
					"prfm pldl1keep,[%[in4], #0x80]\n"
					"zip1   v21.4s, v18.4s, v19.4s\n"
					"zip2   v22.4s, v16.4s, v17.4s\n"
					"zip2   v23.4s, v18.4s, v19.4s\n"

					"zip1   v16.4s, v1.4s, v5.4s\n"
					"prfm pldl1keep,[%[in5], #0x80]\n"
					"zip1   v17.4s, v3.4s, v7.4s\n"
					"stp    q20, q21, [%[out]], #0x20\n"

					"zip1   v18.4s, v9.4s, v13.4s\n"
					"zip1   v19.4s, v11.4s, v15.4s\n"
					"stp    q22, q23, [%[out]], #0x20\n"

					"zip1   v20.4s, v16.4s, v17.4s\n"
					"zip1   v21.4s, v18.4s, v19.4s\n"
					"zip2   v22.4s, v16.4s, v17.4s\n"
					"prfm pldl1keep,[%[in6], #0x80]\n"
					"zip2   v23.4s, v18.4s, v19.4s\n"

					"zip2   v16.4s, v1.4s, v5.4s\n"
					"zip2   v17.4s, v3.4s, v7.4s\n"
					"stp    q20, q21, [%[out]], #0x20\n"

					"zip2   v18.4s, v9.4s, v13.4s\n"
					"prfm pldl1keep,[%[in7], #0x80]\n"
					"zip2   v19.4s, v11.4s, v15.4s\n"
					"stp    q22, q23, [%[out]], #0x20\n"

					"zip1   v20.4s, v16.4s, v17.4s\n"
					"zip1   v21.4s, v18.4s, v19.4s\n"
					"stp    q20, q21, [%[out]], #0x20\n"

					"zip2   v22.4s, v16.4s, v17.4s\n"
					"zip2   v23.4s, v18.4s, v19.4s\n"
					"stp    q22, q23, [%[out]], #0x20\n"
				: [in0]"+r"(in0), [in1]"+r"(in1), [in2]"+r"(in2), [in3]"+r"(in3),
				  [in4]"+r"(in4), [in5]"+r"(in5), [in6]"+r"(in6), [in7]"+r"(in7),
				  [out]"+r"(out)
				:
				: "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
				  "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "memory"
				);
			}
			for( ; len > 0; len-- ) {
				*out++ = *in0++;
				*out++ = *in1++;
				*out++ = *in2++;
				*out++ = *in3++;
				*out++ = *in4++;
				*out++ = *in5++;
				*out++ = *in6++;
				*out++ = *in7++;
			}

			if( height > 8 ) {
				height -= 8;
			} else {
				break;
			}
		}
	}
};

template<>
struct CInterleaver<true, 4> : public CInterleaverBase<true, 4> {
	static void Prepare(float* out, const float* in, size_t stride, size_t width, size_t height) {
		if( height == 0 ) {
			return;
		}
		float zerobuff[16];
		memset(zerobuff, 0, sizeof(zerobuff));
		while( true ) {
			size_t len = width;
			const float* in0 = in;
			const float* in1 = in0 + stride;
			const float* in2 = in1 + stride;
			const float* in3 = in2 + stride;
			in = in3 + stride;
			__asm __volatile(
				"prfm pldl1keep,[%[in0], #0x0]\n"
				"prfm pldl1keep,[%[in0], #0x40]\n"
				"prfm pldl1keep,[%[in1], #0x0]\n"
				"prfm pldl1keep,[%[in1], #0x40]\n"
				"prfm pldl1keep,[%[in2], #0x0]\n"
				"prfm pldl1keep,[%[in2], #0x40]\n"
				"prfm pldl1keep,[%[in3], #0x0]\n"
				"prfm pldl1keep,[%[in3], #0x40]\n"
			:
			: [in0] "r"(in0), [in1] "r"(in1), [in2] "r"(in2), [in3] "r"(in3)
			: "memory"
			);
			for( ; len >= 8; len -= 8 ) {
				switch (height) {
				case 1:
					in1 = zerobuff;
					// nobreak
				case 2:
					in2 = zerobuff;
					// nobreak;
				case 3:
					in3 = zerobuff;
					// nobreak;
				default:
					break;
				}
				__asm __volatile(
					"ldp    q0, q1, [%[in0]], 0x20\n"
					"ldp    q2, q3, [%[in1]], 0x20\n"
					"ldp    q4, q5, [%[in2]], 0x20\n"
					"ldp    q6, q7, [%[in3]], 0x20\n"

					"zip1   v8.4s,  v0.4s, v4.4s\n"
					"zip1   v9.4s,  v2.4s, v6.4s\n"
					"prfm pldl1keep,[%[in0], #0x80]\n"
					"zip1   v10.4s, v8.4s, v9.4s\n"
					"zip2   v11.4s, v8.4s, v9.4s\n"
					"prfm pldl1keep,[%[in1], #0x80]\n"

					"str    q10, [%[out]], #0x10\n"
					"zip2   v8.4s,  v0.4s, v4.4s\n"
					"prfm pldl1keep,[%[in2], #0x80]\n"
					"zip2   v9.4s,  v2.4s, v6.4s\n"
					"str    q11, [%[out]], #0x10\n"
					"zip1   v10.4s, v8.4s, v9.4s\n"
					"zip2   v11.4s, v8.4s, v9.4s\n"
					"prfm pldl1keep,[%[in3], #0x80]\n"

					"str    q10, [%[out]], #0x10\n"
					"zip1   v8.4s,  v1.4s, v5.4s\n"
					"zip1   v9.4s,  v3.4s, v7.4s\n"
					"str    q11, [%[out]], #0x10\n"
					"zip1   v10.4s, v8.4s, v9.4s\n"
					"zip2   v11.4s, v8.4s, v9.4s\n"

					"str    q10, [%[out]], #0x10\n"
					"zip2   v8.4s,  v1.4s, v5.4s\n"
					"zip2   v9.4s,  v3.4s, v7.4s\n"
					"str    q11, [%[out]], #0x10\n"
					"zip1   v10.4s, v8.4s, v9.4s\n"
					"zip2   v11.4s, v8.4s, v9.4s\n"

					"str    q10, [%[out]], #0x10\n"
					"str    q11, [%[out]], #0x10\n"
					: [in0] "+r"(in0), [in1] "+r"(in1), [in2] "+r"(in2), [in3] "+r"(in3), [out] "+r"(out)
					:
					: "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "memory"
					);
			}

			if( len >= 4 ) {
				len -= 4;
				__asm __volatile(
					"ldr    q0, [%[in0]], 0x10\n"
					"ldr    q2, [%[in1]], 0x10\n"
					"ldr    q4, [%[in2]], 0x10\n"
					"ldr    q6, [%[in3]], 0x10\n"

					"zip1   v8.4s,  v0.4s, v4.4s\n"
					"zip1   v9.4s,  v2.4s, v6.4s\n"
					"zip1   v10.4s, v8.4s, v9.4s\n"
					"zip2   v11.4s, v8.4s, v9.4s\n"

					"str    q10, [%[out]], #0x10\n"
					"zip2   v8.4s,  v0.4s, v4.4s\n"
					"zip2   v9.4s,  v2.4s, v6.4s\n"
					"str    q11, [%[out]], #0x10\n"
					"zip1   v10.4s, v8.4s, v9.4s\n"
					"zip2   v11.4s, v8.4s, v9.4s\n"

					"str    q10, [%[out]], #0x10\n"
					"str    q11, [%[out]], #0x10\n"
				: [in0] "+r" (in0), [in1] "+r" (in1), [in2] "+r" (in2), [in3] "+r" (in3), [out] "+r"(out)
				:
				: "v0", "v2", "v4", "v6", "v8", "v9", "v10", "v11", "memory"
				);
			}

			for( ; len > 0; len-- ) {
				*out++ = *in0++;
				*out++ = *in1++;
				*out++ = *in2++;
				*out++ = *in3++;
			}

			if( height > 4 ) {
				height -= 4;
			} else {
				break;
			}
		}
	}
};
