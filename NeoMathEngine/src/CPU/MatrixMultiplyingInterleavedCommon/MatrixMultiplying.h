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

#include "MatrixMultiplier.h"

// The kernel is chosen depending on the architecture
// The kernel header files do not have these ifdef, so they may not be included in the project without a pre-check
// We use relative paths for #include "" which lets you store the multiplication code in a separate project
// This may simplify testing the performance times

#ifdef __aarch64__

#include "CPU/arm/MatrixMultiplyingInterleaved/MicroKernels/Kernel_ARM64_8x12.h"
#include "CPU/arm/MatrixMultiplyingInterleaved/MicroKernels/Kernel_ARM64_8x4.h"
#include "CPU/arm/MatrixMultiplyingInterleaved/MicroKernels/Kernel_ARM64_8x1.h"
#include "CPU/arm/MatrixMultiplyingInterleaved/MicroKernels/Kernel_ARM64_4x12.h"
#include "CPU/arm/MatrixMultiplyingInterleaved/MicroKernels/Kernel_ARM64_4x4.h"
#include "CPU/arm/MatrixMultiplyingInterleaved/MicroKernels/Kernel_ARM64_4x1.h"
#include "CPU/arm/MatrixMultiplyingInterleaved/MicroKernels/Kernel_ARM64_1x12.h"
#include "CPU/arm/MatrixMultiplyingInterleaved/MicroKernels/Kernel_ARM64_1x4.h"
#include "CPU/arm/MatrixMultiplyingInterleaved/Interleavers/Interleaver_ARM64.h"
using CMicroKernel1x1 = CMicroKernelBase<1, 1>;
using CMicroKernelDefault = CKernelCombineVertical<
	CKernelCombineHorizontal<CMicroKernel8x12, CMicroKernel8x4, CMicroKernel8x1>,
	CKernelCombineHorizontal<CMicroKernel4x12, CMicroKernel4x4, CMicroKernel4x1>,
	CKernelCombineHorizontal<CMicroKernel1x12, CMicroKernel1x4, CMicroKernel1x1>
>;
template <bool Transpose, size_t Len> using CInterleaverDefault = CInterleaver<Transpose, Len>;

#elif __arm__ && __ARM_NEON

#include "CPU/arm/MatrixMultiplyingInterleaved/MicroKernels/Kernel_ARM32NEON_6x8.h"
#include "Interleavers/InterleaverBase.h"
using CMicroKernelDefault = CMicroKernel6x8;
template <bool Transpose, size_t Len> using CInterleaverDefault = CInterleaverBase<Transpose, Len>;

#else

#include "MicroKernels/MicroKernelBase.h"
#include "Interleavers/InterleaverBase.h"
using CMicroKernelDefault = CMicroKernelBase<1, 1>;
template <bool Transpose, size_t Len> using CInterleaverDefault = CInterleaverBase<Transpose, Len>;

#endif

template<bool ATransposed, bool BTransposed, class MemoryHandler, class Engine, class CCPUInfo>
inline void MultiplyMatrix(Engine *engine, const CCPUInfo &cpuInfo,
	const float* aPtr, size_t aRowSize,
	const float* bPtr, size_t bRowSize,
	float* cPtr, size_t cRowSize,
	size_t m, size_t n, size_t k)
{
	CMatrixMultiplier<CMicroKernelDefault, CInterleaverDefault, ATransposed, BTransposed, MemoryHandler, Engine>::Multiply
		(engine, cpuInfo, aPtr, aRowSize, bPtr, bRowSize, cPtr, cRowSize, m, n, k);
}