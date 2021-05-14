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

#include <NeoMathEngine/SimdMathEngine.h>

#include <CPUInfo.h>
#include <MatrixMultiplyingInterleavedCommon/MatrixMultiplying.h>
#include <MatrixMultiplyingInterleavedCommon/CpuMemoryHelper.h>

#include <Interleavers.h>
#include <Kernels_6x.h>

namespace NeoML {

using CKernelCombi_16 = CKernelCombineHorizontal<CMicroKernel_6x16>;
using CKernelCombi_8 = CKernelCombineHorizontal<CMicroKernel_6x16, CMicroKernel_6x8>;
using CKernelCombi_4 = CKernelCombineHorizontal<CMicroKernel_6x16, CMicroKernel_6x8, CMicroKernel_6x4>;
using CKernelCombi_full = CKernelCombineHorizontal<CMicroKernel_6x16, CMicroKernel_6x8, CMicroKernel_6x4, CMicroKernel_6x2, CMicroKernel_6x1>;

template< class Kernel>
void AvxMultiplyMatrixSelected( bool transA, bool transB,
	IMathEngine *engine,
	const float* aPtr, size_t aRowSize,
	const float* bPtr, size_t bRowSize,
	float* cPtr, size_t cRowSize,
	size_t m, size_t n, size_t k )
{
	static const CCPUInfo& cpuinfo = CCPUInfo::GetCPUInfo();

	unsigned char transSelector = ( transA ? 0b10 : 0 ) + ( transB ? 0b01 : 0 );
	switch( transSelector ) {
	case 0b00:
		CMatrixMultiplier<Kernel, CInterleaverDefault, false, false, CTmpMemoryHandler, IMathEngine>::Multiply
				( engine, cpuinfo, aPtr, aRowSize, bPtr, bRowSize, cPtr, cRowSize, m, n, k );
		break;
	case 0b01:
		CMatrixMultiplier<Kernel, CInterleaverDefault, false, true, CTmpMemoryHandler, IMathEngine>::Multiply
				( engine, cpuinfo, aPtr, aRowSize, bPtr, bRowSize, cPtr, cRowSize, m, n, k );
		break;
	case 0b10:
		CMatrixMultiplier<Kernel, CInterleaverDefault, true, false, CTmpMemoryHandler, IMathEngine>::Multiply
				( engine, cpuinfo, aPtr, aRowSize, bPtr, bRowSize, cPtr, cRowSize, m, n, k );
		break;
	case 0b11:
		CMatrixMultiplier<Kernel, CInterleaverDefault, true, true, CTmpMemoryHandler, IMathEngine>::Multiply
				( engine, cpuinfo, aPtr, aRowSize, bPtr, bRowSize, cPtr, cRowSize, m, n, k );
		break;
	}
}

void AvxMultiplyMatrix( bool transA, bool transB,
	IMathEngine *engine,
	const float* aPtr, size_t aRowSize,
	const float* bPtr, size_t bRowSize,
	float* cPtr, size_t cRowSize,
	size_t m, size_t n, size_t k )
{
	// In some cases it is better choice to calculate matrix with big kernel in one or two steps rather than iterate over all
	// available kernels. It helps us to save time on preparing.
	switch( n % 16 ) {
	case 3:
	case 11:
		AvxMultiplyMatrixSelected<CKernelCombi_4>( transA, transB, engine, aPtr, aRowSize, bPtr, bRowSize, cPtr, cRowSize, m, n, k );
		break;
	case 5:
	case 6:
	case 7:
		AvxMultiplyMatrixSelected<CKernelCombi_8>( transA, transB, engine, aPtr, aRowSize, bPtr, bRowSize, cPtr, cRowSize, m, n, k );
		break;
	case 13:
	case 14:
	case 15:
		AvxMultiplyMatrixSelected<CKernelCombi_16>( transA, transB, engine, aPtr, aRowSize, bPtr, bRowSize, cPtr, cRowSize, m, n, k );
		break;
	default:
		AvxMultiplyMatrixSelected<CKernelCombi_full>( transA, transB, engine, aPtr, aRowSize, bPtr, bRowSize, cPtr, cRowSize, m, n, k );
	}
}

}
