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

#include <Kernels_6x.h>

namespace NeoML {

void AvxMultiplyMatrix( bool transA, bool transB,
	IMathEngine *engine,
	const float* aPtr, size_t aRowSize,
	const float* bPtr, size_t bRowSize,
	float* cPtr, size_t cRowSize,
	size_t m, size_t n, size_t k )
{
	const CCPUInfo& cpuinfo = CCPUInfo::GetCPUInfo();

	unsigned char transSelector = ( transA ? 0b10 : 0 ) + ( transB ? 0b01 : 0 );
	switch( transSelector ) {
	case 0b00:
		CMatrixMultiplier<CKernelCombi, CInterleaverDefault, false, false, CTmpMemoryHandler, IMathEngine>::Multiply
				( engine, cpuinfo, aPtr, aRowSize, bPtr, bRowSize, cPtr, cRowSize, m, n, k );
		break;
	case 0b01:
		CMatrixMultiplier<CKernelCombi, CInterleaverDefault, false, true, CTmpMemoryHandler, IMathEngine>::Multiply
				( engine, cpuinfo, aPtr, aRowSize, bPtr, bRowSize, cPtr, cRowSize, m, n, k );
		break;
	case 0b10:
		CMatrixMultiplier<CKernelCombi, CInterleaverDefault, true, false, CTmpMemoryHandler, IMathEngine>::Multiply
				( engine, cpuinfo, aPtr, aRowSize, bPtr, bRowSize, cPtr, cRowSize, m, n, k );
		break;
	case 0b11:
		CMatrixMultiplier<CKernelCombi, CInterleaverDefault, true, true, CTmpMemoryHandler, IMathEngine>::Multiply
				( engine, cpuinfo, aPtr, aRowSize, bPtr, bRowSize, cPtr, cRowSize, m, n, k );
		break;
	}

}

}
