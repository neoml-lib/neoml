/* Copyright © 2023 ABBYY

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

#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

// Small matrices multiplication optimization descriptor
// Enabled only for x64 platform and CPU MathEngine, used MKL JIT
struct CCpuSmallMatricesMultiplyDesc : public CSmallMatricesMultiplyDesc {
	using TKernel = void( * )( void*, float*, float*, float* );

	void* MklJitter = nullptr; // pointer to allocated MKL JIT creator
	TKernel MklKernel = nullptr; // pointer to allocated MKL JIT function implemetation
	// Both of them would still be nullptr if:
	//  1. no MKL allowed on the platform  or
	//  2. matrices sizes are too big for effective JIT.

	CCpuSmallMatricesMultiplyDesc(
		int firstHeight, int firstWidth, int secondWidth, int secondRowSize, int resultWidth,
		bool resultAdd, bool trans1, bool trans2 );
	CCpuSmallMatricesMultiplyDesc( CCpuSmallMatricesMultiplyDesc&& ) = delete;
	CCpuSmallMatricesMultiplyDesc( const CCpuSmallMatricesMultiplyDesc& ) = delete;
	~CCpuSmallMatricesMultiplyDesc() override;
};

} // namespace NeoML
