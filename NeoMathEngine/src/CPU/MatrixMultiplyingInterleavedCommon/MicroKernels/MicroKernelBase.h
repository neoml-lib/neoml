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

#include <cstring>

// The base implementation of a micro-kernel for matrix multiplication
// It expects the first multiplier, but not the second, to be transposed
// The two multipliers are stored side-by-side
// This is a naive implementation and should not be used in most cases
template<size_t H, size_t W>
struct CMicroKernelBase {
	static constexpr size_t height = H;
	static constexpr size_t width = W;

	static void Calculate(const float* aPtr, const float* bPtr, float* cPtr, size_t cRowSize, size_t k) 
	{
		for( size_t i = 0; i < height; ++i ) {
			for( size_t j = 0; j < width; ++j ) {
				for( size_t l = 0; l < k; ++l ) {
					cPtr[i * cRowSize + j] += aPtr[l * height + i] * bPtr[l * width + j];
				}
			}
		}
	}
};

// Combine the kernels horizontally
template<class ... Kernels>
struct CKernelCombineHorizontal;

template<class Kernel>
struct CKernelCombineHorizontal<Kernel> : public Kernel {};

template<class Kernel, class ... Tail>
struct CKernelCombineHorizontal<Kernel, Tail ...> : public Kernel {
	static constexpr size_t height = Kernel::height;
	static constexpr size_t width = Kernel::width;
	using TailKernelRight = CKernelCombineHorizontal<Tail...>;
	static_assert(Kernel::height == TailKernelRight::height, "Kernels must have same height");
};

// Combine the kernels vertically
// Not fully safe because no checks of TailKernelRight are performed 
// (they should either not be there or be of equal width for all kernels)
template<class ... Kernels>
struct CKernelCombineVertical;

template<class Kernel>
struct CKernelCombineVertical<Kernel> : public Kernel {};

template<class Kernel, class ... Tail>
struct CKernelCombineVertical<Kernel, Tail ...> : public Kernel {
	static constexpr size_t height = Kernel::height;
	static constexpr size_t width = Kernel::width;
	using TailKernelBottom = CKernelCombineVertical<Tail...>;
	static_assert(Kernel::width == TailKernelBottom::width, "Kernels must have same width");
};
