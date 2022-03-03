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

// This code does not use any NeoML structures, 
// so you can create a mini-project that would not require NeoML dependencies 
// but will let you test, optimize, and compile performance statistics. 
// All types are passed as template parameters

// Matrix product uses a micro-kernel
// The micro-kernel is a function that provides matrix product with the result of specified size
// The micro-kernel is optimized to store the result which is added to in the registers
// As each CPU has different number of registers, they each need a different size micro-kernel
// In this algorithm, the micro-kernel accepts as input an A matrix stored column-by-column (that is, transposed)
// and a B matrix stored row-by row
// The algorithm splits the input matrices into micro-blocks (numbered with #) and processes them using the micro-kernel
// The micro-blocks size should fit into L1 cache
// Micro-blocks are merged into blocks. The B matrix blocks should fit into L2 cache
// The A matrix blocks include the whole "wide" column
// The external algorithm cycle over the K dimension:
// Each step works with a block of the A matrix
// It is copied into temporary memory, repositioning the data 
// so that each micro-block is stored consecutively, column-by-column
// The next cycle goes over the B matrix "wide" row, copying each block into temporary memory.
// Two internal cycles iterate over the micro-blocks in the A and B blocks (stored in temporary memory)
// For each A micro-block all micro-blocks from B are accessed, which should happen fast because B is stored in L2
// If the matrix size is not the exact multiple of the micro-kernel size, additional micro-kernels can be used
// If the micro-kernel has the TailKernelBottom type defined, it will be used to process the rest of the rows,
// and if the TailKernelRight type is defined it will be used to process the rest of the columns
// If a set of micro-kernels is used, for all kernel sizes all combinations should be implemented
// That is, a correct micro-kernel set should fit into the rectangle
// +-------+----+-+
// |       |    | |
// |       |    | |
// |       |    | |
// |       |    | |
// +-------+----+-+
// |       |    | |
// |       |    | |
// +-------+----+-+
// +-------+----+-+
// If no tail kernel is defined but the matrix size requires it, the last kernel is used
// to process the rest of the data, copying it into a temporary buffer and filling the extra space with zeros

//                 <-------N------->
//  <-----K----->       <--n-->
// ^    <-k->      +---------------+  ^
// |    .   .      |    .     .    |  |
// |  ^............|--+-##----+----|  | <-MatrixB
// K  k .          |  | ##btmp|    |  K
// |  v..   .......|--+-##----+----|  |
// |    .   .      |    ..         |  |
// v    .   .      +---------------+  v
//      .   .           ..
//  +-----------+  +---------------+  ^
//  |   |   |   |  |    ..         |  |
//  |   |   |   |  |    ..         |  |
//  |   #####...|..|....##         |  M
//  |   | a |   |  |     ^         |  |
//  |   |tmp|   |  |  microkernel  |  |
//  +-----------+  +---------------+  v
//  <-----K----->  <-------N------->
//     ^               ^
//     MatrixA         MatrixC

// Copies the C submatrix into a temporary buffer to process the boundaries
template<class Kernel>
inline void LoadCPart(float* dst, const float* src,
	size_t rowSize, size_t height, size_t width)
{
	float* dstDataEnd = dst + height * Kernel::width;
	float* dstEnd = dst + Kernel::height * Kernel::width;
	for( ; dst < dstDataEnd; dst += Kernel::width, src += rowSize ) {
		memcpy(dst, src, width * sizeof(float));
		memset(dst + width, 0, (Kernel::width - width) * sizeof(float));
	}
	for ( ; dst < dstEnd; dst += Kernel::width ) {
		memset(dst, 0, Kernel::width * sizeof(float));
	}
}

// Copies the C submatrix from a temporary buffer to process the boundaries
template<class Kernel>
inline void StoreCPart(const float* src, float* dst,
	size_t rowSize, size_t height, size_t width)
{
	const float* srcEnd = src + height * Kernel::width;
	for( ; src < srcEnd; src += Kernel::width, dst += rowSize ) {
		memcpy(dst, src, width * sizeof(float));
	}
}

// A stub class used to get void from any class.
// Further on, several classes have a template parameter SFINAEHolder = void
// which allows partial implementation
// As SFINaeHolder is never passed explicitly in function calls, the partial implementations will always have priority
// unless the class passed to SFINARFilter (assigned to SFINAEHolder in the partial implementations) does not exist
template<class T>
struct SFINAEFilter {
	using type = void;
};

// Prepare the A matrix. Works in the case when the kernel is last in the TailKernelBottom chain
template <bool Trans, class Kernel, template<bool, size_t> class Interleaver, class SFINAEHolder = void>
struct PreparerAHelper {
	static constexpr size_t minHeight = Kernel::height;
	static void Prepare(float* out, const float* in, size_t stride, size_t height, size_t width)
	{
		Interleaver<!Trans, Kernel::height>::Prepare(out, in, stride, width, height);
	}
	static size_t HeightTail(size_t height)
	{
		return height % Kernel::height;
	}
};

// Prepare the A matrix (recursive implementation). Works when at least one other kernel exists
template <bool Trans, class Kernel, template<bool, size_t> class Interleaver>
struct PreparerAHelper<Trans, Kernel, Interleaver, typename SFINAEFilter<typename Kernel::TailKernelBottom>::type> {
	using NextPreparerA = PreparerAHelper<Trans, typename Kernel::TailKernelBottom, Interleaver>;
	static constexpr size_t minHeight = Kernel::height > NextPreparerA::minHeight ? NextPreparerA::minHeight : Kernel::height;
	static void Prepare(float* out, const float* in, size_t stride, size_t height, size_t width)
	{
		size_t hRound = height / Kernel::height * Kernel::height;
		Interleaver<!Trans, Kernel::height>::Prepare(out, in, stride, width, hRound);
		out += hRound * width;
		in += !Trans ? hRound * stride : hRound;
		NextPreparerA::Prepare(out, in, stride, height - hRound, width);
	}
	static size_t HeightTail(size_t height)
	{
		return minHeight == 1 ? 0 : NextPreparerA::HeightTail(height % Kernel::height);
	}
};

// Prepare the B matrix. Works in the case when the kernel is last in the TailKernelRight chain
template <bool Trans, class Kernel, template<bool, size_t> class Interleaver, class SFINAEHolder = void>
struct PreparerBHelper {
	static constexpr size_t minWidth = Kernel::width;
	static void Prepare(float* out, const float* in, size_t stride, size_t height, size_t width)
	{
		Interleaver<Trans, Kernel::width>::Prepare(out, in, stride, height, width);
	}
	static size_t WidthTail(size_t width)
	{
		return width % Kernel::width;
	}
};

// Prepare the B matrix (recursive implementation). Works when at least one other kernel exists
template <bool Trans, class Kernel, template<bool, size_t> class Interleaver>
struct PreparerBHelper<Trans, Kernel, Interleaver, typename SFINAEFilter<typename Kernel::TailKernelRight>::type> {
	using NextPreparerB = PreparerBHelper<Trans, typename Kernel::TailKernelRight, Interleaver>;
	static constexpr size_t minWidth = Kernel::width > NextPreparerB::minWidth ? NextPreparerB::minWidth : Kernel::width;
	static void Prepare(float* out, const float* in, size_t stride, size_t height, size_t width)
	{
		size_t wRound = width / Kernel::width * Kernel::width;
		Interleaver<Trans, Kernel::width>::Prepare(out, in, stride, height, wRound);
		out += wRound * height;
		in += Trans ? wRound * stride : wRound;
		NextPreparerB::Prepare(out, in, stride, height, width - wRound);
	}
	static size_t WidthTail(size_t width)
	{
		return minWidth == 1 ? 0 : NextPreparerB::WidthTail(width % Kernel::width);
	}
};

// Process the columns when there is fewer than the kernel width
// The general implementation uses the same kernel, temp buffer and nullify
template<class Kernel, class SFINAEHolder = void>
struct TailProcessorRight {
	static void Calculate(const float* aPtr, const float* bPtr, float* cPtr,
		size_t cRowSize, size_t k, float* cTmp, size_t height, size_t width)
	{
		LoadCPart<Kernel>(cTmp, cPtr, cRowSize, height, width);
		Kernel::Calculate(aPtr, bPtr, cTmp, Kernel::width, k);
		StoreCPart<Kernel>(cTmp, cPtr, cRowSize, height, width);
	}
};

// Process the rows when there is fewer than the kernel height
// The general implementation uses the same kernel, temp buffer and nullify
template<class Kernel, class SFINAEHolder = void>
struct TailProcessorBottom {
	static void Calculate(const float* aPtr, const float* bPtr, float* cPtr,
		size_t cRowSize, size_t k, float* cTmp, size_t height, size_t width)
	{
		const size_t bStep = Kernel::width * k;
		for( ; width >= Kernel::width; width -= Kernel::width ) {
			LoadCPart<Kernel>(cTmp, cPtr, cRowSize, height, Kernel::width);
			Kernel::Calculate(aPtr, bPtr, cTmp, Kernel::width, k);
			StoreCPart<Kernel>(cTmp, cPtr, cRowSize, height, Kernel::width);
			cPtr += Kernel::width;
			bPtr += bStep;
		}
		if( width > 0 ) {
			TailProcessorRight<Kernel>::Calculate(aPtr, bPtr, cPtr, cRowSize, k, cTmp, height, width);
		}
	}
};

// Process the submatrix with the kernel once A and B are prepared
template<class Kernel>
inline void ProcessKernel(const float* aPtr, const float* bPtr, float* cPtr, size_t cRowSize, size_t k, float* cTmp, size_t height, size_t width)
{
	const size_t aStep = Kernel::height * k;
	const size_t bStep = Kernel::width * k;
	const size_t cStep = Kernel::height * cRowSize;
	for( ; height >= Kernel::height; height -= Kernel::height ) {
		const float* bKernel = bPtr;
		float* cKernel = cPtr;
		size_t columnLeft = width;
		for( ; columnLeft >= Kernel::width; columnLeft -= Kernel::width ) {
			Kernel::Calculate(aPtr, bKernel, cKernel, cRowSize, k);
			cKernel += Kernel::width;
			bKernel += bStep;
		}
		if( columnLeft > 0 ) {
			TailProcessorRight<Kernel>::Calculate(aPtr, bKernel, cKernel, cRowSize, k, cTmp, Kernel::height, columnLeft);
		}
		aPtr += aStep;
		cPtr += cStep;
	}
	if( height > 0 ) {
		TailProcessorBottom<Kernel>::Calculate(aPtr, bPtr, cPtr, cRowSize, k, cTmp, height, width);
	}
}

// Process the columns when there is fewer than the kernel width
// The implementation for the case when another kernel exists
template<class Kernel>
struct TailProcessorRight<Kernel, typename SFINAEFilter<typename Kernel::TailKernelRight>::type> {
	static void Calculate(const float* aPtr, const float* bPtr, float* cPtr,
		size_t cRowSize, size_t k, float* cTmp, size_t height, size_t width)
	{
		using CKernel = typename Kernel::TailKernelRight;
		const size_t bStep = CKernel::width * k;
		for( ; width >= CKernel::width; width -= CKernel::width ) {
			if( height >= CKernel::height ) {
			CKernel::Calculate(aPtr, bPtr, cPtr, cRowSize, k);
			} else {
				LoadCPart<CKernel>(cTmp, cPtr, cRowSize, height, CKernel::width);
				CKernel::Calculate(aPtr, bPtr, cTmp, CKernel::width, k);
				StoreCPart<CKernel>(cTmp, cPtr, cRowSize, height, CKernel::width);
			}
			cPtr += CKernel::width;
			bPtr += bStep;
		}
		if( width > 0 ) {
			TailProcessorRight<CKernel>::Calculate(aPtr, bPtr, cPtr, cRowSize, k, cTmp, height, width);
		}
	}
};

// Process the rows when there is fewer than the kernel height
// The implementation for the case when another kernel exists
template<class Kernel>
struct TailProcessorBottom<Kernel, typename SFINAEFilter<typename Kernel::TailKernelBottom>::type> {
	static void Calculate(const float* aPtr, const float* bPtr, float* cPtr,
		size_t cRowSize, size_t k, float* cTmp, size_t height, size_t width)
	{
		ProcessKernel<typename Kernel::TailKernelBottom>(aPtr, bPtr, cPtr, cRowSize, k, cTmp, height, width);
	}
};

// Matrix product. Calculates the block size to fit into caches, 
// prepares A and B matrix blocks and performs multiplication
template<class Kernel, template<bool, size_t> class Interleaver, bool ATransposed, bool BTransposed, class MemoryHandler, class Engine>
struct CMatrixMultiplier {
	template<class CCPUInfo>
	static void Multiply(Engine *engine, const CCPUInfo &cpuInfo, const float* aPtr, size_t aRowSize,
		const float* bPtr, size_t bRowSize, float* cPtr, size_t cRowSize, size_t m, size_t n, size_t k)
	{
		// Calculate block size
		// A and B micro-blocks should fit into L1, same as the micro-kernel result
		// Several more cache lines may be taken up by the calling function variables
		size_t kBlock =
			(cpuInfo.L1CacheSize - Kernel::height * Kernel::width * sizeof(float) - 64 * 4) /
			((Kernel::height + Kernel::width) * sizeof(float));
		kBlock = Ceildiv(k, Ceildiv(k, kBlock));

		// 10% L2 should be left for overhead, in addition to L1
		size_t nBlock = (cpuInfo.L2CacheSize * 90 / 100 - cpuInfo.L1CacheSize) /
			(kBlock * sizeof(float));
		nBlock = Ceildiv(n, Ceildiv(n, nBlock));
		if( nBlock > Kernel::width && nBlock < n ) {
			nBlock = nBlock / Kernel::width * Kernel::width;
		} else {
			nBlock = Kernel::width;
		}

		// Temporary memory
		MemoryHandler aTmpHandler(engine, kBlock * Ceildiv(m, Kernel::height) * Kernel::height);
		MemoryHandler bTmpHandler(engine, kBlock * nBlock);
		MemoryHandler cTmpHandler(engine, Kernel::height * Kernel::width);
		float* aTmpBuffer = aTmpHandler.get();
		float* bTmp = bTmpHandler.get();
		float* cTmp = cTmpHandler.get();

		// One cycle iteration and the end condition for the A matrix
		// Depends on whether it has been transposed
		size_t aStep;
		const float* aEnd;
		if( ATransposed ) {
			aStep = kBlock * aRowSize;
			aEnd = aPtr + k * aRowSize;
		} else {
			aStep = kBlock;
			aEnd = aPtr + k;
		}

		// One cycle iteration and the end condition for the B matrix
		// Depends on whether it has been transposed
		size_t bWStep;
		size_t bHStep;
		size_t bLineSize;
		if( BTransposed ) {
			bHStep = kBlock;
			bWStep = nBlock * bRowSize;
			bLineSize = n * bRowSize;
		} else {
			bHStep = kBlock * bRowSize;
			bWStep = nBlock;
			bLineSize = n;
		}

		// The cycle over the wide columns of A and wide rows of B
		// Each A wide column is copied to a temporary buffer
		size_t kLeft = k;
		for( ; aPtr < aEnd; aPtr += aStep, bPtr += bHStep, kLeft -= kBlock ) {
			size_t kBlockSize = kBlock < kLeft ? kBlock : kLeft;
			bool APrepared = PreparerA::minHeight == 1 && (!ATransposed || aRowSize == 1) && m == 1;
			//fprintf(stderr, "  Prepared: %c\n", APrepared ? 'T' : 'F');
			//kernel.template PrepareA<ATransposed>(aTmp, aPtr, aRowSize, m, kBlockSize);
			const float* aTmp;
			if( APrepared ) {
				aTmp = aPtr;
			} else {
				PreparerA::Prepare(aTmpBuffer, aPtr, aRowSize, m, kBlockSize);
				aTmp = aTmpBuffer;
			}
			const float* lastBColumn = bPtr + bLineSize;
			float* cColumn = cPtr;
			size_t nLeft = n;
			// The cycle over the B blocks
			// Each block is copied to a temporary buffer
			for( const float* bColumn = bPtr; bColumn < lastBColumn; bColumn += bWStep) {
				size_t nBlockSize = nBlock < nLeft ? nBlock : nLeft;
				PreparerB::Prepare(bTmp, bColumn, bRowSize, kBlockSize, nBlockSize);
				ProcessKernel<Kernel>(aTmp, bTmp, cColumn, cRowSize, kBlockSize, cTmp, m, nBlockSize);
				cColumn += nBlock;
				nLeft -= nBlock;
			}
		}
	}
private:
	using PreparerA = PreparerAHelper<ATransposed, Kernel, Interleaver>;
	using PreparerB = PreparerBHelper<BTransposed, Kernel, Interleaver>;
	// Integer division, rounding up
	static constexpr size_t Ceildiv(size_t a, size_t b) {
		return (a + b - 1) / b;
	}
};
