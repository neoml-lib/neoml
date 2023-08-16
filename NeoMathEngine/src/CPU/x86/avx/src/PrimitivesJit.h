/* Copyright © 2017-2023 ABBYY

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

#include <CpuMathEngineDnnLstm.h>
#include <JitCommon.h>
#include <unordered_map>
#include <array>
#include <mutex>
#include <functional>

namespace NeoML {

class IMathEngine;

class CPrimitivesJit {
public:
	CPrimitivesJit( IMathEngine* );

	void Tanh( float* dst, const float* src, size_t dataSize, bool isMultithread = true );
	void Sigmoid( float* dst, const float* src, size_t dataSize, bool isMultithread = true );
	void Exp( float* dst, const float* src, size_t dataSize, bool isMultithread = true );

	// Process part of lstm layer which follow after fullyconnected layers.
	void RestOfLstm( CMathEngineLstmDesc* desc, const CConstFloatHandle& inputStateBackLink,
		const CFloatHandle& outputStateBackLink, const CFloatHandle& outputMainBackLink,
		bool isMultithread );

private:
	enum class TPrimitive {
		Tanh,
		Sigmoid,
		Exp,
		RestOfLstm,

		Count
	};

	enum class TTableKey {
		// Tanh specific items
		TanhPolyCoeff, // Coefficients of tanh polynome
		TanhIdxBias, // 0x1.0p * 2^-12; Bias is applied to x in order to obtain correct interval for polynomial calculation
		TanhIdxMaskShifted, // This mask aims to obtain correct index of interval for polynomial calculation
		TanhIdxMask, // Applied to extract index
		TanhLineralUBound, // Below this 'x' tanh(x) = x
		TanhSaturationLBound, // Above this 'x' tanh(x) = 1.f

		// Common Items
		Ln2f, // 0.69314718f
		PositiveMask, // Extract value without sign
		Half, // 0.5f
		One, // 1.0f
		Two, // 2.0f
		SignMask, // Extract sign from valiable
		ExpBias, // (127 = 2^7 - 1), gets exponent bits

		LoadMask, // Load 1-7 floats with vmaskmovps instruction

		ExpLog2ef, // 1.44269502f
		ExpFltMax, // logf(FLT_MAX) - max normal value
		ExpFltMin, // logf(FLT_MIN) - min normal value
		ExpPolyCoeff  // Coefficients of exp polynome
	};
	
	struct CGenerator {
		CJitCommon gen;
		std::mutex lock;
	};

	static constexpr int MantissaNumBits = 23;

	using ActivationFunc = void( * )( float* dst, const float* src, size_t offset, size_t count );
	using RestOfLstmFunc = void( * )( size_t hiddenSize, const float* inputStateBackLinkPtr, float* outputStateBackLinkPtr,
		float* outputMainBackLinkPtr, float* inputFullyConnectedResultPtr, float* reccurentFullyConnectedResultPtr, size_t offset, size_t count );

	IMathEngine* const mathEngine;
	const int threadCount = 1; /*deprecated*/

	// Contains jit code generators for partial primitives
	std::array<CGenerator, static_cast<size_t>( TPrimitive::Count )> gens;
	// Table for storing of constans and map for matching table keys to it offsets 
	std::vector<uint32_t> table;
	std::unordered_map<TTableKey, size_t> tableOffsets;

	Xbyak::Reg64 regTablePtr = Xbyak::util::r10;
		
	// Functions for handling table
	void initTable();
	// Get offset in bytes of specific key in table ( optionaly with offset in floats )
	uint32_t getOfft( TTableKey key, uint32_t offset = 0 ) const;
	// Get address of field ( optionaly with offset in floats )
	Xbyak::Address getAddr( TTableKey key, uint32_t offset = 0 ) const;
	// Add vector or value to the table and append appropriate table key to the tableOffsets
	void addVector( TTableKey key, std::initializer_list<uint32_t>&& data, size_t repeatNum = 1 );
	// repeatNum specifies how many times value will be repeated in the table
	void addVal( TTableKey key, uint32_t val, size_t repeatNum = NumFloatInYmm );

	template<TPrimitive P>
	void initPrimitive();
	template<TPrimitive P>
	void initActivationFunction( const std::function<void()>& afterPrologue,
		const reg64Vec_t& preservedGPR, const ymmVec_t& preservedYmm,
		const ymmVec_t& ymmSrc, const ymmVec_t& ymmAux );

	// Function for inserting one primitives into another ones
	// gen - jit generator which is used in paren function.
	// ymmSrc - inplace updated src data (can't be one of ymmAux registers!) (types: ymm_t or ymmVec_t)
	// ymmAux - auxiliary registers which will be used inside function
	template<TPrimitive P>
	void insertPrimitive( CJitCommon& gen, const ymmVec_t& ymmSrc, const ymmVec_t& ymmAux );

	template<TPrimitive P, class PrimitiveFuncType, class... Args>
	void callPrimitive( size_t dataSize, bool isMultithread, Args... args );

	// Check if two arrays have insersected registers and each array contains only unique registers
	template<class RegType, class ArrayType0, class ArrayType1>
	bool isRegArraysIntersected( const ArrayType0& arr0, const ArrayType1& arr1 );

	// Function helps to parse raw aux vector to the small slices
	ymmVec_t initFromAux( int idx, const ymmVec_t& ymmSrc, const ymmVec_t& ymmAux ) {
		const size_t SrcSize = ymmSrc.size();
		auto begin = ymmAux.begin() + idx * SrcSize;
		return ymmVec_t( begin, begin + SrcSize );
	};

	template<class RegType>
	std::vector<RegType> initVecRange( int firstIdx, int lastIdx ) {
		const int VecSize = lastIdx - firstIdx + 1;
		assert( VecSize > 0 );
		assert( firstIdx >= 0 && lastIdx < 16 );
		std::vector<RegType> ret( VecSize );
		int idx = firstIdx;
		for( auto& v : ret ) {
			v = RegType( idx++ );
		}
		return ret;
	};

};

} // namespace NeoML
