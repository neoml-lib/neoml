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

#include <common.h>
#pragma hdrstop

#include <MathEngineDnnLstm.h>
#include <JitCommon.h>
#include <unordered_map>
#include <array>
#include <mutex>
#include <functional>

namespace NeoML {

struct CLstmDesc;
class IMathEngine;

class CPrimitivesJit {
public:
	CPrimitivesJit( IMathEngine* _mathEngine, int _threadCount );

	void Tanh( float* dst, const float* src, size_t dataSize, bool isMultithread = true );
	void Sigmoid( float* dst, const float* src, size_t dataSize, bool isMultithread = true );
	void Exp( float* dst, const float* src, size_t dataSize, bool isMultithread = true );

	void RestOfLstm( CLstmDesc* desc, const CConstFloatHandle& inputStateBackLink,
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
		TanhPolyCoeff,
		TanhIdxBias,
		TanhIdxMaskShifted,
		TanhIdxMask,
		TanhLineralUBound,
		TanhSaturationLBound,

		// Common Items
		Ln2f,
		PositiveMask,
		Half,
		One,
		Two,
		SignMask,
		ExpBias,

		LoadMask,

		ExpLog2ef,
		ExpFltMax,
		ExpFltMin,
		ExpPolyCoeff
	};
	
	struct CGenerator {
		CJitCommon gen;
		std::mutex lock;
	};

	static constexpr int MantissaNumBits = 23;

	using ActivationFunc = void( * )( float* dst, const float* src, size_t offset, size_t count );
	using RestOfLstmFunc = void( * )( size_t hiddenSize, const float* inputStateBackLinkPtr, float* outputStateBackLinkPtr,
		float* outputMainBackLinkPtr, float* inputFullyConnectedResultPtr, float* reccurentFullyConnectedResultPtr, size_t offset, size_t count );

	IMathEngine* mathEngine;
	int threadCount;

	// Contains jit code generators for partial primitives
	std::array<CGenerator, static_cast<size_t>( TPrimitive::Count )> gens;
	// Table for storing of constans and map for matching table keys to it offsets 
	std::vector<uint32_t> table;
	std::unordered_map<TTableKey, size_t> tableOffsets;

	Xbyak::Reg64 regTablePtr = Xbyak::util::r10;
		
	// Functions for handling table
	void initTable();
	uint32_t getOfft( TTableKey key, uint32_t offset = 0 ) const;
	Xbyak::Address getAddr( TTableKey key, uint32_t offset = 0 ) const;
	void addVector( TTableKey key, std::initializer_list<uint32_t>&& data, size_t repeatNum = 1 );
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
		const int SrcSize = ymmSrc.size();
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
