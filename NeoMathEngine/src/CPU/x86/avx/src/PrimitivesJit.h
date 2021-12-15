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

namespace NeoML {

struct CLstmDesc;
class IMathEngine;

class CPrimitivesJit {
public:
	CPrimitivesJit( IMathEngine* _mathEngine, int _threadCount );

	void Tanh( float* dst, const float* src, size_t dataSize, bool isMultithread = true );
	void CalcSigmoid( float* dst, const float* src, size_t dataSize, bool isMultithread = true );
	void RestOfLstm( CLstmDesc* desc, const CConstFloatHandle& inputStateBackLink,
		const CFloatHandle& outputStateBackLink, const CFloatHandle& outputMainBackLink );

private:
	enum class TPrimitive {
		Tanh,
		Sigmoid,
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
		PositiveMask,
		One,
		SignMask,

		LoadMask

	};
	
	struct CGenerator {
		CJitCommon gen;
		std::mutex lock;
	};

	using TanhFunc = void( * )( float*, const float*, size_t );

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
	Xbyak::Address getAddr( TTableKey key ) const;
	void addVector( TTableKey key, std::initializer_list<uint32_t>&& data );
	void addVal( TTableKey key, uint32_t val, size_t repreatNum = NumFloatInYmm );

	TanhFunc initTanh();

	// Function for inserting one primitives into another ones
	// Insert code of tanh function into another code
	// gen - jit generator which is used in paren function.
	// ymmAux - auxiliary registers which will be used inside function
	// ymmData - inplace updated src data (can't be one of ymmAux registers!)
	void insertTanh( CJitCommon& gen, std::vector<Xbyak::Ymm>&& ymmAux, Xbyak::Ymm ymmData );

	// Check if two arrays have insersected registers and each array contains only unique registers
	// TODO: remove
	template<int AuxSize, class RegType>
	bool isIntersected( const std::array<RegType, AuxSize>& arr0, const std::vector<RegType>& arr1 );

};

} // namespace NeoML
