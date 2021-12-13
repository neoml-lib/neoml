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
#include <map>

namespace NeoML {

struct CLstmDescJit : public CMathEngineLstmDesc {

	CLstmDescJit( const CFloatHandle& _inputWeights, const CFloatHandle* _inputFreeTerm,
		const CFloatHandle& _recurrentWeights, const CFloatHandle* _recurrentFreeTerm,
		const CFloatHandle& _inputFullyConnectedResult, const CFloatHandle& _reccurentFullyConnectedResult,
		int _hiddenSize, int _objectCount, int _objectSize, IMathEngine* _mathEngine, int _threadCount );

	void SimdRunOnceRestOfLstm( const CConstFloatHandle& inputStateBackLink,
		const CFloatHandle& outputStateBackLink, const CFloatHandle& outputMainBackLink ) override;
	void CalcTanh( float* data, size_t dataSize ) override;

private:
	class CJitLstm : public Xbyak::CodeGenerator {
	public:
		CJitLstm( CLstmDescJit& ld );

	private:
		enum class TTableKey {
			Coeff,
			TanhIdxBias,
			TanhIdxMask31,
			TanhIdxMask,
			TanhLineralUBound,
			TanhSaturationLBound,
			PositiveMask,
			One,
			SignMask
		};
		map<TTableKey, size_t> tableOffsets;
		vector<uint32_t> table;

		// Passed to 'Run()' function as arguments
		const reg64_t regDataPtr = Param1;
		const reg64_t regDataCount = Param2;
		const reg64_t regTablePtr = Param3; // not param

		void prologue();
		void epilogue();
		void labelAlign( Xbyak::Label& label, int alignment = 16 ) {
			align( alignment );
			L( label );
		}

		void calcTanh( const Xbyak::Ymm& src );

		void initTable();
		size_t getOfft( TTableKey key, size_t offset = 0 );
		Xbyak::Address getAddr( TTableKey key );
	};

	CJitLstm jitLstmCode;
};

} // namespace NeoML
