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

#include <common.h>
#pragma hdrstop

#include <algorithm>
#include <PrimitivesJit.h>
#include <MemoryHandleInternal.h>

namespace NeoML {

CPrimitivesJit::CPrimitivesJit( IMathEngine* _mathEngine, int _threadCount ) :
	mathEngine( _mathEngine ), threadCount( _threadCount )
{
	initTable();
}

void CPrimitivesJit::Tanh( float* dst, const float* src, size_t dataSize, bool isMultithread )
{
	TanhFunc tanh = initTanh();
	const int curThreadCount = isMultithread && IsOmpRelevant( static_cast< int >( dataSize ) ) ? threadCount : 1;
	if( curThreadCount != 1 ) {
		NEOML_OMP_NUM_THREADS( curThreadCount ) {
			int offt, count;
			if( OmpGetTaskIndexAndCount( static_cast< int >( dataSize ), offt, count ) ) {
				tanh( dst + offt, src + offt, count );
			}
		}
	} else {
		tanh( dst, src, dataSize );
	}
}

void CPrimitivesJit::CalcSigmoid( float* dst, const float* src, size_t dataSize, bool isMultithread )
{}

void CPrimitivesJit::RestOfLstm( CLstmDesc* desc, const CConstFloatHandle& inputStateBackLink,
	const CFloatHandle& outputStateBackLink, const CFloatHandle& outputMainBackLink )
{}

void CPrimitivesJit::initTable()
{
    // tanh(x) polynomial approximation
    // For each coefficient, there is 32 entries
	addVector( TTableKey::TanhPolyCoeff, {
		// coefficients of degree 0
		0x00000000, 0x39bfffff, 0x39ffffff, 0x3a3ffffe, 0x3a7ffffb, 0x3abffff7, 0x3affffeb, 0x3b3fffdc,
		0x3b7fffab, 0x3bbfff70, 0x3bfffeab, 0x3c3ffdc0, 0x3c7ffaab, 0x3cbff701, 0x3cffeaad, 0x3d3fdc08,
		0x3d7faacd, 0x3dbf7081, 0x3dfeacc9, 0x3e3dc7fd, 0x3e7acbf5, 0x3eb77a9f, 0x3eec9a9f, 0x3f22991f,
		0x3f42f7d6, 0x3f67b7cc, 0x3f76ca83, 0x3f7ebbe9, 0x3f7fd40c, 0x3f7fff32, 0x3f7ffffc, 0x3f800000,
		// coefficients of degree 1
		0x3f800000, 0x3f800018, 0x3f7fffe8, 0x3f7fffda, 0x3f7fffdc, 0x3f7fffdc, 0x3f7fffac, 0x3f7fff70,
		0x3f7ffeec, 0x3f7ffdc0, 0x3f7ffbed, 0x3f7ff704, 0x3f7feff5, 0x3f7fdbca, 0x3f7fbfff, 0x3f7f7041,
		0x3f7f009b, 0x3f7dc36c, 0x3f7c0aa8, 0x3f7734b8, 0x3f70a4de, 0x3f5f1fd8, 0x3f495493, 0x3f18b9ec,
		0x3ed706cb, 0x3e390b06, 0x3d90b11f, 0x3c21a053, 0x3aaf7fdb, 0x37ccc1a3, 0x355c6733, 0x00000000,
		// coefficients of degree 2
		0x00000000, 0xbe4e0ff1, 0x3d25b1b1, 0x3d6b6dab, 0x3c9fb1d5, 0xbabff06f, 0x3c07b3f6, 0xbb3fc1bc,
		0x3a9f5921, 0xbbbf06f2, 0xbbb0f402, 0xbc47db9e, 0xbc73d5e7, 0xbca25bda, 0xbcfca780, 0xbd40e07c,
		0xbd7dab03, 0xbdbe4a0f, 0xbdfb14a5, 0xbe36cc8d, 0xbe6bd102, 0xbe9fe7c5, 0xbeba0f10, 0xbec206a8,
		0xbea3c388, 0xbe277d62, 0xbd8b7960, 0xbc209f49, 0xbaad44ca, 0xb7c6eeac, 0xb663aa41, 0x00000000,
		// coefficients of degree 3
		0x00000000, 0x45b3ae96, 0xc414eb20, 0xc450e02e, 0xc3152b4e, 0xbead2f56, 0xc2162e02, 0xbeb4bd5a,
		0xc11a59a4, 0xbed2f507, 0xc020d32c, 0x3dd0f506, 0xbf2a75e2, 0xbff950e3, 0xbed47334, 0xbe809b8c,
		0xbeb64532, 0xbe961a5b, 0xbe9b63ac, 0xbea0d4b2, 0xbe828a77, 0xbe378612, 0xbdc20908, 0x3d2d3957,
		0x3dd46e89, 0x3db3f629, 0x3d2c5e7b, 0x3bd20403, 0x3a59dfae, 0x3770af45, 0x372cc014, 0x00000000,
		// coefficients of degree 4
		0x00000000, 0xcc981a1b, 0x4a7edd3d, 0x4ab1007c, 0x48fedd9c, 0x41a557b5, 0x477ee32a, 0x422557f5,
		0x45ff3ce4, 0x42a55641, 0x446e0867, 0xc33dc19a, 0x42915214, 0x43af4fad, 0x4110fe88, 0xc1099b75,
		0x3fc8a8dc, 0xbfbeaef5, 0xbe365aad, 0x3f4d9652, 0x3ddfa08f, 0x3e34e9b8, 0x3e2d07a6, 0x3dc63567,
		0x3cdaeb78, 0xbcd17537, 0xbc92829c, 0xbb43ab99, 0xb9b471dd, 0xb6baad5a, 0xb78bafc7, 0x00000000,
		// coefficients of degree 5
		0x00000000, 0x52f688d5, 0xd0505c72, 0xd08f98e3, 0xce505cc9, 0xc7162b8a, 0xcc5061d6, 0xc7162bdf,
		0xca50b37f, 0xc7162a3a, 0xc8422086, 0x471a714e, 0xc5ece1f1, 0xc70e3d90, 0xc3eba94a, 0x43e0c424,
		0xc21f4552, 0x42217cc8, 0x405e7dc4, 0xc10dd401, 0x3e96b602, 0xbd1a6d2f, 0xbd393883, 0xbd674682,
		0xbd310016, 0xb961e269, 0x3ba32495, 0x3a7680d5, 0x38b3173c, 0x35a9deea, 0x375c3f2a, 0x00000000,
		// coefficients of degree 6
		0x00000000, 0xd8995ed1, 0x558285ea, 0x55b2cd69, 0x53028625, 0x4bc9991f, 0x5082898a, 0x4b4999b3,
		0x4e02c07c, 0x4ac99764, 0x4b72c822, 0xca40c0e1, 0x489413e4, 0x49b12224, 0x46134c4e, 0xc60c2d57,
		0x43c83910, 0xc3c872d1, 0xc186bc9e, 0x42325bc3, 0xbf2ffa4a, 0x3d9a203c, 0xbc545a43, 0xbae08fee,
		0x3c80225d, 0x3b1fd1df, 0xba36b9d1, 0xb91de544, 0xb71f100f, 0xb408e2ed, 0xb685fec8, 0x00000000
		} );
    addVal( TTableKey::TanhIdxBias, 0x39800000 );
    addVal( TTableKey::TanhIdxMaskShifted, 0x0000001f );
    addVal( TTableKey::TanhIdxMask, 0xffc00000 );
    addVal( TTableKey::TanhLineralUBound, 0x39ddb3d7 );
    addVal( TTableKey::TanhSaturationLBound, 0x41102cb3 );

    addVal( TTableKey::PositiveMask, 0x7fffffff );
    addVal( TTableKey::One, 0x3f800000 );
    addVal( TTableKey::SignMask, 0x80000000 );
	addVector( TTableKey::LoadMask, {
		0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
		0xffffffff, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
		0xffffffff, 0xffffffff, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
		0xffffffff, 0xffffffff, 0xffffffff, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
		0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
		0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x00000000, 0x00000000, 0x00000000,
		0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x00000000, 0x00000000,
		0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x00000000
		} );
}

uint32_t CPrimitivesJit::getOfft( TTableKey key, uint32_t offset ) const
{
	auto it = tableOffsets.find( key );
	assert( it != tableOffsets.end() );
    return ( it->second + offset ) * sizeof( decltype( table )::value_type );
}

Xbyak::Address CPrimitivesJit::getAddr( TTableKey key ) const
{
	return Xbyak::util::ptr[regTablePtr + getOfft( key )];
}

void CPrimitivesJit::addVector( TTableKey key, std::initializer_list<uint32_t>&& data )
{
	assert( tableOffsets.find( key ) == tableOffsets.end() );
	size_t currentSize = table.size();
	tableOffsets.emplace( key, currentSize );
	table.resize( currentSize + data.size() );
	uint32_t* pTable = &table[currentSize];
	std::copy( data.begin(), data.end(), pTable );
}

void CPrimitivesJit::addVal( TTableKey key, uint32_t val, size_t repeatNum )
{
	assert( tableOffsets.find( key ) == tableOffsets.end() );
	size_t currentSize = table.size();
	tableOffsets.emplace( key, currentSize );
	table.resize( currentSize + repeatNum );
	uint32_t* pTable = &table[currentSize];
	std::fill_n( pTable, repeatNum, val );
}

CPrimitivesJit::TanhFunc CPrimitivesJit::initTanh()
{
	using namespace Xbyak::util;

	CGenerator& genInst = gens[static_cast< size_t >( TPrimitive::Tanh )];
	lock_guard<mutex> lock( genInst.lock );
	if( genInst.gen.getSize() != 0 ) {
		return genInst.gen.getCode<TanhFunc>();
	}

	// create new instance
	auto& gen = genInst.gen;

	const reg64_t regDstPtr = Param1;
	const reg64_t regSrcPtr = Param2;
	const reg64_t regCountParam = Param3;
	// TODO: Check for Linux
	reg64_t regCount = r11;
	// We could use regCount in SIB addressing when we process tail.
	// For linux we can't use simultaniously r10(regTablePtr) and RDX(param3)
	gen.mov( regCount, regCountParam );
	const vector<ymm_t> preservedYmm = { ymm6, ymm7, ymm8 };

	gen.Prologue( {}, preservedYmm );

	gen.mov( regTablePtr, ( uint64_t )table.data() );

	auto insertCode = [&]( int stepCount, int firstYmmIdx ) {
		assert( ( firstYmmIdx + stepCount ) < MaxYmmCount );
		gen.StartDownCountLoop( regCount, stepCount * NumFloatInYmm );
		for( int i = 0; i < stepCount; i++ ) { gen.vmovups( ymm_t( firstYmmIdx + i ), ptr[regSrcPtr + i * SizeOfYmm] ); }
		for( int i = 0; i < stepCount; i++ ) { insertTanh( gen, { ymm0, ymm1, ymm2, ymm3, ymm4 }, ymm_t( firstYmmIdx + i ) ); }
		for( int i = 0; i < stepCount; i++ ) { gen.vmovups( ptr[regDstPtr + i * SizeOfYmm], ymm_t( firstYmmIdx + i ) ); }
		gen.lea( regSrcPtr, gen.ptr[regSrcPtr + stepCount * SizeOfYmm] );
		gen.lea( regDstPtr, gen.ptr[regDstPtr + stepCount * SizeOfYmm] );
		gen.StopDownCountLoop();
	};
	// 1. Unrolled batch processing (step == 4, first register for data ymm5 )
	insertCode( 4, 5 );
	// 2. Single processing (step == 1, first register for data ymm5 )
	insertCode( 1, 5 );
	// 3. Process tail of array
	gen.cmp( regCount, 0 );
	gen.jz( "end", gen.T_NEAR );
	ymm_t ymmMask = ymm5;
	ymm_t ymmSrc = ymm6;
	// Multiply by 8 for calculate right offset
	gen.shl( regCount, 3 );
	gen.vmovups( ymmMask, gen.ptr[regTablePtr + regCount * sizeof( float ) + getOfft( TTableKey::LoadMask )] );
	gen.vmaskmovps( ymmSrc, ymmMask, gen.ptr[regSrcPtr] );
	insertTanh( gen, { ymm0, ymm1, ymm2, ymm3, ymm4 }, ymmSrc );
	gen.vmaskmovps( gen.ptr[regDstPtr], ymmMask, ymmSrc );
	gen.L( "end" );

	gen.Epilogue( {}, preservedYmm );
	gen.ret();

	// call function
	return genInst.gen.getCode<TanhFunc>();
}

void CPrimitivesJit::insertTanh( CJitCommon& gen, vector<ymm_t>&& ymmAux, ymm_t ymmData )
{
	assert( !any_of( ymmAux.begin(), ymmAux.end(), [=]( ymm_t& reg )->bool { return reg.getIdx() == ymmData.getIdx(); } ) );
	assert( ymmAux.size() == 5 );

	const int PolynomialNum = 32;
	const int PolynomialDegree = 6;
	ymm_t ymmMask = ymmAux[0],
		ymmDst = ymmAux[1], ymmSrcShift = ymmAux[1], ymmCoeff = ymmAux[1],
		ymmPol = ymmAux[2],
		ymmIdx = ymmAux[3],
		ymmSrcOrig = ymmAux[4], ymmSign = ymmAux[4];
	ymm_t& ymmSrc = ymmData;
	// We split the positive domain in 33 intervals:
	// a) [0; linear_ubound]: in this interval tanh(x) = x
	// b) [linear_ubound; 0x1.8p-12]: This interval spans part of a
	//    half binade
	// c) [0x1.8p-12; 0x1.0p-11], ..., [0x1.8p2; 0x1.0p3]:
	//    one interval for each half binade, there are 29 of those
	// d) [0x1.0p3; saturation_ubound]:
	//    This interval spans part of a half binade
	// e) [0x1.205966p3; saturation_ubound]: in this interval, tanh(x) = 1
	// For b-d, we need 31 polynomials and will do a table lookup for those.
	// To simplify the logic, we will also put a) in the table.

	// The polynomials are of degree 6, so we need to gather 7 coefficients.
	// - sse4.1: we do it the naive way using vextract/vinsert.
	//           Here we will extract the indices in gpr only once and
	//           reuse them as there are only 4 of them.
	// - avx: we do the same as for sse4.1 but use half of the 64-bits
	//           registers to store the idx of second half of YMM and half for
	//           responding XMM. Halfway through the copy we exchange Xmm and
	//           higher half of ymm_t and we get the expected result.
	// - avx2: we use vpermps and blend for each coefficient.
	//         This needs an extra vmm to store the mask
	// - avx512: because the table fits in 2 registers, we can use vpermi2d.
	auto coeffs_off = [&]( int coeff_off )->size_t {
		return getOfft( TTableKey::TanhPolyCoeff, coeff_off * PolynomialNum );
	};
	auto gather_coefficient_init = [&]( ymm_t vmm_pol_idx, int nelems ) {
		// needed for gather instruction
		gen.vxorps( ymmMask, ymmMask, ymmMask );
	};
	auto gather_coefficient = [&]( ymm_t ymmCoeff, int coeff_idx,
		ymm_t vmm_pol_idx ) {
			Xbyak::Address idx_addr = gen.ptr[regTablePtr + coeffs_off( coeff_idx )
				+ vmm_pol_idx * sizeof( float )];
			// we set the mask to all ones to gather full
			// register.  needs to be done after each gather since
			// since the gather instructions zeros the mask if
			// successful
			gen.vcmpeqps( ymmMask, ymmMask, ymmMask );
			gen.vgatherdps( ymmCoeff, idx_addr, ymmMask );
	};

	// because tanh(x) = -tanh(-x), we extract sign to make x postive
	// and reapply sign at the end
	gen.vmovups( ymmSrcOrig, ymmSrc );
	gen.vandps( ymmSrc, ymmSrc, getAddr( TTableKey::PositiveMask ) );

	// We compute the indices for the table lookup
	gen.vmovups( ymmIdx, ymmSrc );
	gen.vpsubd( ymmIdx, ymmIdx, getAddr( TTableKey::TanhIdxBias ) );
	gen.vpsrld( ymmIdx, ymmIdx, 22 );
	gen.vandps( ymmIdx, ymmIdx, getAddr( TTableKey::TanhIdxMaskShifted ) );

	// we do the argument reduction
	gen.vmovups( ymmSrcShift, ymmSrc );
	gen.vandps( ymmSrcShift, ymmSrcShift, getAddr( TTableKey::TanhIdxMask ) );
	gen.vsubps( ymmSrc, ymmSrc, ymmSrcShift );

	// we gather and evaluate the polynonials
	gather_coefficient_init( ymmIdx, NumFloatInYmm );
	gather_coefficient( ymmPol, PolynomialDegree, ymmIdx );
	for( int deg = PolynomialDegree - 1; deg >= 0; --deg ) {
		gather_coefficient( ymmCoeff, deg, ymmIdx );
		gen.vfmadd213ps( ymmPol, ymmSrc, ymmCoeff );
	}

	// we restore src with cleared sign, and keep sign
	assert( ymmSign.getIdx() == ymmSrcOrig.getIdx() );
	gen.vmovups( ymmSrc, ymmSrcOrig );
	gen.vandps( ymmSign, ymmSign, getAddr( TTableKey::SignMask ) );
	gen.vandps( ymmSrc, ymmSrc, getAddr( TTableKey::PositiveMask ) );

	// Now we blend the results
	// [saturation_ubound; +inf[ : we return +/- 1
	gen.vmovups( ymmDst, getAddr( TTableKey::One ) );
	// [linear_ubound; saturation_lbound] : we return +/- P(x)
	gen.vmovups( ymmMask, getAddr( TTableKey::TanhSaturationLBound ) );
	gen.vcmpnltps( ymmMask, ymmMask, ymmSrc );
	gen.vblendvps( ymmDst, ymmDst, ymmPol, ymmMask );
	// [0; linear_ubound]  : we return x
	gen.vmovups( ymmMask, getAddr( TTableKey::TanhLineralUBound ) );
	gen.vcmpnltps( ymmMask, ymmMask, ymmSrc );
	gen.vblendvps( ymmDst, ymmDst, ymmSrc, ymmMask );

	// We reapply the sign and return
	gen.vxorps( ymmDst, ymmDst, ymmSign );
	gen.vmovups( ymmSrc, ymmDst );
}

template<int AuxSize, class RegType>
bool CPrimitivesJit::isIntersected( const std::array<RegType, AuxSize>& arr0, const std::vector<RegType>& arr1 )
{
	// ymm_t and GPR has up to 16 registers
	const int MaxRegNum = 16;
	int isAlreadyExists[MaxRegNum] = { 0, };
	for_each( arr0.cbegin(), arr0.cend(), []( const Reg& Type reg ) { isAlreadyExists[reg.idx]++; } );
	for_each( arr1.cbegin(), arr1.cend(), []( const Reg& Type reg ) { isAlreadyExists[reg.idx]++; } );
	int* it = isAlreadyExists;
	for( int* it = isAlreadyExists; it < &isAlreadyExists[MaxRegNum]; it++ ) {
		if( *it > 1 ) {
			// There is an intersection
			return true;
		}
	}
	return false;
}

} // namespace NeoML
