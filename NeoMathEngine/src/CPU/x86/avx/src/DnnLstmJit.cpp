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

#include <DnnLstmJit.h>
#include <MemoryHandleInternal.h>

namespace NeoML {

CLstmDescJit::CLstmDescJit( const CFloatHandle& _inputWeights, const CFloatHandle* _inputFreeTerm,
	const CFloatHandle& _recurrentWeights, const CFloatHandle* _recurrentFreeTerm,
	const CFloatHandle& _inputFullyConnectedResult, const CFloatHandle& _reccurentFullyConnectedResult,
	int _hiddenSize, int _objectCount, int _objectSize, IMathEngine* _mathEngine, int _threadCount ) :
	CMathEngineLstmDesc( _inputWeights, _inputFreeTerm, _recurrentWeights, _recurrentFreeTerm,
		_inputFullyConnectedResult, _reccurentFullyConnectedResult, _hiddenSize, _objectCount, _objectSize,
        _mathEngine, _threadCount ),
	jitLstmCode( *this )
{
}

void CLstmDescJit::SimdRunOnceRestOfLstm( const CConstFloatHandle& inputStateBackLink,
    const CFloatHandle& outputStateBackLink, const CFloatHandle& outputMainBackLink )
{
    const int BacklinkObjectSize = objectCount * hiddenSize;
    const int curThreadCount = IsOmpRelevant( BacklinkObjectSize ) ? threadCount : 1;
    NEOML_OMP_NUM_THREADS( curThreadCount )
    {
    }
}

void CLstmDescJit::CalcTanh( float* data, size_t dataSize )
{
    const int BacklinkObjectSize = dataSize;
    const int curThreadCount = IsOmpRelevant( BacklinkObjectSize ) ? threadCount : 1;
    NEOML_OMP_NUM_THREADS( curThreadCount )
    {
        int dataIdx;
        int dataCount;
        if( OmpGetTaskIndexAndCount( BacklinkObjectSize, dataIdx, dataCount ) ) {
            jitLstmCode.getCode<void( * )( float*, size_t )>()( data + dataIdx, dataCount );
        }
    }
}

CLstmDescJit::CJitLstm::CJitLstm( CLstmDescJit& ld ) {
    using namespace Xbyak;

    // TODO: Initialize globally
    initTable();

    prologue();

    int SmallBatchSize = 1;
    const int BigBatchSize = SmallBatchSize * 8;

    Label labelStartBigBatch, labelEndBigBatch;

    L( labelStartBigBatch );
    cmp( regDataCount, BigBatchSize );
    jl( labelEndBigBatch, T_NEAR );
    vmovups( ymm6, ptr[regDataPtr + 0 * 8 * sizeof( float )] );/*
    vmovups( ymm7, ptr[regDataPtr + 1 * 8 * sizeof( float )] );
    vmovups( ymm8, ptr[regDataPtr + 2 * 8 * sizeof( float )] );
    vmovups( ymm9, ptr[regDataPtr + 3 * 8 * sizeof( float )] );
    vmovups( ymm10, ptr[regDataPtr + 4 * 8 * sizeof( float )] );
    vmovups( ymm11, ptr[regDataPtr + 5 * 8 * sizeof( float )] );
    vmovups( ymm12, ptr[regDataPtr + 6 * 8 * sizeof( float )] );
    vmovups( ymm13, ptr[regDataPtr + 7 * 8 * sizeof( float )] );*/
    calcTanh( ymm6 );/*
    calcTanh( ymm7 );
    calcTanh( ymm8 );
    calcTanh( ymm9 );
    calcTanh( ymm10 );
    calcTanh( ymm11 );
    calcTanh( ymm12 );
    calcTanh( ymm13 );*/
    vmovups( ptr[regDataPtr + 0 * 8 * sizeof( float )], ymm6 );/*
    vmovups( ptr[regDataPtr + 1 * 8 * sizeof( float )], ymm7 );
    vmovups( ptr[regDataPtr + 2 * 8 * sizeof( float )], ymm8 );
    vmovups( ptr[regDataPtr + 3 * 8 * sizeof( float )], ymm9 );
    vmovups( ptr[regDataPtr + 4 * 8 * sizeof( float )], ymm10 );
    vmovups( ptr[regDataPtr + 5 * 8 * sizeof( float )], ymm11 );
    vmovups( ptr[regDataPtr + 6 * 8 * sizeof( float )], ymm12 );
    vmovups( ptr[regDataPtr + 7 * 8 * sizeof( float )], ymm13 );*/
    sub( regDataCount, BigBatchSize );
    lea( regDataPtr, ptr[regDataPtr + BigBatchSize * sizeof( float )] );
    jmp( labelStartBigBatch, T_NEAR );
    L( labelEndBigBatch );
    epilogue();

    // !!! This instruction should always be called at the end of AVX code.
    // Intel® 64 and IA - 32 Architectures Optimization Reference Manual, item 11.3.1
    // Assembly / Compiler Coding Rule 72. ( H impact, H generality ) Add VZEROUPPER instruction after
    // 256 - bit AVX instructions are executed and before any function call that might execute SSE code.Add
    // VZEROUPPER at the end of any function that uses 256 - bit AVX instructions.
    vzeroupper();
    ret();
}

void CLstmDescJit::CJitLstm::prologue()
{
    using namespace Xbyak::util;
    using namespace Xbyak;
    push( rbp );
    mov( rbp, rsp );

    // stack should be alinged to 16 byte because we use vmovaps instruction
    for( int i = 6; i <= 15; i++ ) {
        // '-16' - place for first xmm
        vmovaps( ptr[rsp - 16 - ( i - 6 ) * 16], Xmm( i ) );
    }

    mov( regTablePtr, (uint64_t)( table.data() ) );
}

void CLstmDescJit::CJitLstm::epilogue()
{
    using namespace Xbyak::util;
    using namespace Xbyak;
    // stack should be alinged to 16 byte because we use vmovaps instruction

    for( int i = 15; i >= 6; i-- ) {
        vmovaps( Xmm( i ), ptr[rsp - 16 - ( i - 6 ) * 16] );
    }
    leave();
}

void CLstmDescJit::CJitLstm::calcTanh( const Xbyak::Ymm& vmm_src )
{
    using namespace Xbyak;
    using namespace Xbyak::util;
    using Vmm = Xbyak::Ymm;

    const int XMM_float_lanes_count = 4;
    const int tanh_n_polynomials = 32;

    const unsigned int _cmp_eq_oq = 0u;
    const unsigned int _cmp_gt_os = 6u;
    const int vlen = 32;

    Vmm vmm_mask = ymm0;
    Vmm vmm_aux0 = ymm0;
    Vmm vmm_aux1 = ymm2;
    Vmm vmm_aux2 = ymm3;
    Vmm vmm_aux3 = ymm4;
    Vmm vmm_aux4 = ymm5;
    // register mapping
    // TODO: put sign on stack and alias zmm_table2 with vmm_sign to save a reg ?
    Vmm vmm_dst = vmm_aux1, vmm_src_shift = vmm_aux1, vmm_coeff = vmm_aux1,
        vmm_pol = vmm_aux2, vmm_indices = vmm_aux3, vmm_src_original = vmm_aux4,
        vmm_sign = vmm_aux4;

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
    //           higher half of Ymm and we get the expected result.
    // - avx2: we use vpermps and blend for each coefficient.
    //         This needs an extra vmm to store the mask
    // - avx512: because the table fits in 2 registers, we can use vpermi2d.
    auto coeffs_off = [&]( int coeff_off )->size_t {
        return getOfft( TTableKey::Coeff, coeff_off * tanh_n_polynomials );
    };
    auto gather_coefficient_init = [&]( Vmm vmm_pol_idx, int nelems ) {
            // needed for gather instruction
            vxorps( vmm_mask, vmm_mask, vmm_mask );
    };
    auto gather_coefficient = [&]( Vmm vmm_coeff, int coeff_idx,
        Vmm vmm_pol_idx ) {
            Xbyak::Address idx_addr = ptr[regTablePtr + coeffs_off( coeff_idx )
                + vmm_pol_idx * sizeof( float )];
            // we set the mask to all ones to gather full
            // register.  needs to be done after each gather since
            // since the gather instructions zeros the mask if
            // successful
            vcmpps( vmm_mask, vmm_mask, vmm_mask, _cmp_eq_oq );
            vgatherdps( vmm_coeff, idx_addr, vmm_mask );
    };

    // because tanh(x) = -tanh(-x), we extract sign to make x postive
    // and reapply sign at the end
    vmovups( vmm_src_original, vmm_src );
    vandps( vmm_src, vmm_src, getAddr( TTableKey::PositiveMask ) );

    // We compute the indices for the table lookup
    vmovups( vmm_indices, vmm_src );
    vpsubd( vmm_indices, vmm_indices, getAddr( TTableKey::TanhIdxBias ) );
    vpsrld( vmm_indices, vmm_indices, 22 );
    vandps( vmm_indices, vmm_indices, getAddr( TTableKey::TanhIdxMask31 ) );

    // we do the argument reduction
    vmovups( vmm_src_shift, vmm_src );
    vandps( vmm_src_shift, vmm_src_shift, getAddr( TTableKey::TanhIdxMask ) );
    vsubps( vmm_src, vmm_src, vmm_src_shift );

    // we gather and evaluate the polynonials
    gather_coefficient_init( vmm_indices, vlen / sizeof( float ) );
    gather_coefficient( vmm_pol, 6, vmm_indices );
    for( int deg = 5; deg >= 0; --deg ) {
        gather_coefficient( vmm_coeff, deg, vmm_indices );
        vfmadd213ps( vmm_pol, vmm_src, vmm_coeff );
    }

    // we restore src with cleared sign, and keep sign
    assert( vmm_sign.getIdx() == vmm_src_original.getIdx() );
    vmovups( vmm_src, vmm_src_original );
    vandps( vmm_sign, vmm_sign, getAddr( TTableKey::SignMask ) );
    vandps( vmm_src, vmm_src, getAddr( TTableKey::PositiveMask ) );

    // Now we blend the results
    // [saturation_ubound; +inf[ : we return +/- 1
    vmovups( vmm_dst, getAddr( TTableKey::One ) );
    // [linear_ubound; saturation_lbound] : we return +/- P(x)
    vmovups( vmm_mask, getAddr( TTableKey::TanhSaturationLBound ) );
    vcmpps( vmm_mask, vmm_mask, vmm_src, _cmp_gt_os );
    vblendvps( vmm_dst, vmm_dst, vmm_pol, vmm_mask );
    // [0; linear_ubound]  : we return x
    vmovups( vmm_mask, getAddr( TTableKey::TanhLineralUBound ) );
    vcmpps( vmm_mask, vmm_mask, vmm_src, _cmp_gt_os );
    vblendvps( vmm_dst, vmm_dst, vmm_src, vmm_mask );

    // We reapply the sign and return
    vxorps( vmm_dst, vmm_dst, vmm_sign );
    vmovups( vmm_src, vmm_dst );
}

void CLstmDescJit::CJitLstm::initTable()
{
    auto addTable = [&]( TTableKey key, const uint32_t* data, size_t dataSize ) {
        size_t currentSize = table.size();
        table.resize( currentSize + dataSize );
        
        tableOffsets.emplace( key, currentSize );

        uint32_t* pTable = &table[currentSize];
        for( ; currentSize < table.size(); currentSize++ ) {
            *pTable++ = *data++;
        }
    };

    auto addVal = [&]( TTableKey key, uint32_t val, size_t repeatNum ) {
        size_t currentSize = table.size();
        table.resize( currentSize + repeatNum );

        tableOffsets.emplace( key, currentSize );

        uint32_t* pTable = &table[currentSize];
        for( ; currentSize < table.size(); currentSize++ ) {
            *pTable++ = val;
        }
    };

    // tanh(x) polynomial approximation
    // For each coefficient, there is 32 entries
    const uint32_t TanhPolynomialCoeffs[] = {
        // coefficients of degree 0
        0x00000000,
        0x39bfffff,
        0x39ffffff,
        0x3a3ffffe,
        0x3a7ffffb,
        0x3abffff7,
        0x3affffeb,
        0x3b3fffdc,
        0x3b7fffab,
        0x3bbfff70,
        0x3bfffeab,
        0x3c3ffdc0,
        0x3c7ffaab,
        0x3cbff701,
        0x3cffeaad,
        0x3d3fdc08,
        0x3d7faacd,
        0x3dbf7081,
        0x3dfeacc9,
        0x3e3dc7fd,
        0x3e7acbf5,
        0x3eb77a9f,
        0x3eec9a9f,
        0x3f22991f,
        0x3f42f7d6,
        0x3f67b7cc,
        0x3f76ca83,
        0x3f7ebbe9,
        0x3f7fd40c,
        0x3f7fff32,
        0x3f7ffffc,
        0x3f800000,
        // coefficients of degree 1
        0x3f800000,
        0x3f800018,
        0x3f7fffe8,
        0x3f7fffda,
        0x3f7fffdc,
        0x3f7fffdc,
        0x3f7fffac,
        0x3f7fff70,
        0x3f7ffeec,
        0x3f7ffdc0,
        0x3f7ffbed,
        0x3f7ff704,
        0x3f7feff5,
        0x3f7fdbca,
        0x3f7fbfff,
        0x3f7f7041,
        0x3f7f009b,
        0x3f7dc36c,
        0x3f7c0aa8,
        0x3f7734b8,
        0x3f70a4de,
        0x3f5f1fd8,
        0x3f495493,
        0x3f18b9ec,
        0x3ed706cb,
        0x3e390b06,
        0x3d90b11f,
        0x3c21a053,
        0x3aaf7fdb,
        0x37ccc1a3,
        0x355c6733,
        0x00000000,
        // coefficients of degree 2
        0x00000000,
        0xbe4e0ff1,
        0x3d25b1b1,
        0x3d6b6dab,
        0x3c9fb1d5,
        0xbabff06f,
        0x3c07b3f6,
        0xbb3fc1bc,
        0x3a9f5921,
        0xbbbf06f2,
        0xbbb0f402,
        0xbc47db9e,
        0xbc73d5e7,
        0xbca25bda,
        0xbcfca780,
        0xbd40e07c,
        0xbd7dab03,
        0xbdbe4a0f,
        0xbdfb14a5,
        0xbe36cc8d,
        0xbe6bd102,
        0xbe9fe7c5,
        0xbeba0f10,
        0xbec206a8,
        0xbea3c388,
        0xbe277d62,
        0xbd8b7960,
        0xbc209f49,
        0xbaad44ca,
        0xb7c6eeac,
        0xb663aa41,
        0x00000000,
        // coefficients of degree 3
        0x00000000,
        0x45b3ae96,
        0xc414eb20,
        0xc450e02e,
        0xc3152b4e,
        0xbead2f56,
        0xc2162e02,
        0xbeb4bd5a,
        0xc11a59a4,
        0xbed2f507,
        0xc020d32c,
        0x3dd0f506,
        0xbf2a75e2,
        0xbff950e3,
        0xbed47334,
        0xbe809b8c,
        0xbeb64532,
        0xbe961a5b,
        0xbe9b63ac,
        0xbea0d4b2,
        0xbe828a77,
        0xbe378612,
        0xbdc20908,
        0x3d2d3957,
        0x3dd46e89,
        0x3db3f629,
        0x3d2c5e7b,
        0x3bd20403,
        0x3a59dfae,
        0x3770af45,
        0x372cc014,
        0x00000000,
        // coefficients of degree 4
        0x00000000,
        0xcc981a1b,
        0x4a7edd3d,
        0x4ab1007c,
        0x48fedd9c,
        0x41a557b5,
        0x477ee32a,
        0x422557f5,
        0x45ff3ce4,
        0x42a55641,
        0x446e0867,
        0xc33dc19a,
        0x42915214,
        0x43af4fad,
        0x4110fe88,
        0xc1099b75,
        0x3fc8a8dc,
        0xbfbeaef5,
        0xbe365aad,
        0x3f4d9652,
        0x3ddfa08f,
        0x3e34e9b8,
        0x3e2d07a6,
        0x3dc63567,
        0x3cdaeb78,
        0xbcd17537,
        0xbc92829c,
        0xbb43ab99,
        0xb9b471dd,
        0xb6baad5a,
        0xb78bafc7,
        0x00000000,
        // coefficients of degree 5
        0x00000000,
        0x52f688d5,
        0xd0505c72,
        0xd08f98e3,
        0xce505cc9,
        0xc7162b8a,
        0xcc5061d6,
        0xc7162bdf,
        0xca50b37f,
        0xc7162a3a,
        0xc8422086,
        0x471a714e,
        0xc5ece1f1,
        0xc70e3d90,
        0xc3eba94a,
        0x43e0c424,
        0xc21f4552,
        0x42217cc8,
        0x405e7dc4,
        0xc10dd401,
        0x3e96b602,
        0xbd1a6d2f,
        0xbd393883,
        0xbd674682,
        0xbd310016,
        0xb961e269,
        0x3ba32495,
        0x3a7680d5,
        0x38b3173c,
        0x35a9deea,
        0x375c3f2a,
        0x00000000,
        // coefficients of degree 6
        0x00000000,
        0xd8995ed1,
        0x558285ea,
        0x55b2cd69,
        0x53028625,
        0x4bc9991f,
        0x5082898a,
        0x4b4999b3,
        0x4e02c07c,
        0x4ac99764,
        0x4b72c822,
        0xca40c0e1,
        0x489413e4,
        0x49b12224,
        0x46134c4e,
        0xc60c2d57,
        0x43c83910,
        0xc3c872d1,
        0xc186bc9e,
        0x42325bc3,
        0xbf2ffa4a,
        0x3d9a203c,
        0xbc545a43,
        0xbae08fee,
        0x3c80225d,
        0x3b1fd1df,
        0xba36b9d1,
        0xb91de544,
        0xb71f100f,
        0xb408e2ed,
        0xb685fec8,
        0x00000000,
    };

    addTable( TTableKey::Coeff, TanhPolynomialCoeffs, sizeof( TanhPolynomialCoeffs ) / sizeof( uint32_t ) );
    addVal( TTableKey::TanhIdxBias, 0x39800000, 8 );
    addVal( TTableKey::TanhIdxMask31, 0x0000001f, 8 );
    addVal( TTableKey::TanhIdxMask, 0xffc00000, 8 );
    addVal( TTableKey::TanhLineralUBound, 0x39ddb3d7, 8 );
    addVal( TTableKey::TanhSaturationLBound, 0x41102cb3, 8 );
    addVal( TTableKey::PositiveMask, 0x7fffffff, 8 );
    addVal( TTableKey::One, 0x3f800000, 8 );
    addVal( TTableKey::SignMask, 0x80000000, 8 );
}

inline size_t CLstmDescJit::CJitLstm::getOfft( TTableKey key, size_t offset )
{
    assert( tableOffsets.find( key ) != tableOffsets.end() );
    return ( tableOffsets[key] + offset ) * sizeof( decltype( table )::value_type );
}

inline Xbyak::Address CLstmDescJit::CJitLstm::getAddr( TTableKey key )
{
    return Xbyak::util::ptr[regTablePtr + getOfft( key )];
}


} // namespace NeoML
