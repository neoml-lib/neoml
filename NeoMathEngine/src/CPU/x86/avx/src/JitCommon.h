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

#include <xbyak/xbyak.h>
#include <stack>

namespace NeoML {

using reg64_t = Xbyak::Reg64;
using ymm_t = Xbyak::Ymm;
using ymmVec_t = std::vector<ymm_t>;
using reg64Vec_t = std::vector<reg64_t>;

constexpr Xbyak::Operand::Code CalleeSavedRegisters[] = {
    Xbyak::Operand::RBX,
    Xbyak::Operand::RBP,
    Xbyak::Operand::R12,
    Xbyak::Operand::R13,
    Xbyak::Operand::R14,
    Xbyak::Operand::R15,
    #ifdef _WIN32
    Xbyak::Operand::RDI,
    Xbyak::Operand::RSI,
    #endif
};

#ifdef _WIN32
constexpr reg64_t Param1{Xbyak::Operand::RCX};
constexpr reg64_t Param2{Xbyak::Operand::RDX};
constexpr reg64_t Param3{Xbyak::Operand::R8};
constexpr reg64_t Param4{Xbyak::Operand::R9};
#else
constexpr reg64_t Param1{Xbyak::Operand::RDI};
constexpr reg64_t Param2{Xbyak::Operand::RSI};
constexpr reg64_t Param3{Xbyak::Operand::RDX};
constexpr reg64_t Param4{Xbyak::Operand::RCX};
constexpr reg64_t Param5{Xbyak::Operand::R8};
constexpr reg64_t Param6{Xbyak::Operand::R9};
#endif

constexpr unsigned int NumFloatInYmm = 8;
constexpr unsigned int SizeOfYmm = NumFloatInYmm * sizeof( float );
constexpr unsigned int SizeofReg64 = 8;
constexpr unsigned int MaxYmmCount = 16;

class CJitCommon : public Xbyak::CodeGenerator {
public:
    using Base = Xbyak::CodeGenerator;
    CJitCommon() = default;

    // preservedGPR and preservedYmm will be preserved on stack (must be the same as in Epilogue()!!!)
    // return Address which point to first of arguments is stored on stack if 
    // such arguments exist (passed in gprArgsCount and ymmArgsCount params)
    Xbyak::Address Prologue( const reg64Vec_t& preservedGPR,
        const ymmVec_t& preservedYmm );
    // preservedGPR and preservedYmm will be poped from stack (must be the same as in Prologue()!!!)
    void Epilogue( const reg64Vec_t& preservedGPR,
        const ymmVec_t& preservedYmm );

    // Implement loop (can be nested one into another ):
    // while( counter >= step ) {
    //    // do some useful work
    //    counter -= step;
    // }
    void StartDownCountLoop( reg64_t counter, size_t step );
    void StopDownCountLoop();


    template<class LastVec>
    bool HasSameSize( const LastVec& ) {
        return true;
    }

    template<class Vec1, class Vec2, class... Vecs>
    bool HasSameSize( const Vec1& vec1, const Vec2& vec2, const Vecs&... vecs ) {
        return vec1.size() == vec2.size() && HasSameSize( vec1, vecs... );
    }

    /////////////////////////////////////////////////////////////////////////////////////
    // Macroses and templates which help to redefine xbyak functions for vector case
    // V - vector, S - single
    template<class BaseFunc, class V, class S>
    void VectorImpl_VS( BaseFunc func, const V& v, const S& s ) {
        for( int i = 0; i < v.size(); i++ ) { ( this->*func )( v[i], s ); }
    }

    template<class BaseFunc, class V1, class V2, class S>
    void VectorImpl_2VS( BaseFunc func, const V1& v1, const V2& v2, const S& s ) {
        assert( HasSameSize( v1, v2 ) );
        for( int i = 0; i < v1.size(); i++ ) { ( this->*func )( v1[i], v2[i], s ); }
    }

    template<class BaseFunc, class V1, class V2>
    void VectorImpl_2V( BaseFunc func, const V1& v1, const V2& v2 ) {
        assert( HasSameSize( v1, v2 ) );
        for( int i = 0; i < v1.size(); i++ ) { ( this->*func )( v1[i], v2[i] ); }
    }

    template<class BaseFunc, class V1, class V2, class V3>
    void VectorImpl_3V( BaseFunc func, const V1& v1, const V2& v2, const V3& v3 ) {
        assert( HasSameSize( v1, v2, v3 ) );
        for( int i = 0; i < v1.size(); i++ ) { ( this->*func )( v1[i], v2[i], v3[i] ); }
    }

    template<class BaseFunc, class V1, class V2, class V3, class V4>
    void VectorImpl_4V( BaseFunc func, const V1& v1, const V2& v2, const V3& v3, const V4& v4 ) {
        assert( HasSameSize( v1, v2, v3, v4 ) );
        for( int i = 0; i < v1.size(); i++ ) { ( this->*func )( v1[i], v2[i], v3[i], v4[i] ); }
    }

#define XBYAK_CAST_2( funcName, castType, fromOp1, fromOp2, toOp1, toOp2 ) \
    void funcName( const toOp1& p1, const toOp2& p2 ) { \
        void ( Base::* pFunc )( const Xbyak::fromOp1&, const Xbyak::fromOp2& ) = &Base::funcName; \
        castType( pFunc, p1, p2 ); }

#define XBYAK_FORWARD_CAST_2( funcName, op1, op2 ) \
    void funcName( const Xbyak::op1& p1, const Xbyak::op2& p2 ) { \
        Base::funcName( p1, p2 ); }

#define XBYAK_CAST_3( funcName, castType, fromOp1, fromOp2, fromOp3, toOp1, toOp2, toOp3 ) \
    void funcName( const toOp1& p1, const toOp2& p2, const toOp3& p3 ) { \
        void ( Base::* pFunc )( const Xbyak::fromOp1&, const Xbyak::fromOp2&, const Xbyak::fromOp3& ) = &Base::funcName; \
        castType( pFunc, p1, p2, p3 ); }

#define XBYAK_CAST_3_STRICT( funcName, castType, fromOp1, fromOp2, fromOp3, toOp1, toOp2, toOp3 ) \
    void funcName( toOp1 p1, toOp2 p2, toOp3 p3 ) { \
        void ( Base::* pFunc )( fromOp1, fromOp2, fromOp3 ) = &Base::funcName; \
        castType( pFunc, p1, p2, p3 ); }

#define XBYAK_CAST_4( funcName, castType, fromOp1, fromOp2, fromOp3, fromOp4, toOp1, toOp2, toOp3, toOp4 ) \
    void funcName( const toOp1& p1, const toOp2& p2, const toOp3& p3, const toOp4& p4 ) { \
        void ( Base::* pFunc )( const Xbyak::fromOp1&, const Xbyak::fromOp2&, const Xbyak::fromOp3&, const Xbyak::fromOp4& ) = &Base::funcName; \
        castType( pFunc, p1, p2, p3, p4 ); }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Redefined functions from xbyak for vectors
    XBYAK_CAST_2( vcvtps2dq, VectorImpl_2V, Xmm, Operand, ymmVec_t, ymmVec_t )
    XBYAK_CAST_2( vmovups, VectorImpl_2V, Xmm, Operand, ymmVec_t, ymmVec_t )
    XBYAK_CAST_2( vmovups, VectorImpl_VS, Xmm, Operand, ymmVec_t, Xbyak::Operand )
    XBYAK_CAST_3( vaddps, VectorImpl_2VS, Xmm, Operand, Operand, ymmVec_t, ymmVec_t, Xbyak::Operand )
    XBYAK_CAST_3( vandps, VectorImpl_2VS, Xmm, Operand, Operand, ymmVec_t, ymmVec_t, Xbyak::Operand )
    XBYAK_CAST_3( vcmpeq_uqps, VectorImpl_3V, Xmm, Xmm, Operand, ymmVec_t, ymmVec_t, ymmVec_t )
    XBYAK_CAST_3( vcmpltps, VectorImpl_2VS, Xmm, Xmm, Operand, ymmVec_t, ymmVec_t, Xbyak::Operand )
    XBYAK_CAST_3( vcmpnltps, VectorImpl_3V, Xmm, Xmm, Operand, ymmVec_t, ymmVec_t, ymmVec_t )
    XBYAK_CAST_3( vdivps, VectorImpl_3V, Xmm, Operand, Operand, ymmVec_t, ymmVec_t, ymmVec_t )
    XBYAK_CAST_3( vfmadd213ps, VectorImpl_2VS, Xmm, Xmm, Operand, ymmVec_t, ymmVec_t, Xbyak::Operand )
    XBYAK_CAST_3( vfmadd213ps, VectorImpl_3V, Xmm, Xmm, Operand, ymmVec_t, ymmVec_t, ymmVec_t )
    XBYAK_CAST_3( vfnmadd231ps, VectorImpl_2VS, Xmm, Xmm, Operand, ymmVec_t, ymmVec_t, Xbyak::Operand )
    XBYAK_CAST_3( vgatherdps, VectorImpl_3V, Xmm, Address, Xmm, ymmVec_t, std::vector<Xbyak::Address>, ymmVec_t )
    XBYAK_CAST_3( vmaxps, VectorImpl_2VS, Xmm, Operand, Operand, ymmVec_t, ymmVec_t, Xbyak::Operand )
    XBYAK_CAST_3( vminps, VectorImpl_2VS, Xmm, Operand, Operand, ymmVec_t, ymmVec_t, Xbyak::Operand )
    XBYAK_CAST_3( vmulps, VectorImpl_2VS, Xmm, Operand, Operand, ymmVec_t, ymmVec_t, Xbyak::Operand )
    XBYAK_CAST_3( vmulps, VectorImpl_3V, Xmm, Operand, Operand, ymmVec_t, ymmVec_t, ymmVec_t )
    XBYAK_CAST_3( vpaddd, VectorImpl_2VS, Xmm, Xmm, Operand, ymmVec_t, ymmVec_t, Xbyak::Operand )
    XBYAK_CAST_3( vpsubd, VectorImpl_2VS, Xmm, Xmm, Operand, ymmVec_t, ymmVec_t, Xbyak::Operand )
    XBYAK_CAST_3( vsubps, VectorImpl_2VS, Xmm, Operand, Operand, ymmVec_t, ymmVec_t, Xbyak::Operand )
    XBYAK_CAST_3( vsubps, VectorImpl_3V, Xmm, Operand, Operand, ymmVec_t, ymmVec_t, ymmVec_t )
    XBYAK_CAST_3( vxorps, VectorImpl_3V, Xmm, Operand, Operand, ymmVec_t, ymmVec_t, ymmVec_t )
    XBYAK_CAST_3_STRICT( vpslld, VectorImpl_2VS, const Xbyak::Xmm&, const Xbyak::Operand&, uint8_t, const ymmVec_t&, const ymmVec_t&, uint8_t )
    XBYAK_CAST_3_STRICT( vpsrld, VectorImpl_2VS, const Xbyak::Xmm&, const Xbyak::Operand&, uint8_t, const ymmVec_t&, const ymmVec_t&, uint8_t )
    XBYAK_CAST_3_STRICT( vroundps, VectorImpl_2VS, const Xbyak::Xmm&, const Xbyak::Operand&, uint8_t, const ymmVec_t&, const ymmVec_t&, uint8_t )
    XBYAK_CAST_4( vblendvps, VectorImpl_4V, Xmm, Xmm, Operand, Xmm, ymmVec_t, ymmVec_t, ymmVec_t, ymmVec_t )
    XBYAK_FORWARD_CAST_2( vmovups, Address, Xmm )
    XBYAK_FORWARD_CAST_2( vmovups, Xmm, Operand )

private:
    struct CLoopDesc {
        CLoopDesc( reg64_t counter, size_t step ) : Counter( counter ), Step( step ) {}
        Xbyak::Label StartLabel;
        Xbyak::Label EndLabel;
        reg64_t Counter;
        size_t Step;
    };
    std::stack<CLoopDesc> loopDescs;

};

}
