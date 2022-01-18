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

#include <JitCommon.h>

namespace NeoML {

using namespace std;

Xbyak::Address CJitCommon::Prologue( const reg64Vec_t& preservedGPR,
    const ymmVec_t& preservedYmm )
{
    using namespace Xbyak::util;
    using namespace Xbyak;

#ifdef _WIN32
    const int ShadowSpace = 0x20;
#else
    const int ShadowSpace = 0;
#endif

    push( rbp );
    mov( rbp, rsp );
 
    sub( rsp, static_cast<uint32_t>( preservedYmm.size() * SizeOfYmm ) );
    for( int i = 0; i < preservedYmm.size(); i++ ) {
        vmovdqu( ptr[rsp + i * SizeOfYmm], preservedYmm[i] );
    }

    for( int i = 0; i < preservedGPR.size(); i++ ) {
        push( preservedGPR[i] );
    }

    return ptr[rsp + ShadowSpace + static_cast< uint32_t >( 
        2 * SizeofReg64 + preservedGPR.size() * SizeofReg64 + preservedYmm.size() * SizeOfYmm )];
}

void CJitCommon::Epilogue( const reg64Vec_t& preservedGPR,
    const ymmVec_t& preservedYmm )
{
    for( int i = static_cast<int>( preservedGPR.size() - 1 ); i >= 0; i-- ) {
        pop( preservedGPR[i] );
    }

    for( int i = 0; i < static_cast< int >( preservedYmm.size() ); i++ ) {
        vmovdqu( preservedYmm[i], ptr[rsp + static_cast< uint32_t >( i * SizeOfYmm )] );
    }

    leave();
}

void CJitCommon::StartDownCountLoop( reg64_t counter, size_t step )
{
    loopDescs.emplace( counter, step );
    CLoopDesc& loopDesc = loopDescs.top();
    L( loopDesc.StartLabel );
    cmp( loopDesc.Counter, loopDesc.Step );
    jl( loopDesc.EndLabel, T_NEAR );
}

void CJitCommon::StopDownCountLoop()
{
    assert( !loopDescs.empty() );
    CLoopDesc& loopDesc = loopDescs.top();
    sub( loopDesc.Counter, loopDesc.Step );
    jmp( loopDesc.StartLabel, T_NEAR );
    L( loopDesc.EndLabel );
    loopDescs.pop();
}

}
