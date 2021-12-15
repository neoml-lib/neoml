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
    CJitCommon() = default;

    // preservedGPR and preservedYmm will be preserved on stack (must be the same as in Epilogue()!!!)
    // return Address which point to first of arguments is stored on stack if 
    // such arguments exist (passed in gprArgsCount and ymmArgsCount params)
    Xbyak::Address Prologue( const std::vector<reg64_t>& preservedGPR,
        const std::vector<ymm_t>& preservedYmm );
    // preservedGPR and preservedYmm will be poped from stack (must be the same as in Prologue()!!!)
    void Epilogue( const std::vector<reg64_t>& preservedGPR,
        const std::vector<ymm_t>& preservedYmm );

    // Implement loop (can be nested one into another ):
    // while( counter >= step ) {
    //    // do some useful work
    //    counter -= step;
    // }
    void StartDownCountLoop( reg64_t counter, size_t step );
    void StopDownCountLoop();
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
