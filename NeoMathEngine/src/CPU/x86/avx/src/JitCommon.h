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
#include <xbyak/xbyak.h>

namespace NeoML {

using reg64_t = Xbyak::Reg64;

#ifdef XBYAK64
        static constexpr Xbyak::Operand::Code CalleeSavedRegisters[] = {
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
        static const reg64_t FuncParam1{Xbyak::Operand::RCX};
        static const reg64_t FuncParam2{Xbyak::Operand::RDX};
        static const reg64_t FuncParam3{Xbyak::Operand::R8};
        static const reg64_t FuncParam4{Xbyak::Operand::R9};
#else
        static const reg64_t FuncParam1{Xbyak::Operand::RDI};
        static const reg64_t FuncParam2{Xbyak::Operand::RSI};
        static const reg64_t FuncParam3{Xbyak::Operand::RDX};
        static const reg64_t FuncParam4{Xbyak::Operand::RCX};
        static const reg64_t FuncParam5{Xbyak::Operand::R8};
        static const reg64_t FuncParam6{Xbyak::Operand::R9};
#endif
#endif

}
