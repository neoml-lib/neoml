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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_SSE
#include <pmmintrin.h>
#endif

namespace NeoML {

// Scope which sets register for calculations on CPU
// E.g. denormalized float control
class CCpuExecutionScope {
public:
	CCpuExecutionScope();
	~CCpuExecutionScope();

	CCpuExecutionScope( const CCpuExecutionScope& ) = delete;
	CCpuExecutionScope& operator= ( const CCpuExecutionScope& ) = delete;

private:

#ifdef NEOML_USE_SSE
	unsigned int prevDenormalZero;
	unsigned int prevFlushZero;
#endif // NEOML_USE_SSE

#ifdef NEOML_USE_NEON

#define ARM_FLUSH_TO_ZERO (1 << 24)

#if FINE_PLATFORM( FINE_ARM64 )
uint64_t prevFpcr;
#else // FINE_PLATFORM( FINE_ARM64 )
uint32_t prevFpscr;
#endif // FINE_PLATFORM( FINE_ARM64 )

#endif // NEOML_USE_NEON

};

inline CCpuExecutionScope::CCpuExecutionScope()
{
#ifdef NEOML_USE_SSE
	// Turning on DAZ and FTZ registers for denormalized floats
	prevDenormalZero = _MM_GET_DENORMALS_ZERO_MODE();
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
	prevFlushZero = _MM_GET_FLUSH_ZERO_MODE();
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#endif // NEOML_USE_SSE

#ifdef NEOML_USE_NEON

#if FINE_PLATFORM( FINE_ARM64 )
	__asm __volatile("mrs %[fpcr], fpcr" : [fpcr] "=r" (prevFpcr));
	uint64_t newFpcr = ( prevFpcr | ARM_FLUSH_TO_ZERO );
	__asm __volatile("msr fpcr, %[fpcr]" : : [fpcr] "r" (newFpcr));
#else // FINE_PLATFORM( FINE_ARM64 )
	__asm __volatile("vmrs %[fpscr], fpscr" : [fpscr] "=r" (prevFpscr));
	uint32_t newFpscr = ( prevFpscr | ARM_FLUSH_TO_ZERO );
	__asm __volatile("vmsr fpscr, %[fpscr]" : : [fpscr] "r" (newFpscr));
#endif // FINE_PLATFORM( FINE_ARM64 )

#endif // NEOML_USE_NEON
}

inline CCpuExecutionScope::~CCpuExecutionScope()
{
#ifdef NEOML_USE_SSE
	_MM_SET_DENORMALS_ZERO_MODE(prevDenormalZero);
	_MM_SET_FLUSH_ZERO_MODE(prevFlushZero);
#endif // NEOML_USE_SSE

#ifdef NEOML_USE_NEON

#if FINE_PLATFORM( FINE_ARM64 )
	__asm __volatile("msr fpcr, %[fpcr]" : : [fpcr] "r" (prevFpcr));
#else // FINE_PLATFORM( FINE_ARM64 )
	__asm __volatile("vmsr fpscr, %[fpscr]" : : [fpscr] "r" (prevFpscr));
#endif // FINE_PLATFORM( FINE_ARM64 )

#endif // NEOML_USE_NEON
}

} // namespace NeoML
