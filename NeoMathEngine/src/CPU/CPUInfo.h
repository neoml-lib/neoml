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

#if !FINE_ARCHITECTURE( FINE_ARM64 ) && !FINE_ARCHITECTURE( FINE_ARM )

#if FINE_PLATFORM( FINE_DARWIN ) || FINE_PLATFORM( FINE_LINUX )
#include <cpuid.h>
#elif FINE_PLATFORM( FINE_WINDOWS )
#include <intrin.h>
#endif

#endif // !FINE_ARCHITECTURE( FINE_ARM64 )

#include <cstring>


// The structure with CPU information
struct CCPUInfo {
	enum class TCpuArch {
		Intel,
		AMD,
		Others
	};

	size_t L1CacheSize = 0;
	size_t L2CacheSize = 0;
	size_t L3CacheSize = 0;

	constexpr CCPUInfo() = default;
	constexpr CCPUInfo(size_t L1CacheSize, size_t L2CacheSize = 0, size_t L3CacheSize = 0) :
		L1CacheSize(L1CacheSize),
		L2CacheSize(L2CacheSize),
		L3CacheSize(L3CacheSize)
	{}


	// Try to retrieve CPU info from hardware. Returns nullptr if feature isn't implemented.
	static CCPUInfo GetCPUInfo()
	{
		CCPUInfo cpuInfo;

		Regs regs;
		switch( GetCpuArch() ) {
		case TCpuArch::Intel:
		{
			// Get cache size
			auto calcCacheSize = [&]( int cacheNumber ) -> int {
				// Cache Size in Bytes
				// = (Ways + 1) * (Partitions + 1) * (Line_Size + 1) * (Sets + 1)
				// = (EBX[31:22] + 1) * (EBX[21:12] + 1) * (EBX[11:0] + 1) * (ECX + 1)

				// Get Ways, Partitions, Line_size and Sets
				callCpuIdEx( regs, 4, cacheNumber );
				int ways = ( regs.ebx >> 22 ) & 0x3ff; // EBX[31:22]
				int partitions = ( regs.ebx >> 12 ) &0x3ff; // EBX[21:12]
				int lineSize = regs.ebx & 0xfff; // EBX[11:0]
				int sets = regs.ecx;

				// Get cache size
				callCpuIdEx( regs, 4, 1 );
				return  ( ways + 1 ) * ( partitions + 1 ) * ( lineSize + 1 ) * (sets + 1 );
			};

			cpuInfo.L1CacheSize = calcCacheSize( 1 );
			cpuInfo.L2CacheSize = calcCacheSize( 2 );
			cpuInfo.L3CacheSize = calcCacheSize( 3 );
			break;
		}
		case TCpuArch::AMD:
		{
			// Get cache size
			// L1 Data Cache Information
			callCpuId( regs, 0x80000005 );
			cpuInfo.L1CacheSize = ( ( regs.ecx >> 24 ) & 0xff ) * 1024; // ECX[31:24] - L1 data cache size in KB
			callCpuId( regs, 0x80000006 );
			cpuInfo.L2CacheSize = ( ( regs.ecx >> 16 ) & 0xffff ) * 1024; // ECX[31:16] - L2 data cache size in KB
			callCpuId( regs, 0x80000006 );
			cpuInfo.L3CacheSize = ( ( regs.edx >> 18 ) & 0x3fff ) * 512 * 1024; // EDX[31:18] - L3 data cache size in KB
			break;
		}
		default:
			break;
		}

		return cpuInfo;
	}

	static TCpuArch GetCpuArch() {
		Regs regs;
		callCpuId( regs, 0 );
		// ebx, ecx, edx contain architecture description
		if( strncmp( ( const char* )( &regs.ebx ), "GenuntelineI", 12 ) == 0 ) {
			return TCpuArch::Intel;
		} else if ( strncmp( ( const char* )( &regs.ebx ), "AuthcAMDenti", 12 ) == 0 ) {
			return TCpuArch::AMD;
		} else {
			return TCpuArch::Others;
		}
	}

	// Defines the float alignment
	static int DefineFloatAlignment()
	{
#ifdef NEOML_USE_NEON
		return 4;
#else
		int floatAlignment = 4; // SSE alignment

		Regs regs;
		callCpuId( regs, 1 );

		// avx and osxsave
		const int AvxFlag = ( 1 << 28 ) + ( 1 << 27 );

		if( (regs.ecx & AvxFlag) == AvxFlag ) {
#if FINE_PLATFORM(FINE_WINDOWS)
			// AVX supported
			// Check OS support (if it keeps AVX register when switching contexts - OSXSAVE)
			int64_t res = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
			const int64_t OsFlag = 0x6;
			if((res & OsFlag) == OsFlag) {
				// AVX supported, change the alignment for better operation of mkl
				floatAlignment = 8;
			}
#elif FINE_PLATFORM(FINE_LINUX) || FINE_PLATFORM(FINE_DARWIN) || FINE_PLATFORM(FINE_ANDROID) || FINE_PLATFORM(FINE_IOS)
			floatAlignment = 8;
#else
#error "Platform isn't supported!"
#endif
		}

		return floatAlignment;
#endif // NEOML_USE_NEON
	}

	static bool IsAvxAndFmaAvailable()
	{
		Regs regs;
		callCpuId( regs, 1 );

		const unsigned int fmaBit = ( 1 << 12 );
		const bool fmaIsAvailable = ( regs.ecx & fmaBit ) == fmaBit;
		if( !fmaIsAvailable ) {
			return false;
		}

		callCpuIdEx( regs, 7, 0 );

		const unsigned int avx2Bit = ( 1 << 5 );
		const bool avx2IsAvailable = ( regs.ebx & avx2Bit ) == avx2Bit;

		return avx2IsAvailable;
	}

	static bool IsAvx512Available()
	{
		Regs regs;
		callCpuIdEx( regs, 7, 0 );

		// Check avx512_f bit in EBX ( any CPU with AVX512 has this bit )
		bool AnyAvx512IsAvailable = regs.ebx & ( 1 << 16 );

		return AnyAvx512IsAvailable;
	}

private:

#if FINE_PLATFORM(FINE_WINDOWS)
	typedef int RegType;
#else
	typedef unsigned int RegType;
#endif
	struct Regs {
		RegType eax;
		RegType ebx;
		RegType ecx;
		RegType edx;
	};

	static void callCpuId( Regs& outRegs, const RegType& eax ) {
		outRegs = { 0, 0, 0, 0 };
#if !FINE_ARCHITECTURE( FINE_ARM64 ) && !FINE_ARCHITECTURE( FINE_ARM )
#if FINE_PLATFORM( FINE_WINDOWS )
		__cpuid( ( RegType* )( &outRegs ), eax );
#elif FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN )
		__get_cpuid( eax, &outRegs.eax, &outRegs.ebx, &outRegs.ecx, &outRegs.edx );
#else
	( void ) eax;
#endif
#else
	( void ) eax;
#endif // !FINE_ARCHITECTURE( FINE_ARM64 )
	}

	static void callCpuIdEx( Regs& outRegs, const RegType& eax, const RegType& ecx ) {
		outRegs = { 0, 0, 0, 0 };
#if !FINE_ARCHITECTURE( FINE_ARM64 ) && !FINE_ARCHITECTURE( FINE_ARM )
#if FINE_PLATFORM( FINE_WINDOWS )
		__cpuidex((RegType*)( &outRegs ), eax, ecx );
#elif FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN )
		__cpuid_count( eax, ecx, outRegs.eax, outRegs.ebx, outRegs.ecx, outRegs.edx );
#else
	( void ) eax;
	( void ) ecx;
#endif
#else
	( void ) eax;
	( void ) ecx;
#endif // !FINE_ARCHITECTURE( FINE_ARM64 )
	}

};
