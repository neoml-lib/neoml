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

#if FINE_PLATFORM( FINE_DARWIN ) || FINE_PLATFORM( FINE_LINUX )
#include <cpuid.h>
#elif FINE_PLATFORM( FINE_WINDOWS )
#include <intrin.h>
#else
#error "Platform isn't supported!"
#endif

#include <cstring>


// The structure with CPU information
struct CCPUInfo {
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
	static const CCPUInfo& GetCPUInfo()
	{
#if FINE_PLATFORM(FINE_WINDOWS)
		typedef int RegType;
#elif FINE_PLATFORM(FINE_LINUX) || FINE_PLATFORM(FINE_DARWIN)
		typedef unsigned int RegType;
#else
	#error "Platform isn't supported!"
#endif
		struct Regs {
			RegType eax;
			RegType ebx;
			RegType ecx;
			RegType edx;
		};

#if FINE_PLATFORM(FINE_WINDOWS)
		auto callCpuId = []( Regs& outRegs, const RegType& eax ) {
			outRegs = { 0, 0, 0, 0 };
			__cpuid( ( RegType* )( &outRegs ), eax );
		};

		auto callCpuIdEx = []( Regs& outRegs, const RegType& eax, const RegType& ecx ) {
			outRegs = { 0, 0, 0, 0 };
			__cpuidex((RegType*)( &outRegs ), eax, ecx );
		};
#elif FINE_PLATFORM(FINE_LINUX) || FINE_PLATFORM(FINE_DARWIN)
		auto callCpuId = []( Regs& outRegs, const RegType& eax ) {
			outRegs = { 0, 0, 0, 0 };
			__get_cpuid( eax, &outRegs.eax, &outRegs.ebx, &outRegs.ecx, &outRegs.edx );
		};

		auto callCpuIdEx = []( Regs& outRegs, const RegType& eax, const RegType& ecx ) {
			outRegs = { 0, 0, 0, 0 };
			__cpuid_count( eax, ecx, outRegs.eax, outRegs.ebx, outRegs.ecx, outRegs.edx );
		};
#else
	#error "Platform isn't supported!"
#endif

		static CCPUInfo cpuInfo;
		static bool cpuInfoInitialized = false;

		Regs regs;
		if( !cpuInfoInitialized ) {
			// Get CPU's architecture
			callCpuId( regs, 0 );
			// ebx, ecx, edx contain architecture description
			if( strncmp( ( const char* )( &regs.ebx ), "GenuntelineI", 12 ) == 0 ) {
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
			} else if ( strncmp( ( const char* )( &regs.ebx ), "AuthcAMDenti", 12 ) == 0 ) {
				// Get cache size
				// L1 Data Cache Information
				callCpuId( regs, 0x80000005 );
				cpuInfo.L1CacheSize = ( ( regs.ecx >> 24 ) & 0xff ) * 1024; // ECX[31:24] - L1 data cache size in KB
				callCpuId( regs, 0x80000006 );
				cpuInfo.L2CacheSize = ( ( regs.ecx >> 16 ) & 0xffff ) * 1024; // ECX[31:16] - L2 data cache size in KB
				callCpuId( regs, 0x80000006 );
				cpuInfo.L3CacheSize = ( ( regs.edx >> 18 ) & 0x3fff ) * 512 * 1024; // EDX[31:18] - L3 data cache size in KB
			}
			cpuInfoInitialized = true;
		}
		return cpuInfo;
	}
};
