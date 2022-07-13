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

#include <NeoMathEngine/Platforms.h>

#if FINE_PLATFORM( FINE_WINDOWS )

#pragma warning( disable : 4251 ) // class '' needs to have dll-interface to be used by clients of class ''
#pragma warning( disable : 4275 ) // non dll-interface class '' used as base for dll-interface class ''
#pragma warning( disable : 4310 ) // cast truncates constant value
#pragma warning( disable : 4458 ) // declaration of 'x' hides class member
#pragma warning( disable : 4702 ) // unreachable code

#include <Windows.h>

#elif FINE_PLATFORM( FINE_ANDROID ) || FINE_PLATFORM( FINE_IOS ) || FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN )

#include <cstdint>
#include <chrono>

#if FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN ) || FINE_PLATFORM( FINE_ANDROID ) || FINE_PLATFORM( FINE_IOS )
	#ifndef MEMORY_ALLOCATION_ALIGNMENT
		#define MEMORY_ALLOCATION_ALIGNMENT (2 * sizeof(void*))
	#endif
#endif

template <typename T, int size>
inline constexpr int _countof( T(&)[size] ) { return size; }

typedef unsigned char BYTE;
typedef unsigned int DWORD;
#define __int64 long long
typedef uintptr_t UINT_PTR;
constexpr int NOT_FOUND = -1;

inline unsigned char _BitScanForward( unsigned long *index, unsigned long mask )
{
	if( !mask ) {
		return 0;
	}
	*index = __builtin_ctzl(mask);
	return 1;
}

inline unsigned char _BitScanForward64( unsigned long *index, unsigned __int64 mask )
{
  	if( !mask ) {
    		return 0;
  	}	
  	*index = __builtin_ctzll(mask);
  	return 1;
}

inline unsigned long long GetTickCount()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>( steady_clock::now().time_since_epoch() ).count();
}

#else

#error "Platform is not supported!"

#endif // FINE_PLATFORM( FINE_WINDOWS )

#include <algorithm>
#include <memory.h>
#include <initializer_list>
#include <mutex>
#include <cmath>
#include <atomic>
#include <string>
#include <sstream>
#include <cassert>

#define FINE_DEBUG_NEW new
