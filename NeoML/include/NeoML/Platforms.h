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

// The macros are defined for determining the parameters of the platform for which you are building the application
// Operating system:
// FINE_WINDOWS
// FINE_LINUX
// FINE_DARWIN
// Use the FINE_PLATFORM( FINE_* ) macro to check the OS
//
// Bit architecture:
// FINE_32_BIT
// FINE_64_BIT
// Use the FINE_BIT( FINE_*_BIT ) macro to check the OS bit architecture
//
// Byte order:
// FINE_BIG_ENDIAN
// FINE_LITTLE_ENDIAN
// Use the FINE_BYTE_ORDER( FINE_*_ENDIAN ) macro to check the byte order
//
// Some additional rules
//  1) DO NOT use #else and #ifndef
//  2) If you have a lot of platform-dependent code, we recommend that you use separate files for different platforms

#if defined(_WIN32)
	#define FINE_PLATFORM( a ) a
	#define FINE_BIT( a ) a
	#define FINE_BYTE_ORDER( a ) a

	#define FINE_WINDOWS 1
	#ifdef _WIN64
		#define FINE_64_BIT 1
	#else
		#define FINE_32_BIT 1
	#endif
	#define FINE_LITTLE_ENDIAN 1
#elif defined(_ANDROID)
	#define FINE_PLATFORM( a ) a
	#define FINE_BIT( a ) a
	#define FINE_BYTE_ORDER( a ) a

	#define FINE_ANDROID 1
	#ifdef _PLATFORM_64_BIT
		#define FINE_64_BIT 1
	#else
		#define FINE_32_BIT 1
	#endif
	#define FINE_LITTLE_ENDIAN 1
#elif defined(_IOS)
	#define FINE_PLATFORM( a ) a
	#define FINE_BIT( a ) a
	#define FINE_BYTE_ORDER( a ) a

	#define FINE_IOS 1
	#ifdef _PLATFORM_64_BIT
		#define FINE_64_BIT 1
	#else
		#define FINE_32_BIT 1
	#endif
	#define FINE_LITTLE_ENDIAN 1
#elif defined(_LINUX)
	#define FINE_PLATFORM( a ) a
	#define FINE_BIT( a ) a
	#define FINE_BYTE_ORDER( a ) a

	#define FINE_LINUX 1
	#ifdef __x86_64__
		#define FINE_64_BIT 1
	#else
		#define FINE_32_BIT 1
	#endif
	#define FINE_LITTLE_ENDIAN 1
#elif defined(_DARWIN)
	#define FINE_PLATFORM( a ) a
	#define FINE_BIT( a ) a
	#define FINE_BYTE_ORDER( a ) a

	#define FINE_DARWIN 1
	#ifdef _PLATFORM_64_BIT
		#define FINE_64_BIT 1
	#else
		#define FINE_32_BIT 1
	#endif
	#define FINE_LITTLE_ENDIAN 1
#endif

#if defined(_M_IX86) || defined(__i386__)
	#define FINE_ARCHITECTURE( a ) a

	#define FINE_X86 1
#elif defined(__x86_64) || defined(_M_X64)
	#define FINE_ARCHITECTURE( a ) a

	#define FINE_X64 1
#elif defined(__arm__) || defined(_M_ARM)
	#define FINE_ARCHITECTURE( a ) a

	#define FINE_ARM 1
#elif defined(__aarch64__) || defined(_M_ARM64)
	#define FINE_ARCHITECTURE( a ) a

	#define FINE_ARM64 1
#endif

#if defined(_MSC_VER)
	#define FINE_COMPILER( a ) a

	#define FINE_VS 1
#elif defined(__clang__)
	#define FINE_COMPILER( a ) a

	#define FINE_CLANG 1
#elif defined(__GNUC__)
	#define FINE_COMPILER( a ) a

	#define FINE_GCC 1
#endif
