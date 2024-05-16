/* Copyright © 2017-2024 ABBYY

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

#include <CpuMathEngine.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>
#include <NeoMathEngine/SimdMathEngine.h>
#include <DllLoader.h>
#include <CPUInfo.h>
#include <PerformanceCountersDefault.h>

#if FINE_PLATFORM( FINE_ANDROID ) || FINE_PLATFORM( FINE_LINUX )
#include <PerformanceCountersCpuLinux.h>
#elif FINE_PLATFORM( FINE_WINDOWS ) || FINE_PLATFORM( FINE_DARWIN ) || FINE_PLATFORM( FINE_IOS )
#else
#error "Platform is not supported!";
#endif

#ifdef NEOML_USE_MKL

#if FINE_PLATFORM( FINE_WINDOWS ) || FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN )
#include <mkl.h>
#else
#error Unknown platform
#endif

#endif // NEOML_USE_MKL

const bool CCPUInfo::HasAvxAndFma = CCPUInfo::IsAvxAndFmaAvailable();
const bool CCPUInfo::IsNotIntel = CCPUInfo::GetCpuArch() != CCPUInfo::TCpuArch::Intel;

namespace NeoML {

int NEOMATHENGINE_API FloatAlignment = CCPUInfo::DefineFloatAlignment();

CCpuMathEngine::CCpuMathEngine( size_t _memoryLimit,
		std::shared_ptr<CMultiThreadDistributedCommunicator> communicator,
		const CMathEngineDistributedInfo& distributedInfo ) :
	floatAlignment( FloatAlignment ),
	communicator( communicator ),
	distributedInfo( distributedInfo ),
	dllLoader( CDllLoader::AVX_DLL )
{
	InitializeMemory( this, _memoryLimit, static_cast<int>( floatAlignment * sizeof( float ) ),
		/*reuse*/IsDistributed(), /*hostStack*/false );
#ifdef NEOML_USE_AVX
	if( dllLoader.IsLoaded( CDllLoader::AVX_DLL ) ) {
		simdMathEngine = std::unique_ptr<ISimdMathEngine>( CDllLoader::avxDll->CreateSimdMathEngine( this ) );
		// Don't use custom sgemm function when we are compiled with MKL or MLAS.
#if !defined( NEOML_USE_MKL ) && !defined( NEOML_USE_MLAS )
		customSgemmFunction = simdMathEngine->GetSgemmFunction();
#endif // !defined( NEOML_USE_MKL ) && !defined( NEOML_USE_MLAS )
	}
#else  // !NEOML_USE_AVX
	// warning fix
	( void ) customSgemmFunction;
#endif // !NEOML_USE_AVX
#ifdef NEOML_USE_MKL
	vmlSetMode( VML_ERRMODE_NOERR );
#endif // NEOML_USE_MKL
}

CCpuMathEngine::~CCpuMathEngine()
{
	CleanUp();
}

void CCpuMathEngine::CleanUpSpecial()
{
#ifdef NEOML_USE_MKL
	mkl_thread_free_buffers();
#endif // NEOML_USE_MKL
}

void* CCpuMathEngine::GetBuffer( const CMemoryHandle& handle, size_t pos, size_t, bool exchange )
{
	( void ) exchange; // always returned, no need to copy
	return reinterpret_cast<char*>( GetRaw( handle ) ) + pos;
}

void CCpuMathEngine::ReleaseBuffer( const CMemoryHandle&, void*, bool )
{
	// no action needed
}

void CCpuMathEngine::DataExchangeRaw( const CMemoryHandle& handle, const void* data, size_t size )
{
	ASSERT_EXPR( handle.GetMathEngine() == this );
	::memcpy( GetRaw( handle ), data, size );
}

void CCpuMathEngine::DataExchangeRaw( void* data, const CMemoryHandle& handle, size_t size )
{
	ASSERT_EXPR( handle.GetMathEngine() == this );
	::memcpy( data, GetRaw( handle ), size );
}

CMemoryHandle CCpuMathEngine::Alloc( size_t size )
{
	// Ensure the correct alignment
	void* ptr = 0;
	if( MEMORY_ALLOCATION_ALIGNMENT % MemoryAlignment == 0 ) {
		ptr = malloc(size);
	} else {
		char* p = static_cast<char*>(malloc(size + MemoryAlignment));
		if( p != 0 ) {
			const intptr_t delta = MemoryAlignment - std::abs( ( reinterpret_cast<intptr_t>( p ) % MemoryAlignment ) );
			ASSERT_EXPR( delta > 0 && delta <= static_cast<intptr_t>( MemoryAlignment ) );

			p[delta - 1] = static_cast<char>( delta - 1 );
			ptr = p + delta;
		}
	}

	if( ptr == 0 ) {
		return CMemoryHandle();
	}

	return CMemoryHandleInternal::CreateMemoryHandle( this, ptr );
}

void CCpuMathEngine::Free( const CMemoryHandle& handle )
{
	ASSERT_EXPR( handle.GetMathEngine() == this );

	char* ptr = GetRaw( CTypedMemoryHandle<char>( handle ) );

	if( MEMORY_ALLOCATION_ALIGNMENT % MemoryAlignment == 0 ) {
		free(ptr);
		return;
	}

	ptr = ptr - ptr[-1] - 1;
	free(ptr);
}

void CCpuMathEngine::GetMathEngineInfo( CMathEngineInfo& info ) const
{
	info.Type = MET_Cpu;
	::strcpy( info.Name, "CPU" );
	info.Id = 0;
	info.AvailableMemory = SIZE_MAX;
}

#if FINE_PLATFORM( FINE_ANDROID ) || FINE_PLATFORM( FINE_LINUX )
IPerformanceCounters* CCpuMathEngine::CreatePerformanceCounters( bool isOnlyTime ) const {
	if ( isOnlyTime ) {
		return new CPerformanceCountersDefault();
	}
	return new CPerformanceCountersCpuLinux();
}
#elif FINE_PLATFORM( FINE_WINDOWS ) || FINE_PLATFORM( FINE_DARWIN ) || FINE_PLATFORM( FINE_IOS )
IPerformanceCounters* CCpuMathEngine::CreatePerformanceCounters( bool ) const {
	return new CPerformanceCountersDefault();
}
#else
IPerformanceCounters* CCpuMathEngine::CreatePerformanceCounters( bool ) const {
	#error "Platform is not supported!";
	return 0;
}
#endif

void CCpuMathEngine::AllReduce( const CFloatHandle& handle, int size )
{
	if( communicator != nullptr ){
		communicator->AllReduce( handle, size );
	}
}

void CCpuMathEngine::Broadcast( const CFloatHandle& handle, int size, int root )
{
	if( communicator != nullptr ){
		communicator->Broadcast( handle, size, root );
	}
}

void CCpuMathEngine::AbortDistributed() 
{
	if( communicator != nullptr ){
		communicator->Abort();
	}
}

void CpuMathEngineCleanUp()
{
#ifdef NEOML_USE_MKL
	// mkl_thread_free_buffers does not free the memory completely
	// Looks like a bug in mkl
	mkl_free_buffers();
#endif
}

void DeinitializeNeoMathEngine()
{
#ifdef NEOML_USE_MKL
	mkl_free_buffers();
#if FINE_PLATFORM( FINE_WINDOWS )
	MKLFreeTls( DLL_PROCESS_DETACH );
#endif
	mkl_finalize();
#endif
}

} // namespace NeoML
