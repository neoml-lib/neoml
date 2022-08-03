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

#include <common.h>
#pragma hdrstop

#include <CpuMathEngine.h>
#include <MathEngineDeviceStackAllocator.h>
#include <MathEngineHostStackAllocator.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>
#include <NeoMathEngine/SimdMathEngine.h>
#include <DllLoader.h>
#include <CPUInfo.h>

#if FINE_PLATFORM( FINE_ANDROID ) || FINE_PLATFORM( FINE_LINUX )
#include <PerformanceCountersCpuLinux.h>
#elif FINE_PLATFORM( FINE_WINDOWS ) || FINE_PLATFORM( FINE_DARWIN ) || FINE_PLATFORM( FINE_IOS )
#include <PerformanceCountersDefault.h>
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

namespace NeoML {

static int FloatAlignment = CCPUInfo::DefineFloatAlignment();
static CCPUInfo::TCpuArch CPUArch = CCPUInfo::GetCpuArch();

CCpuMathEngine::CCpuMathEngine( int _threadCount, size_t _memoryLimit,
		std::shared_ptr<CMultiThreadDistributedCommunicator> communicator,
		const CMathEngineDistributedInfo& distributedInfo ) :
	threadCount( _threadCount <= 0 ? OmpGetMaxThreadCount() : _threadCount ),
	floatAlignment( FloatAlignment ),
	memoryAlignment( floatAlignment * sizeof(float) ),
	communicator( communicator ),
	distributedInfo( distributedInfo ),
	memoryPool( new CMemoryPool( _memoryLimit == 0 ? SIZE_MAX : _memoryLimit, this, distributedInfo.Threads > 1 ) ),
	stackAllocator( new CDeviceStackAllocator( *memoryPool, memoryAlignment ) ),
	dllLoader( CDllLoader::AVX_DLL ),
	simdMathEngine( nullptr ),
	customSgemmFunction( nullptr )
{
#ifdef NEOML_USE_AVX
	if( dllLoader.IsLoaded( CDllLoader::AVX_DLL ) ) {
		simdMathEngine = unique_ptr<ISimdMathEngine>( CDllLoader::avxDll->CreateSimdMathEngine( this, threadCount ) );
		// Don't use custom sgemm function when we are compiled with MKL and when we are on Intel CPU.
		if( CPUArch == CCPUInfo::TCpuArch::Intel ) {
#ifndef NEOML_USE_MKL
			customSgemmFunction = simdMathEngine->GetSgemmFunction();
#endif
		} else {
			// Non Intel architectures
			customSgemmFunction = simdMathEngine->GetSgemmFunction();
		}
	}
#else // NEOML_USE_AVX
	// warning fix
	(void)customSgemmFunction;
#endif
#ifdef NEOML_USE_MKL
	vmlSetMode( VML_ERRMODE_NOERR );
#endif
}

CCpuMathEngine::~CCpuMathEngine()
{
	CleanUp();
}

void CCpuMathEngine::SetReuseMemoryMode( bool enable )
{
	// Distributed CPU math engine always uses memory pools
	// because big simultaneous allocations on multiple (20+) threads are extremely slow
	if( IsDistributed() ) {
		return;
	}

	std::lock_guard<std::mutex> lock( mutex );
	memoryPool->SetReuseMemoryMode( enable );
}

CMemoryHandle CCpuMathEngine::HeapAlloc( size_t size )
{
	std::lock_guard<std::mutex> lock( mutex );
	CMemoryHandle result = memoryPool->Alloc( size );
	if( result.IsNull() ) {
		THROW_MEMORY_EXCEPTION;
	}
	return result;
}

void CCpuMathEngine::HeapFree( const CMemoryHandle& handle )
{
	ASSERT_EXPR( handle.GetMathEngine() == this );

	std::lock_guard<std::mutex> lock( mutex );
	memoryPool->Free( handle );
}

CMemoryHandle CCpuMathEngine::StackAlloc( size_t size )
{
	std::lock_guard<std::mutex> lock( mutex );
	CMemoryHandle result = stackAllocator->Alloc(size);
	if( result.IsNull() ) {
		THROW_MEMORY_EXCEPTION;
	}
	return result;
}

void CCpuMathEngine::StackFree( const CMemoryHandle& ptr )
{
	std::lock_guard<std::mutex> lock( mutex );
	stackAllocator->Free( ptr );
}

size_t CCpuMathEngine::GetFreeMemorySize() const
{
	std::lock_guard<std::mutex> lock( mutex );
	return memoryPool->GetFreeMemorySize();
}

size_t CCpuMathEngine::GetPeakMemoryUsage() const
{
	std::lock_guard<std::mutex> lock( mutex );
	return memoryPool->GetPeakMemoryUsage();
}

size_t CCpuMathEngine::GetMemoryInPools() const
{
	std::lock_guard<std::mutex> lock( mutex );
	return memoryPool->GetMemoryInPools();
}

void CCpuMathEngine::CleanUp()
{
	std::lock_guard<std::mutex> lock( mutex );
	stackAllocator->CleanUp();
	memoryPool->CleanUp();
#ifdef NEOML_USE_MKL
	NEOML_OMP_NUM_THREADS( threadCount )
	{
		mkl_thread_free_buffers();
	}
#endif
}

void* CCpuMathEngine::GetBuffer( const CMemoryHandle& handle, size_t pos, size_t, bool exchange )
{
	(void) exchange; // always returned, no need to copy
	return reinterpret_cast<char*>( GetRaw( handle ) ) + pos;
}

void CCpuMathEngine::ReleaseBuffer( const CMemoryHandle&, void*, bool )
{
	// no action needed
}

void CCpuMathEngine::DataExchangeRaw(const CMemoryHandle& handle, const void* data, size_t size)
{
	ASSERT_EXPR( handle.GetMathEngine() == this );

	::memcpy( GetRaw(handle), data, size);
}

void CCpuMathEngine::DataExchangeRaw(void* data, const CMemoryHandle& handle, size_t size)
{
	ASSERT_EXPR( handle.GetMathEngine() == this );

	::memcpy( data, GetRaw(handle), size );
}

CMemoryHandle CCpuMathEngine::CopyFrom( const CMemoryHandle& handle, size_t size )
{
	CMemoryHandle result = HeapAlloc( size );

	IMathEngine* otherMathEngine = handle.GetMathEngine();
	otherMathEngine->DataExchangeRaw( GetRaw( result ), handle, size );

	return result;
}

CMemoryHandle CCpuMathEngine::Alloc( size_t size )
{
	// Ensure the correct alignment
	void* ptr = 0;
	if( MEMORY_ALLOCATION_ALIGNMENT % memoryAlignment == 0 ) {
		ptr = malloc(size);
	} else {
		char* p = static_cast<char*>(malloc(size + memoryAlignment));
		if( p != 0 ) {
			const intptr_t delta = memoryAlignment - std::abs( ( reinterpret_cast<intptr_t>( p ) % memoryAlignment ) );
			ASSERT_EXPR( delta > 0 && delta <= static_cast<intptr_t>( memoryAlignment ) );

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

	if( MEMORY_ALLOCATION_ALIGNMENT % memoryAlignment == 0 ) {
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

IPerformanceCounters* CCpuMathEngine::CreatePerformanceCounters() const
{
#if FINE_PLATFORM( FINE_ANDROID ) || FINE_PLATFORM( FINE_LINUX )
	return new CPerformanceCountersCpuLinux();
#elif FINE_PLATFORM( FINE_WINDOWS ) || FINE_PLATFORM( FINE_DARWIN ) || FINE_PLATFORM( FINE_IOS )
	return new CPerformanceCountersDefault();
#else
	#error "Platform is not supported!";
	return 0;
#endif
}

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
