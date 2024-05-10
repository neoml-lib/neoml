/* Copyright © 2024 ABBYY

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

#include <NeoMathEngine/NeoMathEngine.h>
#include <MathEngineAllocator.h>
#include <MathEngineCommon.h>
#include <MathEngineDeviceStackAllocator.h>
#include <MathEngineHostStackAllocator.h>
#include <MemoryEngineMixin.h>
#include <MemoryHandleInternal.h>
#include <MemoryPool.h>

namespace NeoML {

void CMemoryEngineMixin::InitializeMemory( IRawMemoryManager* _rawManager, size_t _memoryLimit, int _memoryAlignment,
		bool _reuse, bool _hostStack )
{
	MemoryAlignment = _memoryAlignment;
	MemoryPool.reset( new CMemoryPool( _memoryLimit == 0 ? SIZE_MAX : _memoryLimit, _rawManager, _reuse ) );
	DeviceStackAllocator.reset( new CDeviceStackAllocator( *MemoryPool, MemoryAlignment ) );
	if( _hostStack == true ) {
		HostStackAllocator.reset( new CHostStackAllocator( MemoryAlignment ) );
	}
}

void CMemoryEngineMixin::SetReuseMemoryMode( bool enable )
{
	switch( GetType() ) {
	case MET_Cuda:
		// Always true, because allocation is sync
		break;
	case MET_Cpu:
		// Distributed CPU math engine always uses memory pools
		// because big simultaneous allocations on multiple (20+) threads are extremely slow
		if( IsDistributed() ) {
			break;
		}
		// fallthrough
	case MET_Metal:
	case MET_Vulkan:
	{
		std::lock_guard<std::mutex> lock( Mutex );
		MemoryPool->SetReuseMemoryMode( enable );
		break;
	}
	default:
		ASSERT_EXPR( false );
	}
}

bool CMemoryEngineMixin::GetReuseMemoryMode() const
{
	switch( GetType() ) {
	case MET_Cuda:
		// Always true, because allocation is sync
		return true;
	case MET_Cpu:
		// Distributed CPU math engine always uses memory pools
		if( IsDistributed() ) {
			return true;
		}
		// fallthrough
	case MET_Metal:
	case MET_Vulkan:
	{
		std::lock_guard<std::mutex> lock( Mutex );
		return MemoryPool->GetReuseMemoryMode();
	}
	default:
		ASSERT_EXPR( false );
	}
	return false;
}

void CMemoryEngineMixin::SetThreadBufferMemoryThreshold( size_t threshold )
{
	std::lock_guard<std::mutex> lock( Mutex );
	MemoryPool->SetThreadBufferMemoryThreshold( threshold );
}

size_t CMemoryEngineMixin::GetThreadBufferMemoryThreshold() const
{
	std::lock_guard<std::mutex> lock( Mutex );
	return MemoryPool->GetThreadBufferMemoryThreshold();
}

CMemoryHandle CMemoryEngineMixin::HeapAlloc( size_t size )
{
	std::lock_guard<std::mutex> lock( Mutex );
	CMemoryHandle result = MemoryPool->Alloc( size );
	if( result.IsNull() ) {
		THROW_MEMORY_EXCEPTION;
	}
	return result;
}

void CMemoryEngineMixin::HeapFree( const CMemoryHandle& handle )
{
	ASSERT_EXPR( handle.GetMathEngine() == this );

	std::lock_guard<std::mutex> lock( Mutex );
	MemoryPool->Free( handle );
}

void CMemoryEngineMixin::TransferHandleToThisThread( const CMemoryHandle& handle, size_t size )
{
	ASSERT_EXPR( GetType() == MET_Cpu || GetType() == MET_Cuda );
	ASSERT_EXPR( handle.GetMathEngine() == this );

	std::lock_guard<std::mutex> lock( Mutex );
	MemoryPool->TransferHandleToThisThread( handle, size );
}

CMemoryHandle CMemoryEngineMixin::StackAlloc( size_t size )
{
	std::lock_guard<std::mutex> lock( Mutex );
	CMemoryHandle result = DeviceStackAllocator->Alloc( size );
	if( result.IsNull() ) {
		THROW_MEMORY_EXCEPTION;
	}
	return result;
}

void CMemoryEngineMixin::StackFree( const CMemoryHandle& ptr )
{
	std::lock_guard<std::mutex> lock( Mutex );
	DeviceStackAllocator->Free( ptr );
}

size_t CMemoryEngineMixin::GetFreeMemorySize() const
{
	std::lock_guard<std::mutex> lock( Mutex );
	return MemoryPool->GetFreeMemorySize();
}

size_t CMemoryEngineMixin::GetPeakMemoryUsage() const
{
	std::lock_guard<std::mutex> lock( Mutex );
	return MemoryPool->GetPeakMemoryUsage();
}

void CMemoryEngineMixin::ResetPeakMemoryUsage()
{
	std::lock_guard<std::mutex> lock( Mutex );
	MemoryPool->ResetPeakMemoryUsage();
}

size_t CMemoryEngineMixin::GetCurrentMemoryUsage() const
{
	std::lock_guard<std::mutex> lock( Mutex );
	return MemoryPool->GetCurrentMemoryUsage();
}

size_t CMemoryEngineMixin::GetMemoryInPools() const
{
	std::lock_guard<std::mutex> lock( Mutex );
	return MemoryPool->GetMemoryInPools();
}

void CMemoryEngineMixin::CleanUp()
{
	std::lock_guard<std::mutex> lock( Mutex );
	DeviceStackAllocator->CleanUp();
	if( HostStackAllocator != nullptr ) {
		HostStackAllocator->CleanUp();
	}
	CleanUpSpecial();
	MemoryPool->CleanUp();
}

void* CMemoryEngineMixin::GetBuffer( const CMemoryHandle& handle, size_t pos, size_t size, bool exchange )
{
	ASSERT_EXPR( HostStackAllocator != nullptr );
	ASSERT_EXPR( handle.GetMathEngine() == this );

	size_t realSize = size + 16;
	char* result = static_cast<char*>( HostStackAllocator->Alloc( realSize ) );
	size_t* posPtr = reinterpret_cast<size_t*>( result );
	*posPtr = pos;
	size_t* sizePtr = reinterpret_cast<size_t*>( result ) + 1;
	*sizePtr = size;
	if( exchange ) {
		DataExchangeRaw( result + 16, handle, size );
	}
	return result + 16;
}

void CMemoryEngineMixin::ReleaseBuffer( const CMemoryHandle& handle, void* ptr, bool exchange )
{
	ASSERT_EXPR( HostStackAllocator != nullptr );
	ASSERT_EXPR( handle.GetMathEngine() == this );

	if( exchange ) {
		size_t* posPtr = reinterpret_cast<size_t*>( static_cast<char*>( ptr ) - 16 );
		size_t pos = *posPtr;
		size_t* sizePtr = posPtr + 1;
		size_t size = *sizePtr;

		DataExchangeRaw( CTypedMemoryHandle<char>( handle ) + pos, ptr, size );
	}
	HostStackAllocator->Free( static_cast<char*>( ptr ) - 16 );
}

CMemoryHandle CMemoryEngineMixin::CopyFrom( const CMemoryHandle& handle, size_t size )
{
	CMemoryHandle result = HeapAlloc( size );

	IMathEngine* otherMathEngine = handle.GetMathEngine();
	otherMathEngine->DataExchangeRaw( GetRaw( result ), handle, size );

	return result;
}

} // namespace NeoML
