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

#pragma once

#include <NeoMathEngine/NeoMathEngineDefs.h>
#include <MathEngineStackAllocator.h>
#include <RawMemoryManager.h>
#include <MemoryPool.h>
#include <memory>
#include <mutex>

namespace NeoML {

// Memory management engine base class
class CMemoryEngineMixin : public IMathEngine {
public:
	void InitializeMemory( IRawMemoryManager* _rawManager, size_t _memoryLimit, int _memoryAlignment,
		bool _reuse, bool _hostStack );
	bool IsInitialized() const override { return MemoryAlignment > 0 && MemoryPool != 0 && DeviceStackAllocator != 0; }

	void SetReuseMemoryMode( bool enable ) override;
	bool GetReuseMemoryMode() const override;
	void SetThreadBufferMemoryThreshold( size_t threshold ) override;
	size_t GetThreadBufferMemoryThreshold() const override;
	CMemoryHandle HeapAlloc( size_t count ) override;
	void HeapFree( const CMemoryHandle& handle ) override;
	void TransferHandleToThisThread( const CMemoryHandle& handle, size_t size ) override;
	CMemoryHandle StackAlloc( size_t count ) override;
	void StackFree( const CMemoryHandle& handle ) override;
	size_t GetFreeMemorySize() const override;
	size_t GetPeakMemoryUsage() const override;
	void ResetPeakMemoryUsage() override;
	size_t GetCurrentMemoryUsage() const override;
	size_t GetMemoryInPools() const override;

	void CleanUp() override;
	void* GetBuffer( const CMemoryHandle& handle, size_t pos, size_t size, bool exchange ) override;
	void ReleaseBuffer( const CMemoryHandle& handle, void* ptr, bool exchange ) override;
	CMemoryHandle CopyFrom( const CMemoryHandle& handle, size_t size ) override;

protected:
	int MemoryAlignment = 0; // allocation alignment
	mutable std::mutex Mutex; // protecting the data below from non-thread-safe use
	std::unique_ptr<CMemoryPool> MemoryPool; // memory manager
	std::unique_ptr<IStackAllocator, CStackAllocatorDeleter> DeviceStackAllocator; // stack allocator for GPU memory
	std::unique_ptr<IStackAllocator, CStackAllocatorDeleter> HostStackAllocator; // stack allocator for regular memory

	void CleanUpSpecial() override {}

	// All below is need to avoid excess (8 bytes) field in each CMemoryHandler

	// Special constructor
	explicit CMemoryEngineMixin( int/*cannot be no call*/ ) :
		CurrentEntity( ++MathEngineEntitiesNumerator )
	{
		ASSERT_EXPR( CurrentEntity < CMemoryHandle::MaxMathEngineEntities );
		MathEngineEntitiesArray[CurrentEntity] = this;
	}

	// Generation for indices of all IMathEngine entities,
	// Incremets evey moment new MathEngine created to generate its CurrentEntity value.
	static size_t MathEngineEntitiesNumerator;
	// Array for pointers to all entities of IMathEngine
	// No cache ping-pong, because pointers are created once and never changes
	static IMathEngine* MathEngineEntitiesArray[CMemoryHandle::MaxMathEngineEntities];
	// Index of the current IMathEngine entity
	const size_t CurrentEntity;

	friend IMathEngine* GetMathEngineByIndex( size_t currentEntity );
	friend size_t GetIndexOfMathEngine( const IMathEngine* mathEngine );
};

} // namespace NeoML
