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

#pragma once

#include <MathEngineAllocator.h>
#include <RawMemoryManager.h>
#include <unordered_map>
#include <thread>
#include <vector>

namespace NeoML {

class CMemoryBufferPool;
class CMemoryBuffer;

// The memory manager
class CMemoryPool : public CCrtAllocatedObject {
public:
	CMemoryPool( size_t memoryLimit, IRawMemoryManager* rawMemoryManager, bool reuseMemoryMode );
	~CMemoryPool();

	// Turns on and off the memory reuse mode for the current thread
	void SetReuseMemoryMode( bool enable );
	// Get the memory reuse mode state for the current thread
	bool GetReuseMemoryMode() const;

	// Change the memory blocks' sizes threshold for this thread from 1GB to the user size in bytes
	void SetThreadBufferMemoryThreshold( size_t threshold );
	// Get the memory blocks' sizes threshold for this thread
	size_t GetThreadBufferMemoryThreshold() const;

	// Allocates the specified amount of memory
	CMemoryHandle Alloc( size_t size );
	// Frees the memory
	void Free( const CMemoryHandle& handle );

	// Gets the amount of memory currently available
	size_t GetFreeMemorySize() const { return freeMemorySize; }

	// Gets the peak memory usage achieved during processing
	size_t GetPeakMemoryUsage() const { return peakMemoryUsage; }
	// Reset the peak memory counter to the current memory usage value
	void ResetPeakMemoryUsage() { peakMemoryUsage = allocatedMemory; }
	// The current memory usage size
	size_t GetCurrentMemoryUsage() const { return allocatedMemory; }
	// Gets the amount of memory used for the pools
	size_t GetMemoryInPools() const;

	// Frees all memory on the current thread
	void CleanUp();

	// Transfers handle from other thread owner to this thread
	void TransferHandleToThisThread( const CMemoryHandle& handle, size_t size );

private:
	using TMemoryBufferPoolVector = std::vector<CMemoryBufferPool*, CrtAllocator<CMemoryBufferPool*>>;
	// The information about all of memory buffers pools of unused non-cleared blocks
	struct CThreadData final {
		static const size_t DefaultBufferMemoryThreshold;
		TMemoryBufferPoolVector Pool;
		bool Enabled = false; // default 'reuse' mode is disabled
		size_t BufferMemoryThreshold = DefaultBufferMemoryThreshold; // default max = 1GB
	};
	using TThreadDataMap = std::unordered_map<
		std::thread::id, CThreadData, // (key, value)
		std::hash<std::thread::id>,
		std::equal_to<std::thread::id>,
		CrtAllocator< std::pair<const std::thread::id, CThreadData>>
	>;
	// The information about a memory block address
	struct CUsedInfo final {
		size_t size = 0;
		CMemoryBuffer* buffer = nullptr;

		CUsedInfo( size_t _size = 0 ) : size( _size ) {}
		CUsedInfo( CMemoryBuffer* _buffer ) : buffer( _buffer ) {}
	};
	// The memory blocks addresses map
	using TUsedAddressMap = std::unordered_map<
		void*, CUsedInfo, // (key, value)
		std::hash<void*>,
		std::equal_to<void*>,
		CrtAllocator< std::pair<void* const, CUsedInfo>>
	>;

	const size_t memoryLimit;
	IRawMemoryManager* const rawMemoryManager;
	const bool defaultReuseMemoryMode;

	TThreadDataMap pools;
	size_t allocatedMemory; // the amount of memory allocated on device (belonging to the user + used for the pools)
	size_t freeMemorySize; // the amount of free avialable memory
	size_t peakMemoryUsage; // peak memory usage
	TUsedAddressMap usedMap;

	const CThreadData* getThreadData() const;
	CThreadData* getThreadData( bool forceCreate = true );
	CThreadData* createPools( std::thread::id id );
	void cleanUp( CThreadData* threadData );
	CMemoryHandle tryAlloc( size_t size, CThreadData& data );
	CMemoryHandle alloc( size_t size );
	void freeMemory( size_t size, const CMemoryHandle& data );
};

} // namespace NeoML
