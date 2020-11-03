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

#include <MathEngineAllocator.h>
#include <NeoMathEngine/MemoryHandle.h>
#include <NeoMathEngine/CrtAllocatedObject.h>
#include <RawMemoryManager.h>
#include <MathEngineCommon.h>

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

	// Allocates the specified amount of memory
	CMemoryHandle Alloc( size_t size );
	
	// Frees the memory
	void Free( const CMemoryHandle& handle );

	// Gets the amount of memory currently available
	size_t GetFreeMemorySize() const { std::lock_guard<std::mutex> lock( mutex ); return freeMemorySize; }

	// Gets the peak memory usage achieved during processing
	size_t GetPeakMemoryUsage() const { std::lock_guard<std::mutex> lock( mutex ); return peakMemoryUsage; }

	// Frees all memory on the current thread
	void CleanUp();

private:
	mutable std::mutex mutex;
	const size_t memoryLimit;
	IRawMemoryManager* const rawMemoryManager;
	const bool defaultReuseMemoryMode;

	struct CPoolData;
	CThreadData<CPoolData> poolData;

	size_t allocatedMemory; // the amount of memory allocated on device (belonging to the user + used for the pools)
	size_t freeMemorySize; // the amount of free avialable memory
	size_t peakMemoryUsage; // peak memory usage

	// The information about a memory block
	struct CUsedInfo {
		size_t size;
		CMemoryBuffer* buffer;
		CMemoryBufferPool* pool;

		CUsedInfo() :
			size( 0 ), buffer( nullptr), pool( 0 ) {}
		CUsedInfo(size_t _size, CMemoryBuffer* _buffer, CMemoryBufferPool* _pool) :
			size(_size), buffer(_buffer), pool(_pool) {}
	};
	
	unordered_map<void*, CUsedInfo> usedMap;

	CMemoryHandle tryAlloc( size_t size, CPoolData& data );
	CMemoryHandle alloc( size_t size );
	void freeMemory( size_t size, const CMemoryHandle& data );
};

} // namespace NeoML
