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

#include <NeoMathEngine/CrtAllocatedObject.h>
#include <MemoryPool.h>
#include <MemoryHandleInternal.h>
#include <algorithm>
#include <numeric>

namespace NeoML {

// An element in the buffer pool
class CMemoryBuffer : public CCrtAllocatedObject {
public:
	CMemoryHandle Data; // the pointer to the data

	explicit CMemoryBuffer() : next(0) {}

	CMemoryBuffer* GetNext() { return next; }
	void SetNext( CMemoryBuffer* _next ) { next = _next; }

private:
	CMemoryBuffer* next;
};

//------------------------------------------------------------------------------------------------------------

// A pool of buffers of the same size
class CMemoryBufferPool : public CCrtAllocatedObject {
public:
	const size_t BufferSize;

	explicit CMemoryBufferPool( size_t bufferSize ) : BufferSize(bufferSize), head(0), memoryInPool(0) {}

	// Allocates a buffer; returns 0 if no free buffers are available
	CMemoryBuffer* TryAlloc();

	// Releases a buffer
	void Free( CMemoryBuffer* data );

	// Gets the amount of memory used for the pool
	size_t GetMemoryInPool() const { return memoryInPool; }

private:
	// Currently free buffers (a singly-linked list)
	CMemoryBuffer* head;
	size_t memoryInPool; // the amount of memory used for the pool
};

CMemoryBuffer* CMemoryBufferPool::TryAlloc()
{
	CMemoryBuffer* result = head;

	if( result != 0 ) {
		head = result->GetNext();
		result->SetNext(0);
		memoryInPool -= BufferSize;
	}

	return result;
}

void CMemoryBufferPool::Free(CMemoryBuffer* data)
{
	data->SetNext( head );
	head = data;
	memoryInPool += BufferSize;
}

//------------------------------------------------------------------------------------------------------------

static const unsigned int BufferSizes[] = {
	256, 512, 1024, 2048, 2048 + 1024, 4096, 4096 + 2048, 8192,
	8192 + 4096, 16384, 16384 + 8192, 32768, 32768 + 16384, 65536, 65536 + 32768, 131072,
	131072 + 65536, 262144, 262144 + 131072, 524288, 524288 + 262144, 1048576, 1048576 + 524288, 2097152,
	2097152 + 1048576, 4194304, 4194304 + 2097152, 8388608, 8388608 + 4194304, 16777216, 16777216 + 8388608, 33554432,
	33554432 + 16777216, 67108864, 67108864 + 33554432, 134217728, 134217728 + 67108864, 268435456, 268435456 + 134217728,
	536870912, 536870912 + 268435456, 1073741824 // 1 GB max
};

template <typename T, int size>
inline constexpr int lengthof( T(&)[size] ) { return size; }

CMemoryPool::CMemoryPool( size_t _memoryLimit, IRawMemoryManager* _rawMemoryManager, bool reuseMemoryMode ) :
	memoryLimit( _memoryLimit ),
	rawMemoryManager( _rawMemoryManager ),
	defaultReuseMemoryMode( reuseMemoryMode ),
	allocatedMemory( 0 ),
	freeMemorySize( _memoryLimit ),
	peakMemoryUsage( 0 )
{
}

CMemoryPool::~CMemoryPool()
{
	for( auto curPool : pools ) {
		cleanUp( curPool.first );
		for( auto curMemBufferPool : curPool.second.Pool ) {
			delete curMemBufferPool;
		}
	}
}

void CMemoryPool::SetReuseMemoryMode( bool enable )
{
	thread::id id = this_thread::get_id();
	auto pool = pools.find( id );
	if( pool == pools.end() ) {
		createPools( id );
		pool = pools.find( id );
	}

	pool->second.Enabled = enable;
}

CMemoryHandle CMemoryPool::Alloc( size_t size )
{
	thread::id id = this_thread::get_id();
	auto pool = pools.find( id );
	if( pool == pools.end() ) {
		createPools( id );
		pool = pools.find( id );
	}

	CMemoryHandle result = tryAlloc( size, pool->second );
	if( !result.IsNull() ) {
		return result;
	}

	// Not enough memory. Try to free all allocated pools
	CleanUp();
	return tryAlloc( size, pool->second );
}

void CMemoryPool::Free( const CMemoryHandle& handle )
{
	TUsedAddressMap::const_iterator pos = usedMap.find( GetRaw(handle) );
	const CUsedInfo& info = pos->second;
	if( info.pool != 0 ) {
		info.pool->Free(info.buffer);
		freeMemorySize += info.pool->BufferSize;
	} else {
		// Large buffer, don't use the pool
		freeMemory(info.size, handle);
		freeMemorySize += info.size;
	}
	usedMap.erase( pos );
}

size_t CMemoryPool::GetMemoryInPools() const
{
	thread::id id = this_thread::get_id();
	auto pool = pools.find( id );
	if( pool == pools.end() ) {
		return 0;
	}
	const TPoolVector& threadPools = pool->second.Pool;
	return std::accumulate( threadPools.begin(), threadPools.end(), size_t( 0 ),
		[] ( const size_t& sum, const CMemoryBufferPool* cur ) { return sum + cur->GetMemoryInPool(); } );
}

void CMemoryPool::CleanUp()
{
	cleanUp( this_thread::get_id() );
}

void CMemoryPool::createPools( thread::id id )
{
	CThreadData threadData;
	threadData.Enabled = defaultReuseMemoryMode;
	for( size_t i = 0; i < sizeof( BufferSizes ) / sizeof( *BufferSizes ); ++i ) {
		threadData.Pool.push_back( new CMemoryBufferPool( BufferSizes[i] ) );
	}

	pools[id] = threadData;
}

void CMemoryPool::cleanUp( thread::id id )
{
	auto pool = pools.find( id );
	if( pool == pools.end() ) {
		return;
	}

	for( auto cur : pool->second.Pool ) {
		CMemoryBuffer* buffer = cur->TryAlloc();
		while( buffer != 0 ) {
			freeMemory(cur->BufferSize, buffer->Data);
			delete buffer;
			buffer = cur->TryAlloc();
		}
	}
}

inline static bool poolsCompare( const CMemoryBufferPool* a, const size_t& b )
{
	return a->BufferSize < b;
}

// Tries to allocate memory
CMemoryHandle CMemoryPool::tryAlloc( size_t size, CThreadData& data )
{
	if( !data.Enabled || size > BufferSizes[lengthof(BufferSizes) - 1] ) {
		// Allocate without using the buffers pool
		CMemoryHandle result = alloc( size );
		if( !result.IsNull() ) {
			usedMap[GetRaw( result )] = CUsedInfo( size, 0, 0 );
			freeMemorySize -= size;
		}
		return result;
	}

	// Allocate via the buffers pool
	TPoolVector::iterator pos = std::lower_bound( data.Pool.begin(), data.Pool.end(), size, &poolsCompare );

	CMemoryBufferPool* pool = *pos;
	CMemoryBuffer* buffer = pool->TryAlloc();
	if( buffer == 0 ) {
		buffer = new CMemoryBuffer();
		buffer->Data = alloc( pool->BufferSize );
		if( buffer->Data.IsNull() ) {
			delete buffer;
			return CMemoryHandle();
		}
	}
	freeMemorySize -= pool->BufferSize;
	usedMap[GetRaw(buffer->Data)] = CUsedInfo(size, buffer, pool);

	return buffer->Data;
}

CMemoryHandle CMemoryPool::alloc( size_t size )
{
	if( size > memoryLimit || allocatedMemory > memoryLimit - size ) {
		return CMemoryHandle();
	}
	
	CMemoryHandle result = rawMemoryManager->Alloc( size );

	if( !result.IsNull() ) {
		allocatedMemory += size;
	}
	peakMemoryUsage = max( peakMemoryUsage, allocatedMemory );

	return result;
}

void CMemoryPool::freeMemory( size_t size, const CMemoryHandle& data )
{
	allocatedMemory -= size;

	rawMemoryManager->Free( data );
}

} // namespace NeoML
