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
	CMemoryBuffer* Next = nullptr; // the pointer to next buffer in the pool
	CMemoryBufferPool* OwnerPool; // the pointer to the owner pool of buffers

	explicit CMemoryBuffer( CMemoryBufferPool* ownerPool ) : OwnerPool( ownerPool ) {}
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
	CMemoryBuffer* buffer = head;
	if( buffer != nullptr ) {
		head = buffer->Next;
		buffer->Next = nullptr;
		memoryInPool -= BufferSize;
	}
	return buffer;
}

void CMemoryBufferPool::Free( CMemoryBuffer* buffer )
{
	buffer->Next = head;
	head = buffer;
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

const size_t CMemoryPool::CThreadData::DefaultBufferMemoryThreshold = BufferSizes[lengthof( BufferSizes ) - 1];

//------------------------------------------------------------------------------------------------------------

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
	for( auto& curPool : pools ) {
		cleanUp( &curPool.second );
		for( auto curMemBufferPool : curPool.second.Pool ) {
			delete curMemBufferPool;
		}
	}
}

void CMemoryPool::SetReuseMemoryMode( bool enable )
{
	getThreadData()->Enabled = enable;
}

bool CMemoryPool::GetReuseMemoryMode() const
{
	const CThreadData* threadData = getThreadData();
	return ( threadData != nullptr ) ? threadData->Enabled : false;
}

void CMemoryPool::SetThreadBufferMemoryThreshold( size_t threshold )
{
	getThreadData()->BufferMemoryThreshold = threshold;
}

size_t CMemoryPool::GetThreadBufferMemoryThreshold() const
{
	const CThreadData* threadData = getThreadData();
	return ( threadData != nullptr )
		? threadData->BufferMemoryThreshold
		: CThreadData::DefaultBufferMemoryThreshold;
}

CMemoryHandle CMemoryPool::Alloc( size_t size )
{
	CThreadData& threadData = *getThreadData();
	CMemoryHandle result = tryAlloc( size, threadData );
	if( !result.IsNull() ) {
		return result;
	}

	// Not enough memory. Try to free all allocated pools
	CleanUp();
	return tryAlloc( size, threadData );
}

void CMemoryPool::Free( const CMemoryHandle& handle )
{
	TUsedAddressMap::const_iterator it = usedMap.find( GetRaw( handle ) );
	const CUsedInfo& info = it->second;

	if( info.HasPoolBuffer() ) {
		CMemoryBufferPool* const pool = info.Buffer()->OwnerPool;
		pool->Free( info.Buffer() );
		freeMemorySize += pool->BufferSize;
	} else {
		// Large buffer, don't use the pool
		freeMemory( info.Size(), handle );
		freeMemorySize += info.Size();
	}
	usedMap.erase( it );
}

size_t CMemoryPool::GetMemoryInPools() const
{
	const CThreadData* threadData = getThreadData();
	if( threadData == nullptr ) {
		return 0;
	}
	return std::accumulate( threadData->Pool.begin(), threadData->Pool.end(), size_t( 0 ),
		[] ( const size_t& sum, const CMemoryBufferPool* cur ) { return sum + cur->GetMemoryInPool(); } );
}

void CMemoryPool::CleanUp()
{
	cleanUp( getThreadData( /*forceCreate*/false ) );
}

// Transfers handle from other thread owner to this thread
void CMemoryPool::TransferHandleToThisThread( const CMemoryHandle& handle, size_t size )
{
	TUsedAddressMap::iterator usedMapIt = usedMap.find( GetRaw( handle ) );
	CUsedInfo& info = usedMapIt->second;

	if( info.HasPoolBuffer() ) {
		// Find the buffer pool to steal from
		CMemoryBufferPool* otherThreadBufferPool = info.Buffer()->OwnerPool;
		ASSERT_EXPR( size <= otherThreadBufferPool->BufferSize );
		size = otherThreadBufferPool->BufferSize; // set actual allocated size

		CThreadData& thisThreadData = *getThreadData();
		// If on this thread pools are turned off
		if( !thisThreadData.Enabled ) {
			// Transfer the handle from that thread's pool just to heap, so
			// it wouldn't be cleaned-up for that thread after mathEngine.CleanUp().
			delete info.Buffer();
			info = CUsedInfo( size );
		} else {
			// Find the buffer in this thread's pool to append it to
			CMemoryBufferPool* thisThreadBufferPool = nullptr;
			for( int i = 0; i < static_cast<int>( thisThreadData.Pool.size() ); ++i ) {
				if( thisThreadData.Pool[i]->BufferSize == size ) {
					thisThreadBufferPool = thisThreadData.Pool[i];
					break; // Ascending sorted vector
				}
			}
			// Transfer the handle from other thread-owner to this thread-owner
			info.Buffer()->OwnerPool = thisThreadBufferPool;
		}
	} else { // Large buffers don't use the pools
		const size_t validSize = *std::lower_bound( std::begin( BufferSizes ), std::end( BufferSizes ), size );
		ASSERT_EXPR( size == info.Size() || validSize  == info.Size() );
		// No need to transfer, because
		// it wouldn't be cleaned-up for that thread after mathEngine.CleanUp().
	}
}

const CMemoryPool::CThreadData* CMemoryPool::getThreadData() const
{
	auto it = pools.find( std::this_thread::get_id() );
	return ( it == pools.end() ) ? nullptr : &( it->second );
}

CMemoryPool::CThreadData* CMemoryPool::getThreadData( bool forceCreate )
{
	std::thread::id id = std::this_thread::get_id();
	auto it = pools.find( id );
	if( it == pools.end() ) {
		if( !forceCreate ) {
			return nullptr;
		}
		return createPools( id );
	}
	return &( it->second );
}

CMemoryPool::CThreadData* CMemoryPool::createPools( std::thread::id id )
{
	CThreadData threadData;
	threadData.Enabled = defaultReuseMemoryMode;
	for( size_t i = 0; i < sizeof( BufferSizes ) / sizeof( *BufferSizes ); ++i ) {
		threadData.Pool.push_back( new CMemoryBufferPool( BufferSizes[i] ) );
	}
	return &( pools[id] = threadData );
}

void CMemoryPool::cleanUp( CThreadData* threadData )
{
	if( threadData == nullptr ) {
		return;
	}
	for( CMemoryBufferPool* pool : threadData->Pool ) {
		CMemoryBuffer* buffer = pool->TryAlloc();
		while( buffer != 0 ) {
			freeMemory( pool->BufferSize, buffer->Data );
			delete buffer;
			buffer = pool->TryAlloc();
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
	if( !data.Enabled || size > data.BufferMemoryThreshold ) {
		// Allocate without using the buffers pool
		CMemoryHandle result = alloc( size );
		if( !result.IsNull() ) {
			usedMap[GetRaw( result )] = CUsedInfo( size );
			freeMemorySize -= size;
		}
		return result;
	}

	// Allocate via the buffers pool
	CMemoryBufferPool* pool = *std::lower_bound(
		data.Pool.begin(), data.Pool.end(), size, &poolsCompare );

	CMemoryBuffer* buffer = pool->TryAlloc();
	if( buffer == nullptr ) {
		buffer = new CMemoryBuffer( pool );
		buffer->Data = alloc( pool->BufferSize );
		if( buffer->Data.IsNull() ) {
			delete buffer;
			return CMemoryHandle();
		}
	}
	freeMemorySize -= pool->BufferSize;
	usedMap[GetRaw(buffer->Data)] = CUsedInfo( buffer );

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
	peakMemoryUsage = std::max( peakMemoryUsage, allocatedMemory );
	return result;
}

void CMemoryPool::freeMemory( size_t size, const CMemoryHandle& data )
{
	allocatedMemory -= size;
	rawMemoryManager->Free( data );
}

} // namespace NeoML
