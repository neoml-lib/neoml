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

	explicit CMemoryBufferPool( size_t bufferSize ) : BufferSize(bufferSize), head(0) {}

	// Allocates a buffer; returns 0 if no free buffers are available
	CMemoryBuffer* TryAlloc();

	// Releases a buffer
	void Free( CMemoryBuffer* data );

private:
	// Currently free buffers (a singly-linked list)
	CMemoryBuffer* head;
};

CMemoryBuffer* CMemoryBufferPool::TryAlloc()
{
	CMemoryBuffer* result = head;

	if( result != 0 ) {
		head = result->GetNext();
		result->SetNext(0);
	}

	return result;
}

void CMemoryBufferPool::Free( CMemoryBuffer* data )
{
	data->SetNext( head );
	head = data;
}

//------------------------------------------------------------------------------------------------------------

static constexpr unsigned int BufferSizes[] = {
	256, 512, 1024, 2048, 2048 + 1024, 4096, 4096 + 2048, 8192,
	8192 + 4096, 16384, 16384 + 8192, 32768, 32768 + 16384, 65536, 65536 + 32768, 131072,
	131072 + 65536, 262144, 262144 + 131072, 524288, 524288 + 262144, 1048576, 1048576 + 524288, 2097152,
	2097152 + 1048576, 4194304, 4194304 + 2097152, 8388608, 8388608 + 4194304, 16777216, 16777216 + 8388608, 33554432,
	33554432 + 16777216, 67108864, 67108864 + 33554432, 134217728, 134217728 + 67108864, 268435456, 268435456 + 134217728,
	536870912, 536870912 + 268435456, 1073741824 // 1 GB max
};

template <typename T, int size>
constexpr int lengthof( T(&)[size] ) { return size; }

struct CMemoryPool::CPoolData : CCrtAllocatedObject {
	
	vector<std::unique_ptr<CMemoryBufferPool>> Pool;
	bool Enabled;

	explicit CPoolData( CMemoryPool& _memoryPool, bool reuse ) :
		Enabled( reuse ),
		memoryPool( _memoryPool )
	{
		Pool.reserve( lengthof( BufferSizes ) );
		for( auto size: BufferSizes ) {
			Pool.push_back( std::unique_ptr<CMemoryBufferPool>( new CMemoryBufferPool( size ) ) );
		}
	}

	void CleanUp()
	{
		for( auto& cur : Pool ) {
			CMemoryBuffer* buffer = cur->TryAlloc();
			while( buffer != nullptr ) {
				memoryPool.freeMemory( cur->BufferSize, buffer->Data );
				delete buffer;
				buffer = cur->TryAlloc();
			}
		}
	}

	~CPoolData() { CleanUp(); }

private:
	CMemoryPool& memoryPool;
};

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
}

void CMemoryPool::SetReuseMemoryMode( bool enable )
{
	auto data = poolData.Get();
	if( data == nullptr ) {
		std::unique_ptr<CPoolData> newData( new CPoolData( *this, defaultReuseMemoryMode ) );
		data = poolData.Set( newData.release() );
	}

	data->Enabled = enable;
}

CMemoryHandle CMemoryPool::Alloc( size_t size )
{
	auto data = poolData.Get();
	if( data == nullptr ) {
		std::unique_ptr<CPoolData> newData( new CPoolData( *this, defaultReuseMemoryMode ) );
		data = poolData.Set( newData.release() );
	}

	CMemoryHandle result = tryAlloc( size, *data );
	if( !result.IsNull() ) {
		return result;
	}

	// Not enough memory. Try to free all allocated pools
	CleanUp();
	return tryAlloc( size, *data );
}

void CMemoryPool::Free( const CMemoryHandle& handle )
{
	std::unique_lock<std::mutex> lock( mutex );
	auto pos = usedMap.find( GetRaw( handle ) );
	const CUsedInfo& info = pos->second;
	lock.unlock();

	std::size_t size = 0;
	if( info.pool != 0 ) {
		info.pool->Free( info.buffer );
		size += info.pool->BufferSize;
	} else {
		// Large buffer, don't use the pool
		freeMemory( info.size, handle );
		size += info.size;
	}

	lock.lock();
	freeMemorySize += size;
	usedMap.erase( pos );
}

void CMemoryPool::CleanUp()
{
	auto data = poolData.Get();
	if( data == nullptr ) {
		return;
	}

	data->CleanUp();
}

inline static bool poolsCompare( const std::unique_ptr<CMemoryBufferPool>& a, const size_t& b )
{
	return a->BufferSize < b;
}

// Tries to allocate memory
CMemoryHandle CMemoryPool::tryAlloc( size_t size, CPoolData& data )
{
	if( !data.Enabled || size > BufferSizes[lengthof( BufferSizes ) - 1] ) {
		// Allocate without using the buffers pool
		CMemoryHandle result = alloc( size );
		if( !result.IsNull() ) {
			std::lock_guard<std::mutex> lock( mutex );
			usedMap[GetRaw( result )] = CUsedInfo( size, 0, 0 );
			freeMemorySize -= size;
		}
		return result;
	}

	// Allocate via the buffers pool
	auto pos = std::lower_bound( data.Pool.begin(), data.Pool.end(), size, &poolsCompare );

	CMemoryBufferPool* pool = pos->get();
	CMemoryBuffer* buffer = pool->TryAlloc();
	if( buffer == 0 ) {
		buffer = new CMemoryBuffer();
		buffer->Data = alloc( pool->BufferSize );
		if( buffer->Data.IsNull() ) {
			delete buffer;
			return CMemoryHandle();
		}
	}
	
	{
		std::lock_guard<std::mutex> lock( mutex );
		freeMemorySize -= pool->BufferSize;
		usedMap[GetRaw( buffer->Data )] = CUsedInfo( size, buffer, pool );
	}
	return buffer->Data;
}

CMemoryHandle CMemoryPool::alloc( size_t size )
{
	if( size > memoryLimit ) {
		return CMemoryHandle();
	}
	
	std::lock_guard<std::mutex> lock( mutex );
	if( allocatedMemory > memoryLimit - size ) {
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
	std::lock_guard<std::mutex> lock( mutex );
	
	allocatedMemory -= size;

	rawMemoryManager->Free( data );
}

} // namespace NeoML
