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

#include <NeoMathEngine/CrtAllocatedObject.h>
#include <NeoMathEngine/NeoMathEngineException.h>
#include <MathEngineStackAllocator.h>
#include <MathEngineAllocator.h>
#include <RawMemoryManager.h>
#include <MemoryHandleInternal.h>
#include <MemoryPool.h>
#include <mutex>
#include <unordered_map>
#include <thread>

namespace NeoML {

static_assert( int( TStackAlloc::Count_ ) == 2, "TStackAlloc: Host and Device are allowed only" );

// Host
CStackMemoryHandle::CStackMemoryHandle( void* host ) :
	handle( CMemoryHandleInternal::CreateMemoryHandle( /*mathEngine*/nullptr, host ) )
{
}

// Device
CStackMemoryHandle::CStackMemoryHandle( CMemoryHandle device ) :
	handle( device )
{
	PRESUME_EXPR( handle.GetMathEngine() != nullptr );
}

// Host
CStackMemoryHandle::operator void*() const
{
	ASSERT_EXPR( handle.GetMathEngine() == nullptr );
	return GetRaw( handle );
}

// Device
CStackMemoryHandle::operator CMemoryHandle() const
{
	ASSERT_EXPR( handle.GetMathEngine() != nullptr );
	return handle;
}

CStackMemoryHandle CStackMemoryHandle::operator+( size_t size ) const
{
	PRESUME_EXPR( size <= size_t( PTRDIFF_MAX ) );
	return CStackMemoryHandle( CTypedMemoryHandle<char>( handle ) + size );
}

int CStackMemoryHandle::operator-( const CStackMemoryHandle& ptr ) const
{
	return CTypedMemoryHandle<char>( handle ) - CTypedMemoryHandle<char>( ptr.handle );
}

//---------------------------------------------------------------------------------------------------------------------

class CStackAllocManager final : public CCrtAllocatedObject {
public:
	explicit CStackAllocManager( CMemoryPool* memoryPool ) : memoryPool( memoryPool ) {}

	TStackAlloc Type() const { return ( memoryPool == nullptr ) ? TStackAlloc::Host : TStackAlloc::Device; }

	CStackMemoryHandle Alloc( size_t size );
	void Free( const CStackMemoryHandle& handle );

private:
	CMemoryPool* const memoryPool;
};

CStackMemoryHandle CStackAllocManager::Alloc( size_t size )
{
	return ( memoryPool != nullptr )
		? CStackMemoryHandle( CMemoryHandle( memoryPool->Alloc( size ) ) )
		: CStackMemoryHandle( malloc( size ) );
}

void CStackAllocManager::Free( const CStackMemoryHandle& ptr )
{
	PRESUME_EXPR( !ptr.IsNull() );
	( memoryPool != nullptr )
		? memoryPool->Free( ptr )
		: free( static_cast<void*>( ptr ) );
}

//---------------------------------------------------------------------------------------------------------------------

// Device memory block used for the stack
class CStackBlock final : public CCrtAllocatedObject  {
public:
	CStackBlock( CStackAllocManager& allocManager, size_t size, CStackBlock* prev );
	~CStackBlock() { manager.Free( buffer ); }

	size_t GetBlockSize() const { return blockSize; }
	size_t GetAllocSize() const { return allocSize; }
	// One-dimensional list
	CStackBlock* Previous() const { return previous; }

	CStackMemoryHandle TryAlloc( size_t size );
	CStackMemoryHandle Alloc( size_t size );
	// Returns the size of released block
	size_t Free( const CStackMemoryHandle& ptr );

private:
	// The memory manager for stack allocation. Tries to keep all allocated memory in one block
	static constexpr int quantum = 64 * 1024; // 64 KB

	CStackAllocManager& manager;
	const size_t blockSize;
	size_t allocSize;
	const CStackMemoryHandle buffer;
	CStackBlock* const previous; // One-dimensional list
};

CStackBlock::CStackBlock( CStackAllocManager& allocManager, size_t size, CStackBlock* prev ) :
	manager( allocManager ),
	blockSize( ( size + quantum - 1 ) / quantum * quantum ),
	allocSize( 0 ),
	buffer( manager.Alloc( blockSize ) ),
	previous( prev )
{}

CStackMemoryHandle CStackBlock::TryAlloc( size_t size )
{
	const size_t newAllocSize = allocSize + size;
	if( newAllocSize > blockSize ) {
		return CStackMemoryHandle{};
	}

	const CStackMemoryHandle result = buffer + allocSize;
	allocSize = newAllocSize;
	return result;
}

CStackMemoryHandle CStackBlock::Alloc( size_t size )
{
	const CStackMemoryHandle result = TryAlloc( size );
	PRESUME_EXPR( !result.IsNull() );
	return result;
}

size_t CStackBlock::Free( const CStackMemoryHandle& ptr )
{
	const int diff = ptr - buffer;
	PRESUME_EXPR( 0 <= diff && diff < static_cast<int>( blockSize ) ); // the pointer belongs to this block

	const size_t result = allocSize - diff;
	allocSize = diff;
	return result;
}

//---------------------------------------------------------------------------------------------------------------------

// The manager class
class CStackBlockManager final : public CCrtAllocatedObject {
public:
	explicit CStackBlockManager( CStackAllocManager& allocManager ) : manager( allocManager ) {}
	~CStackBlockManager() { cleanUpWorker(); }

	void CleanUp();

	CStackMemoryHandle Alloc( size_t size );
	void Free( const CStackMemoryHandle& ptr );

private:
	CStackAllocManager& manager;
	CStackBlock* head = nullptr;
	size_t maxAllocSize = 0;
	size_t curAllocSize = 0;

	void cleanUpWorker();
};

void CStackBlockManager::CleanUp()
{
	PRESUME_EXPR( head == nullptr || ( head->Previous() == nullptr && head->GetAllocSize() == 0 ) );
	cleanUpWorker();
}

CStackMemoryHandle CStackBlockManager::Alloc( size_t size )
{
	curAllocSize += size;
	if( maxAllocSize < curAllocSize ) {
		maxAllocSize = curAllocSize;
	}

	if( head == nullptr
		|| ( head->Previous() == nullptr && head->GetBlockSize() < maxAllocSize && head->GetAllocSize() == 0 ) )
	{
		// Allocate a new block for all required memory
		if( head != nullptr ) {
			head->~CStackBlock();
			::new( head ) CStackBlock( manager, maxAllocSize, nullptr );
		} else {
			head = new CStackBlock( manager, maxAllocSize, nullptr );
		}
		return head->Alloc( size );
	}

	// Try to allocate space in the current block
	CStackMemoryHandle result = head->TryAlloc( size );
	if( !result.IsNull() ) {
		return result;
	}

	// Create a new block
	head = new CStackBlock( manager, size, head );
	return head->Alloc( size );
}

void CStackBlockManager::Free( const CStackMemoryHandle& ptr )
{
	PRESUME_EXPR( head != nullptr );

	const size_t size = head->Free( ptr );
	PRESUME_EXPR( size <= curAllocSize );

	curAllocSize -= size;

	// Delete all free blocks except last free block, to reuse it at the next allocation
	if( head->GetAllocSize() == 0 && head->Previous() != nullptr ) {
		CStackBlock* blockToDelete = head;
		head = head->Previous();
		delete blockToDelete;
	}
}

void CStackBlockManager::cleanUpWorker()
{
	while( head != nullptr ) {
		CStackBlock* blockToDelete = head;
		head = head->Previous();
		delete blockToDelete;
	}
	maxAllocSize = 0;
	curAllocSize = 0;
}

//---------------------------------------------------------------------------------------------------------------------

// Device or Host memory stack implementation for MathEngine
class CStackAllocator : public IStackAllocator {
public:
	CStackAllocator( CMemoryPool* memoryPool, int memoryAlignment );

	TStackAlloc Type() const override { return manager.Type(); }	
	void CleanUp() override;

	CStackMemoryHandle Alloc( size_t size ) override;
	void Free( const CStackMemoryHandle& ptr ) override;

private:
	using TStackBlockManagers = std::unordered_map<
		std::thread::id, CStackBlockManager, // (key, value)
		std::hash<std::thread::id>,
		std::equal_to<std::thread::id>,
		CrtAllocator<std::pair<const std::thread::id, CStackBlockManager>>
	>;

	CStackAllocManager manager;
	const int memoryAlignment;
	TStackBlockManagers stackManagers;
};

CStackAllocator::CStackAllocator( CMemoryPool* _memoryPool, int _memoryAlignment ) :
	manager( _memoryPool ),
	memoryAlignment( _memoryAlignment )
{
}

void CStackAllocator::CleanUp()
{
	auto it = stackManagers.find( std::this_thread::get_id() );
	if( it != stackManagers.end() ) {
		it->second.CleanUp();
	}
}

CStackMemoryHandle CStackAllocator::Alloc( size_t size )
{
	// Align size to keep correct data alignment
	size = ( ( size + memoryAlignment - 1 ) / memoryAlignment ) * memoryAlignment;

	std::thread::id id = std::this_thread::get_id();
	auto it = stackManagers.find( id );
	if( it == stackManagers.end() ) {
		it = stackManagers.emplace( id, CStackBlockManager( manager ) ).first;
	}
	CStackBlockManager& manager = it->second;
	return manager.Alloc(size);
}

void CStackAllocator::Free( const CStackMemoryHandle& ptr )
{
	if( ptr.IsNull() ) {
		return;
	}

	std::thread::id id = std::this_thread::get_id();
	CStackBlockManager& manager = stackManagers.find( id )->second;
	manager.Free(ptr);
}

//---------------------------------------------------------------------------------------------------------------------

IStackAllocator* CreateStackAllocator( TStackAlloc type, CMemoryPool* memoryPool, int memoryAlignment )
{
	ASSERT_EXPR( memoryAlignment > 0 );
	ASSERT_EXPR( ( type == TStackAlloc::Host && memoryPool == nullptr )
			|| ( type == TStackAlloc::Device && memoryPool != nullptr ) );
	return new CStackAllocator( memoryPool, memoryAlignment );
}

void CStackAllocatorDeleter::operator()( IStackAllocator* allocator ) const
{
	delete allocator;
}

} // namespace NeoML

