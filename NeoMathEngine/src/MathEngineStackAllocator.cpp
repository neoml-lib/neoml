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
#include <MemoryPool.h>
#include <mutex>
#include <unordered_map>
#include <thread>

namespace NeoML {

//------------------------------------------------------------------------------------------------------------

class CStackAllocManager final : public CCrtAllocatedObject {
public:
	explicit CStackAllocManager( CMemoryPool* memoryPool ) : memoryPool( memoryPool ) {}

	TTypeStackAlloc Type() const { return ( memoryPool == nullptr ) ? TSA_Host : TSA_Device; }

	CStackAllocResult Alloc( size_t size );
	void Free( const CStackAllocResult& ptr );

	ptrdiff_t Diff( const CStackAllocResult& buffer, const CStackAllocResult& ptr ) const;
	CStackAllocResult Add( const CStackAllocResult& ptr, size_t value ) const;

private:
	CMemoryPool* const memoryPool;
};

CStackAllocResult CStackAllocManager::Alloc( size_t size )
{
	return ( memoryPool != nullptr )
		? CStackAllocResult( CMemoryHandle( memoryPool->Alloc( size ) ) )
		: CStackAllocResult( malloc( size ) );
}

void CStackAllocManager::Free( const CStackAllocResult& ptr )
{
	PRESUME_EXPR( !ptr.IsNull() );
	( memoryPool != nullptr )
		? memoryPool->Free( ptr.device )
		: free( ptr.host );
}

ptrdiff_t CStackAllocManager::Diff( const CStackAllocResult& buffer, const CStackAllocResult& ptr ) const
{
	return ( memoryPool != nullptr )
		? ( CTypedMemoryHandle<char>( ptr.device ) - CTypedMemoryHandle<char>( buffer.device ) )
		: ( static_cast<char*>( ptr.host ) - static_cast<char*>( buffer.host ) );
}

CStackAllocResult CStackAllocManager::Add( const CStackAllocResult& ptr, size_t value ) const
{
	return ( memoryPool != nullptr )
		? CStackAllocResult( CTypedMemoryHandle<char>( ptr.device ) + value )
		: CStackAllocResult( static_cast<char*>( ptr.host ) + value );
}

//------------------------------------------------------------------------------------------------------------

// Device memory block used for the stack
class CStackBlock final : public CCrtAllocatedObject  {
public:
	// The memory manager for stack allocation. Tries to keep all allocated memory in one block
	static constexpr int StackBlockQuantum = 64 * 1024; // 64 K

	CStackBlock( CStackAllocManager& allocManager, size_t size, CStackBlock* prev );
	~CStackBlock() { manager.Free( buffer ); }

	size_t GetBlockSize() const { return blockSize; }
	size_t GetAllocSize() const { return allocSize; }

	CStackAllocResult TryAlloc( size_t size );
	CStackAllocResult Alloc( size_t size );
	// Returns the size of released block
	size_t Free( const CStackAllocResult& ptr );

	CStackBlock* const Prev;

private:
	CStackAllocManager& manager;
	const size_t blockSize;
	size_t allocSize;
	const CStackAllocResult buffer;
};

CStackBlock::CStackBlock( CStackAllocManager& allocManager, size_t size, CStackBlock* prev ) :
	Prev( prev ),
	manager( allocManager ),
	blockSize( ( size + StackBlockQuantum - 1 ) / StackBlockQuantum * StackBlockQuantum ),
	allocSize( 0 ),
	buffer( manager.Alloc( blockSize ) )
{}

CStackAllocResult CStackBlock::TryAlloc( size_t size )
{
	const size_t newAllocSize = allocSize + size;
	if( newAllocSize > blockSize ) {
		return CStackAllocResult{};
	}

	const CStackAllocResult result = manager.Add( buffer, allocSize );
	allocSize = newAllocSize;
	return result;
}

CStackAllocResult CStackBlock::Alloc( size_t size )
{
	const CStackAllocResult result = TryAlloc( size );
	PRESUME_EXPR( !result.IsNull() );
	return result;
}

size_t CStackBlock::Free( const CStackAllocResult& ptr )
{
	const ptrdiff_t diff = manager.Diff( buffer, ptr );
	PRESUME_EXPR( 0 <= diff && diff < static_cast<ptrdiff_t>( blockSize ) ); // the pointer belongs to this block

	const size_t result = allocSize - diff;
	allocSize = diff;
	return result;
}

//------------------------------------------------------------------------------------------------------------

// The manager class
class CStackBlockManager final : public CCrtAllocatedObject {
public:
	explicit CStackBlockManager( CStackAllocManager& allocManager ) : manager( allocManager ) {}
	~CStackBlockManager() { cleanUpWorker(); }

	void CleanUp();

	CStackAllocResult Alloc( size_t size );
	void Free( const CStackAllocResult& ptr );

private:
	CStackAllocManager& manager;
	CStackBlock* head = nullptr;
	size_t maxAllocSize = 0;
	size_t curAllocSize = 0;

	void cleanUpWorker();
};

void CStackBlockManager::CleanUp()
{
	PRESUME_EXPR( head == nullptr || ( head->Prev == nullptr && head->GetAllocSize() == 0 ) );
	cleanUpWorker();
}

CStackAllocResult CStackBlockManager::Alloc( size_t size )
{
	curAllocSize += size;
	if( maxAllocSize < curAllocSize ) {
		maxAllocSize = curAllocSize;
	}

	if( head == nullptr || ( head->Prev == nullptr && head->GetBlockSize() < maxAllocSize && head->GetAllocSize() == 0 ) ) {
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
	CStackAllocResult result = head->TryAlloc( size );
	if( !result.IsNull() ) {
		return result;
	}

	// Create a new block
	head = new CStackBlock( manager, size, head );
	return head->Alloc( size );
}

void CStackBlockManager::Free( const CStackAllocResult& ptr )
{
	PRESUME_EXPR( head != nullptr );

	const size_t size = head->Free( ptr );
	PRESUME_EXPR( size <= curAllocSize );

	curAllocSize -= size;

	// Delete all free blocks except last free block, to reuse it at the next allocation
	if( head->GetAllocSize() == 0 && head->Prev != nullptr ) {
		CStackBlock* blockToDelete = head;
		head = head->Prev;
		delete blockToDelete;
	}
}

void CStackBlockManager::cleanUpWorker()
{
	while( head != nullptr ) {
		CStackBlock* blockToDelete = head;
		head = head->Prev;
		delete blockToDelete;
	}
	maxAllocSize = 0;
	curAllocSize = 0;
}

//------------------------------------------------------------------------------------------------------------

// Device or Host memory stack implementation for MathEngine
class CStackAllocator : public IStackAllocator {
public:
	CStackAllocator( CMemoryPool* memoryPool, int memoryAlignment );

	TTypeStackAlloc Type() const override { return manager.Type(); }	
	void CleanUp() override;

	CStackAllocResult Alloc( size_t size ) override;
	void Free( const CStackAllocResult& ptr ) override;

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

CStackAllocResult CStackAllocator::Alloc( size_t size )
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

void CStackAllocator::Free( const CStackAllocResult& ptr )
{
	if( ptr.IsNull() ) {
		return;
	}

	std::thread::id id = std::this_thread::get_id();
	CStackBlockManager& manager = stackManagers.find( id )->second;
	manager.Free(ptr);
}

//------------------------------------------------------------------------------------------------------------

IStackAllocator* CreateStackAllocator( TTypeStackAlloc type, CMemoryPool* memoryPool, int memoryAlignment )
{
	ASSERT_EXPR( memoryAlignment > 0 );
	ASSERT_EXPR( ( type == TSA_Host && memoryPool == nullptr )
		|| ( type == TSA_Device && memoryPool != nullptr ) );
	return new CStackAllocator( memoryPool, memoryAlignment );
}

void CStackAllocatorDeleter::operator()( IStackAllocator* allocator ) const
{
	delete allocator;
}

} // namespace NeoML

