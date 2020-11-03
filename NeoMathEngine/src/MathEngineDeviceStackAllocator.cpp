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
#include <MathEngineDeviceStackAllocator.h>
#include <RawMemoryManager.h>
#include <MemoryPool.h>

namespace NeoML {

// The memory manager for stack allocation. Tries to keep all allocated memory in one block
static const int StackBlockQuantum = 64 * 1024; // 64 K

// Device memory block used for the stack
class CDeviceStackBlock : public CCrtAllocatedObject  {
public:
	CDeviceStackBlock( CMemoryPool& _memoryManager, size_t size, CDeviceStackBlock* prev ) :
		Prev(prev),
		memoryManager( _memoryManager ),
		blockSize( ( size + StackBlockQuantum - 1 ) / StackBlockQuantum * StackBlockQuantum ),
		allocSize(0)
	{
		buffer = CTypedMemoryHandle<char>( memoryManager.Alloc( blockSize ) );
	}

	~CDeviceStackBlock() { memoryManager.Free( buffer ); }

	size_t GetBlockSize() const { return blockSize; }

	size_t GetAllocSize() const { return allocSize; }

	CMemoryHandle TryAlloc(size_t size)
	{
		size_t newAllocSize = allocSize + size;
		if( newAllocSize > blockSize ) {
			return CMemoryHandle();
		}

		CMemoryHandle result = buffer + allocSize;
		allocSize = newAllocSize;

		return result;
	}

	CMemoryHandle Alloc( size_t size )
	{
		CMemoryHandle res = TryAlloc(size);
		PRESUME_EXPR( !res.IsNull() );
		return res;
	}

	// Returns the size of released block
	size_t Free( const CMemoryHandle& ptr )
	{
		ptrdiff_t diff = CTypedMemoryHandle<char>( ptr ) - buffer;
		PRESUME_EXPR(0 <= diff && diff < (ptrdiff_t)blockSize); // the pointer belongs to this block

		size_t ret = allocSize - diff;
		allocSize = diff;

		return ret;
	}

	CDeviceStackBlock* const Prev;

private:
	CMemoryPool& memoryManager;
	const size_t blockSize;
	size_t allocSize;
	CTypedMemoryHandle<char> buffer;
};

//------------------------------------------------------------------------------------------------------------

// The manager class
class CDeviceStackMemoryManager : public CCrtAllocatedObject {
public:
	explicit CDeviceStackMemoryManager( CMemoryPool& _memoryManager ) :
		memoryManager( _memoryManager ),
		head(0),
		maxAllocSize(0),
		curAllocSize(0)
	{
	}

	~CDeviceStackMemoryManager()
	{
		cleanUpWorker();
	}

	void CleanUp()
	{
		PRESUME_EXPR( head == 0 || ( head->Prev == 0 && head->GetAllocSize() == 0 ) );
		cleanUpWorker();
	}

	CMemoryHandle Alloc(size_t size)
	{
		curAllocSize += size;
		if(maxAllocSize < curAllocSize) {
			maxAllocSize = curAllocSize;
		}

		if( head == 0 || ( head->Prev == 0 && head->GetBlockSize() < maxAllocSize && head->GetAllocSize() == 0 ) ) {
			// Allocate a new block for all required memory
			if(head != 0) {
				delete head;
			}
			head = new CDeviceStackBlock( memoryManager, maxAllocSize, 0 );
			return head->Alloc(size);
		}

		// Try to allocate space in the current block
		CMemoryHandle res = head->TryAlloc(size);
		if( !res.IsNull() ) {
			return res;
		}

		// Create a new block
		head = new CDeviceStackBlock( memoryManager, size, head );
		return head->Alloc(size);
	}

	void Free( const CMemoryHandle& ptr )
	{
		PRESUME_EXPR(head != 0);

		size_t size = head->Free(ptr);
		PRESUME_EXPR(size <= curAllocSize);

		curAllocSize -= size;

		if(head->GetAllocSize() == 0 && head->Prev != 0) {
			CDeviceStackBlock* blockToDelete = head;
			head = head->Prev;
			delete blockToDelete;
		}
	}

private:
	CMemoryPool& memoryManager;
	CDeviceStackBlock *head;
	size_t maxAllocSize;
	size_t curAllocSize;

	void cleanUpWorker()
	{
		while(head != 0) {
			CDeviceStackBlock* blockToDelete = head;
			head = head->Prev;
			delete blockToDelete;
		}
		maxAllocSize = 0;
		curAllocSize = 0;
	}

};

//------------------------------------------------------------------------------------------------------------

CDeviceStackAllocator::CDeviceStackAllocator( CMemoryPool& _memoryPool, int _memoryAlignment ) :
	memoryPool( _memoryPool ),
	memoryAlignment( _memoryAlignment )
{
}

CDeviceStackAllocator::~CDeviceStackAllocator()
{
}

void CDeviceStackAllocator::CleanUp()
{
	auto manager = stackManager.Get();
	if( manager ) {
		manager->CleanUp();
	}
}

CMemoryHandle CDeviceStackAllocator::Alloc( size_t size )
{
	// Align size to keep correct data alignment
	size = ( ( size + memoryAlignment - 1 ) / memoryAlignment ) * memoryAlignment;
	
	if( !stackManager ) {
		stackManager.Reset( new CDeviceStackMemoryManager( memoryPool ) );
	}
	return stackManager->Alloc( size );
}

void CDeviceStackAllocator::Free( const CMemoryHandle& ptr )
{
	if( ptr.IsNull() ) {
		return;
	}

	auto manager = stackManager.Get();
	assert( manager != nullptr );
	manager->Free( ptr );
}

} // namespace NeoML

