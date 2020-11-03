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
#include <MathEngineHostStackAllocator.h>

namespace NeoML {

// The memory manager for stack allocation. Tries to keep all allocated memory in one block
static constexpr int StackBlockQuantum = 64 * 1024; // 64 K

// Host memory block used for the stack
class CHostStackBlock : public CCrtAllocatedObject  {
public:
	CHostStackBlock( size_t size, CHostStackBlock* prev ) :
		Prev(prev),
		blockSize( ( size + StackBlockQuantum - 1 ) / StackBlockQuantum * StackBlockQuantum ),
		allocSize(0)
	{
		buffer = reinterpret_cast<char*>( malloc( blockSize ) );
	}

	~CHostStackBlock() { free( buffer ); }

	size_t GetBlockSize() const { return blockSize; }

	size_t GetAllocSize() const { return allocSize; }

	void* TryAlloc( size_t size )
	{
		size_t newAllocSize = allocSize + size;
		if( newAllocSize > blockSize ) {
			return 0;
		}

		char* result = buffer + allocSize;
		allocSize = newAllocSize;

		return result;
	}

	void* Alloc( size_t size )
	{
		void* res = TryAlloc(size);
		assert( res != 0 );
		return res;
	}

	// Returns the size of released block
	size_t Free( void* ptr )
	{
		ptrdiff_t diff = reinterpret_cast<char*>( ptr ) - buffer;
		assert(0 <= diff && diff < (ptrdiff_t)blockSize); // the pointer belongs to this block

		size_t ret = allocSize - diff;
		allocSize = diff;

		return ret;
	}

	CHostStackBlock* const Prev;

private:
	const size_t blockSize;
	size_t allocSize;
	char* buffer;
};

//------------------------------------------------------------------------------------------------------------

// The manager class
class CHostStackMemoryManager : public CCrtAllocatedObject {
public:
	explicit CHostStackMemoryManager() : head(0), maxAllocSize(0), curAllocSize(0)
	{
	}

	~CHostStackMemoryManager()
	{
		cleanUpWorker();
	}

	void CleanUp()
	{
		assert( head == 0 || ( head->Prev == 0 && head->GetAllocSize() == 0 ) );
		cleanUpWorker();
	}

	void* Alloc(size_t size)
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
			head = new CHostStackBlock( maxAllocSize, 0 );
			return head->Alloc(size);
		}

		// Try to allocate space in the current block
		void* res = head->TryAlloc(size);
		if( res != 0 ) {
			return res;
		}

		// Create a new block
		head = new CHostStackBlock( size, head );
		return head->Alloc(size);
	}

	void Free( void* ptr )
	{
		assert(head != 0);

		size_t size = head->Free(ptr);
		assert(size <= curAllocSize);

		curAllocSize -= size;

		if( head->GetAllocSize() == 0 && head->Prev != 0 ) {
			CHostStackBlock* blockToDelete = head;
			head = head->Prev;
			delete blockToDelete;
		}
	}

private:
	CHostStackBlock *head;
	size_t maxAllocSize;
	size_t curAllocSize;

	void cleanUpWorker()
	{
		while( head != 0 ) {
			CHostStackBlock* blockToDelete = head;
			head = head->Prev;
			delete blockToDelete;
		}
		maxAllocSize = 0;
		curAllocSize = 0;
	}
};

//------------------------------------------------------------------------------------------------------------

CHostStackAllocator::CHostStackAllocator( int _memoryAlignment ) :
	memoryAlignment( _memoryAlignment )
{
}

CHostStackAllocator::~CHostStackAllocator()
{
}

void CHostStackAllocator::CleanUp()
{
	auto manager = stackManager.Get();
	if( manager ) {
		manager->CleanUp();
	}
}

void* CHostStackAllocator::Alloc( size_t size )
{
	// Align size to keep correct data alignment
	size = ( ( size + memoryAlignment - 1 ) / memoryAlignment ) * memoryAlignment;
	auto manager = stackManager.Get();
	if( manager == nullptr ) {
		manager = stackManager.Set( new CHostStackMemoryManager );
	}
	return manager->Alloc( size );
}

void CHostStackAllocator::Free( void* ptr )
{
	if( ptr == nullptr ) {
		return;
	}

	auto manager = stackManager.Get();
	assert( manager != nullptr );
	manager->Free( ptr );
}

} // namespace NeoML

