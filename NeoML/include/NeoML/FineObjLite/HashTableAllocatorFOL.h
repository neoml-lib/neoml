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

namespace FObj {

struct CHashTableAllocatorFreeBlock {
	CHashTableAllocatorFreeBlock* NextBlock;
};

struct CHashTableAllocatorPage {
	CHashTableAllocatorPage* PrevPage;
	int DataSize;
	char* Data()
	{
		return reinterpret_cast<char*>( this + 1 );
	}
};

const int MinHashTableAllocatorBlockSize = sizeof( CHashTableAllocatorFreeBlock );

template<class BaseAllocator, int BlockSize>
class CHashTableAllocator {
public:
	CHashTableAllocator();
	~CHashTableAllocator();

	void* Alloc();

	void Free( void* block ); 

	void MoveTo( CHashTableAllocator& dest );

	void Reserve( int blocksCount );

	void FreeBuffer();

private:
	enum {
		BlocksInFirstPage = 16,
		MinPageDataSize = BlockSize * BlocksInFirstPage,
		MaxPageDataSize = 1 << 20
	};

	CHashTableAllocatorPage* currentPage;
	CHashTableAllocatorFreeBlock* firstFreeBlock;
	int allocatedInCurrentPage;
	int nextPageDataSize;

	void allocPage();
	void freeAllPages();

	CHashTableAllocator( const CHashTableAllocator& );
	void operator=( const CHashTableAllocator& );
};

template<class BaseAllocator, int BlockSize>
CHashTableAllocator<BaseAllocator, BlockSize>::CHashTableAllocator() :
	currentPage( 0 ),
	firstFreeBlock( 0 ),
	allocatedInCurrentPage( 0 )
{
	nextPageDataSize = MinPageDataSize;
}

template<class BaseAllocator, int BlockSize>
CHashTableAllocator<BaseAllocator, BlockSize>::~CHashTableAllocator()
{
	freeAllPages();
}

template<class BaseAllocator, int BlockSize>
inline void* CHashTableAllocator<BaseAllocator, BlockSize>::Alloc()
{
	if( firstFreeBlock == 0 ) {
		if( currentPage == 0 || allocatedInCurrentPage + BlockSize > currentPage->DataSize ) {
			allocPage();
		}
		void* ret = currentPage->Data() + allocatedInCurrentPage;
		allocatedInCurrentPage += BlockSize;
		return ret;
	} else {
		void* ptr = firstFreeBlock;
		firstFreeBlock = firstFreeBlock->NextBlock;
		return ptr;
	}
}

template<class BaseAllocator, int BlockSize>
inline void CHashTableAllocator<BaseAllocator, BlockSize>::Free( void* block )
{
	PresumeFO( block != 0 );
	CHashTableAllocatorFreeBlock* freeHeader = new( block ) CHashTableAllocatorFreeBlock;
	freeHeader->NextBlock = firstFreeBlock;
	firstFreeBlock = freeHeader;
}

template<class BaseAllocator, int BlockSize>
inline void CHashTableAllocator<BaseAllocator, BlockSize>::MoveTo( CHashTableAllocator& dest )
{
	PresumeFO( &dest != this );
	dest.FreeBuffer();

	dest.currentPage = currentPage;
	dest.firstFreeBlock = firstFreeBlock;
	dest.allocatedInCurrentPage = allocatedInCurrentPage;
	dest.nextPageDataSize = nextPageDataSize;
	currentPage = 0;
	firstFreeBlock = 0;
	allocatedInCurrentPage = 0;
	nextPageDataSize = MinPageDataSize;
}

template<class BaseAllocator, int BlockSize>
void CHashTableAllocator<BaseAllocator, BlockSize>::Reserve(int blocksCount)
{
	int size = blocksCount * BlockSize;
	int totalDataSize = 0;
	for(CHashTableAllocatorPage* page = currentPage; page != 0; page = page->PrevPage) {
		totalDataSize += page->DataSize;
	}
	if(totalDataSize < size) {
		for(; nextPageDataSize <= MaxPageDataSize; nextPageDataSize <<= 1) {
			if(size <= totalDataSize + nextPageDataSize) {
				break;
			}
		}
	}
}

template<class BaseAllocator, int BlockSize>
void CHashTableAllocator<BaseAllocator, BlockSize>::FreeBuffer()
{
	firstFreeBlock = 0;
	allocatedInCurrentPage = 0;
	freeAllPages();
	nextPageDataSize = MinPageDataSize;
}

template<class BaseAllocator, int BlockSize>
void CHashTableAllocator<BaseAllocator, BlockSize>::allocPage()
{
	CHashTableAllocatorPage* page = static_cast<CHashTableAllocatorPage*>( 
		ALLOCATE_MEMORY( BaseAllocator, sizeof( CHashTableAllocatorPage ) + nextPageDataSize ) );
	page->PrevPage = currentPage;
	page->DataSize = nextPageDataSize;
	currentPage = page;
	allocatedInCurrentPage = 0;
	nextPageDataSize = min((currentPage->DataSize << 1), static_cast<int>(MaxPageDataSize));
}

template<class BaseAllocator, int BlockSize>
inline void CHashTableAllocator<BaseAllocator, BlockSize>::freeAllPages()
{
	while( currentPage != 0 ) {
		CHashTableAllocatorPage* page = currentPage;
		currentPage = page->PrevPage;
		BaseAllocator::Free( page );
	}
}

} // namespace FObj
