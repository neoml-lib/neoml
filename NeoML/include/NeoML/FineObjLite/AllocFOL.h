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

#define ALLOCATE_MEMORY( allocator, size ) allocator::Alloc( ( size ) )

// A common base class for all memory management classes
class IMemoryManager {
public:
	virtual ~IMemoryManager() {}

	virtual void* Alloc( size_t size ) = 0;
	virtual void Free( void* ptr ) = 0;

#ifdef _DEBUG
	virtual void* Alloc( size_t size, const char* file, int line ) = 0;
#endif
};

//-------------------------------------------------------------------------------------------

// Working with memory on the current memory manager
class CurrentMemoryManager {
public:
	static void* Alloc( size_t size );
	static void Free( void* ptr );
};

inline void* CurrentMemoryManager::Alloc( size_t size )
{
	return ::operator new( size );
}

inline void CurrentMemoryManager::Free( void* ptr )
{
	::operator delete( ptr );
}

//-----------------------------------------------------------------------------------------------

// Working with memory via malloc/free
class RuntimeHeap {
public:
	static void* Alloc( size_t size );
	static void Free( void* ptr );
};

inline void* RuntimeHeap::Alloc( size_t size )
{
	return ::malloc( size );
}

inline void RuntimeHeap::Free( void* ptr )
{
	::free( ptr );
}

} // namespace FObj
