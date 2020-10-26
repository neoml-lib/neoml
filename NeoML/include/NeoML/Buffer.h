/* Copyright © 2020 ABBYY Production LLC

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

#include <NeoML/NeoMLDefs.h>

namespace NeoML {

// A tiny wrapper over a raw buffer
template<typename T, typename TMemoryManager = CurrentMemoryManager>
class CBuffer {
public:
	CBuffer() : ptr( nullptr ) {}
	CBuffer( CBuffer&& other ) : ptr( nullptr ) { *this = other; }
	// Allocates buffer with elementsCount elements
	explicit CBuffer( int elementsCount );
	~CBuffer() { free(); }

	// Allow construction from an rvalue reference
	CBuffer& operator=( CBuffer&& other );

	// Copies elementsToCopy elements from src
	void CopyFrom( const T* src, int elementsToCopy );

	// Swaps buffers
	void Swap( CBuffer& buf ) { swap( buf.ptr, ptr ); }

	// Detaches pointer so that it won't be deallocated on destruction
	T* Detach();

	// Implicit convertion to T*
	operator T*() const { return ptr; }

	CBuffer( const CBuffer& ) = delete;
	CBuffer& operator=( const CBuffer&& ) = delete;

private:
	T* ptr;

	void alloc( int elementsCount );
	void free();
};

template<typename T, typename TMemoryManager>
CBuffer<T, TMemoryManager>::CBuffer( int elementsCount )
{
	alloc( elementsCount );
}

template<typename T, typename TMemoryManager>
inline CBuffer<T, TMemoryManager>& CBuffer<T, TMemoryManager>::operator=( CBuffer&& other )
{
	NeoPresume( ptr == nullptr );
	Swap( other );
	return *this;
}

template<typename T, typename TMemoryManager>
inline void CBuffer<T, TMemoryManager>::CopyFrom( const T* src, int elementsToCopy )
{
	::memcpy( ptr, src, elementsToCopy * sizeof( T ) );
}

template<typename T, typename TMemoryManager>
inline T* CBuffer<T, TMemoryManager>::Detach()
{
	T* res = ptr;
	ptr = nullptr;
	return res;
}

// Allocates elementsCount elements of type T on TMemoryManager allocator
template<typename T, typename TMemoryManager>
inline void CBuffer<T, TMemoryManager>::alloc( int elementsCount )
{
	ptr = static_cast<T*>( ALLOCATE_MEMORY( TMemoryManager, elementsCount * sizeof( T ) ) );
}

// Deallocates controlled buffer
template<typename T, typename TMemoryManager>
inline void CBuffer<T, TMemoryManager>::free()
{
	TMemoryManager::Free( ptr );
}

} // namespace NeoML
