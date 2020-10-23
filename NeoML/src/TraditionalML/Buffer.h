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

namespace NeoML {

template<typename T, typename TMemoryManager = CurrentMemoryManager>
class CBuffer {
public:
	// Allocates buffer with elementsCount elements
	explicit CBuffer( int elementsCount );
	// Allocates buffer and copies elementsCount elements from src
	CBuffer( const T* src, int elementsCount );
	~CBuffer() { free(); }

	// Copies elementsCount elements from src
	void CopyFrom( const T* src, int elementsCount );
	// Copies elementsCount elements from src, then swaps buffers with src
	void CopyFromAndReplace( T*& src, int elementsCount );

	// Detaches pointer so that it won't be deallocated on destruction
	T* Detach();

	CBuffer( const CBuffer& ) = delete;
	CBuffer& operator=( CBuffer ) = delete;

private:
	T* ptr;

	void alloc( int size );
	void free();
};

template<typename T, typename TMemoryManager>
CBuffer<T, TMemoryManager>::CBuffer( int elementsCount )
{
	alloc( elementsCount );
}

template<typename T, typename TMemoryManager>
CBuffer<T, TMemoryManager>::CBuffer( const T* src, int elementsCount )
{
	alloc( elementsCount );
	CopyFrom( src, elementsCount );
}

template<typename T, typename TMemoryManager>
inline void CBuffer<T, TMemoryManager>::CopyFrom( const T* src, int elementsCount )
{
	::memcpy( ptr, src, elementsCount * sizeof( T ) );
}

template<typename T, typename TMemoryManager>
inline void CBuffer<T, TMemoryManager>::CopyFromAndReplace( T*& src, int elementsCount )
{
	CopyFrom( src, elementsCount );
	swap( src, ptr );
}

template<typename T, typename TMemoryManager>
inline T* CBuffer<T, TMemoryManager>::Detach()
{
	T* res = ptr;
	ptr = 0;
	return res;
}

// Allocates elementsCount elements of type T on TMemoryManager allocator
template<typename T, typename TMemoryManager>
inline void CBuffer<T, TMemoryManager>::alloc( int elementsCount )
{
	ptr = static_cast<T*>( ALLOCATE_MEMORY( TMemoryManager, elementsCount * sizeof( T ) ) );
}

// If not detached, deallocates controlled buffer
template<typename T, typename TMemoryManager>
inline void CBuffer<T, TMemoryManager>::free()
{
	TMemoryManager::Free( ptr );
}

} // namespace NeoML
