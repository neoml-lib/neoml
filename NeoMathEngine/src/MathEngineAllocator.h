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

#include <NeoMathEngine/CrtAllocatedObject.h>
#include <stdlib.h>
#include <type_traits>

// Allocator for std containers
template<class _Ty>
class CrtAllocator : public NeoML::CCrtAllocatedObject {
public:
typedef void _Not_user_specialized;

typedef _Ty value_type;

typedef value_type *pointer;
typedef const value_type *const_pointer;

typedef value_type& reference;
typedef const value_type& const_reference;

typedef size_t size_type;
typedef ptrdiff_t difference_type;

typedef std::true_type propagate_on_container_move_assignment;
typedef std::true_type is_always_equal;

template<class _Other>
struct rebind {	// convert this type to allocator<_Other>
	typedef CrtAllocator<_Other> other;
};

pointer address(reference _Val) const noexcept
{	// return address of mutable _Val
	return (reinterpret_cast<pointer>(
		&const_cast<char&>(
		reinterpret_cast<const volatile char&>(_Val))));
}

const_pointer address(const_reference _Val) const noexcept
{	// return address of nonmutable _Val
	return (reinterpret_cast<const_pointer>(
		&const_cast<const char&>(
		reinterpret_cast<const volatile char&>(_Val))));
}

CrtAllocator() noexcept
{	// construct default allocator (do nothing)
}

CrtAllocator(const CrtAllocator<_Ty>&) noexcept
{	// construct by copying (do nothing)
}

template<class _Other>
CrtAllocator(const CrtAllocator<_Other>&) noexcept
{	// construct from a related allocator (do nothing)
}

template<class _Other>
CrtAllocator<_Ty>& operator=(const CrtAllocator<_Other>&)
{	// assign from a related allocator (do nothing)
	return (*this);
}

void deallocate(pointer _Ptr, size_type)
{
	free( _Ptr );
}

pointer allocate(size_type _Count)
{
	return reinterpret_cast<pointer>( malloc( _Count * sizeof (_Ty) ) );
}

pointer allocate(size_type _Count, const void *)
{
	return allocate(_Count);
}

size_t max_size() const noexcept
{	// estimate maximum array size
	return ((size_t)(-1) / sizeof (_Ty));
}

};

template<class _Ty, class _Other>
inline bool operator==(const CrtAllocator<_Ty>&, const CrtAllocator<_Other>&) noexcept
{	// test for allocator equality
	return true;
}

template<class _Ty, class _Other>
inline bool operator!=(const CrtAllocator<_Ty>&, const CrtAllocator<_Other>&) noexcept
{	// test for allocator equality
	return false;
}

