/* Copyright © 2017-2024 ABBYY

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

#include <NeoMathEngine/MemoryHandle.h>
#ifdef NEOML_USE_VULKAN
#include <VulkanMemory.h>
#endif // NEOML_USE_VULKAN

namespace NeoML {

// Operations with CMemoryHandle inside MathEngine
class CMemoryHandleInternal : public CCrtAllocatedObject {
public:

#ifdef NEOML_USE_VULKAN
	static CVulkanMemory* GetRawAllocation( const CMemoryHandle& handle )
		{ return reinterpret_cast<CVulkanMemory*>( const_cast<void*>( handle.Object ) ); }
#endif // NEOML_USE_VULKAN

#ifdef NEOML_USE_METAL
	static void* GetRawAllocation( const CMemoryHandle& handle )
		{ return const_cast<void*>( handle.Object ); }
#endif // NEOML_USE_METAL

#if (defined NEOML_USE_METAL) | (defined NEOML_USE_VULKAN)
	static ptrdiff_t GetRawOffset( const CMemoryHandle& handle ) { return handle.Offset; }
#endif // NEOML_USE_METAL || NEOML_USE_VULKAN

	template<class Type>
	static Type* GetRaw( const CTypedMemoryHandle<Type>& handle )
		{ return const_cast<Type*>( reinterpret_cast<const Type*>( reinterpret_cast<const char*>( handle.Object ) + handle.Offset ) ); }

	template<class Type>
	static const Type* GetRaw( const CTypedMemoryHandle<const Type>& handle )
		{ return reinterpret_cast<const Type*>( reinterpret_cast<const char*>( handle.Object ) + handle.Offset ); }

	static void* GetRaw( const CMemoryHandle& handle )
		{ return const_cast<void*>( reinterpret_cast<const void*>( reinterpret_cast<const char*>( handle.Object ) + handle.Offset ) ); }

	static CMemoryHandle CreateMemoryHandle( IMathEngine* mathEngine, const void* object ) { return CMemoryHandle( mathEngine, object, 0 ); }
};

#ifdef NEOML_USE_METAL
inline static void* GetRawAllocation( const CMemoryHandle& handle )
{ 
	return CMemoryHandleInternal::GetRawAllocation( handle );
}
#endif // NEOML_USE_METAL

#ifdef NEOML_USE_VULKAN
inline static CVulkanMemory* GetRawAllocation( const CMemoryHandle& handle )
{ 
	return CMemoryHandleInternal::GetRawAllocation( handle );
}
#endif // NEOML_USE_VULKAN

#if (defined NEOML_USE_METAL) | (defined NEOML_USE_VULKAN)
inline static ptrdiff_t GetRawOffset( const CMemoryHandle& handle )
{ 
	return CMemoryHandleInternal::GetRawOffset( handle );
}
#endif // NEOML_USE_METAL || NEOML_USE_VULKAN

template<class Type>
inline static Type* GetRaw( const CTypedMemoryHandle<Type>& handle )
{
	return CMemoryHandleInternal::GetRaw( handle );
}

template<class Type>
inline static const Type* GetRaw( const CTypedMemoryHandle<const Type>& handle )
{
	return CMemoryHandleInternal::GetRaw( handle );
}

inline static void* GetRaw( const CMemoryHandle& handle )
{
	return CMemoryHandleInternal::GetRaw( handle );
}

} // namespace NeoML
