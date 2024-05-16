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

#pragma once

#include <NeoMathEngine/NeoMathEngineDefs.h>
#include <NeoMathEngine/MemoryHandle.h>

namespace NeoML {

class CMemoryPool;

// Device or Host memory stack allocator type
enum TTypeStackAlloc { TSA_Host, TSA_Device, TSA_Count_ };

// Stack pointer wrapper for Device or Host memory block
class CStackAllocResult final {
public:
	// Convert ctors
	CStackAllocResult( void* ptr = 0 ) : type( TSA_Host ), host( ptr ) {} // Default ctor
	CStackAllocResult( CMemoryHandle ptr ) : type( TSA_Device ), device( ptr ) {}

	bool IsNull() const { return ( type == TSA_Host ) ? ( host == 0 ) : device.IsNull(); }
	// Casts
	operator CMemoryHandle() { ASSERT_EXPR( type == TSA_Device ); return device; }
	operator void*() { ASSERT_EXPR( type == TSA_Host ); return host; }

private:
	static_assert( TSA_Count_ == 2, "Only TSA_Host and TSA_Device allowed" );
	TTypeStackAlloc type;
	union {
		void* host{};
		CMemoryHandle device;
	};
	friend class CStackAllocManager;
};

//------------------------------------------------------------------------------------------------------------

// Device or Host memory stack allocator implementation for MathEngine
class IStackAllocator : public CCrtAllocatedObject {
public:
	virtual ~IStackAllocator() {}

	virtual TTypeStackAlloc Type() const = 0;
	virtual void CleanUp() = 0;

	virtual CStackAllocResult Alloc( size_t size ) = 0;
	virtual void Free( const CStackAllocResult& ptr ) = 0;
};

//------------------------------------------------------------------------------------------------------------

// Either a memoryPool is equal 0, the stack allocator for the Host memory is crated, or for the Device memory
IStackAllocator* CreateStackAllocator( TTypeStackAlloc type, CMemoryPool* memoryPool, int memoryAlignment );

// Deleter for smart pointers
struct CStackAllocatorDeleter final : public CCrtAllocatedObject {
	void operator()( IStackAllocator* ) const;
};

} // namespace NeoML
