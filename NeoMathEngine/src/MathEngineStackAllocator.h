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
enum class TStackAlloc { Host, Device, Count_ };

// Stack pointer wrapper for Device or Host memory block
class CStackMemoryHandle final {
public:
	// Convert ctors
	CStackMemoryHandle( void* host = nullptr ); // default ctor
	CStackMemoryHandle( CMemoryHandle device );
	// Be copied and moved by default

	bool IsNull() const { return handle.IsNull(); }
	// Types casts
	operator void*() const; // host
	operator CMemoryHandle() const; // device

	CStackMemoryHandle operator+( size_t size ) const;
	int operator-( const CStackMemoryHandle& ptr ) const;

private:
	const CMemoryHandle handle;

	CStackMemoryHandle( CTypedMemoryHandle<char>&& ptr ) : handle( ptr ) {}
};

//------------------------------------------------------------------------------------------------------------

// Device or Host memory stack allocator implementation for MathEngine
class IStackAllocator : public CCrtAllocatedObject {
public:
	virtual ~IStackAllocator() = default;

	virtual TStackAlloc Type() const = 0;
	virtual void CleanUp() = 0;

	virtual CStackMemoryHandle Alloc( size_t size ) = 0;
	virtual void Free( const CStackMemoryHandle& ptr ) = 0;
};

//------------------------------------------------------------------------------------------------------------

// Either a memoryPool is equal 0, the stack allocator for the Host memory is crated, or for the Device memory
IStackAllocator* CreateStackAllocator( TStackAlloc type, CMemoryPool* memoryPool, int memoryAlignment );

// Deleter for smart pointers
struct CStackAllocatorDeleter final : public CCrtAllocatedObject {
	void operator()( IStackAllocator* ) const;
};

} // namespace NeoML
