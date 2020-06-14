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

#include <NeoMathEngine/NeoMathEngineDefs.h>
#include <MathEngineCommon.h>

#ifdef NEOML_USE_METAL

#include <MetalMathEngine.h>
#include <MemoryHandleInternal.h>
#include <MetalCommandQueue.h>
#include <MathEngineCommon.h>

@import std.vector;

#include <MemoryPool.h>
#include <MathEngineDeviceStackAllocator.h>

@import Foundation;
@import MetalKit;

namespace NeoML {

static size_t defineMemoryLimit()
{
	size_t result = 0;
	id<MTLDevice> device = MTLCreateSystemDefaultDevice();
	if( device != nil ) {
		result = SIZE_MAX;
		[device release];
	}
	return result;
}

bool LoadMetalEngineInfo( CMathEngineInfo& info )
{
	id<MTLDevice> device = MTLCreateSystemDefaultDevice();
	if( device == nil ) {
		return false;
	}

	info.AvailableMemory = SIZE_MAX;
	NSString* name = device.name;
	::memset( info.Name, 0, sizeof( info.Name) );
	const int len = static_cast<int>( MIN( name.length, sizeof( info.Name ) ) );
	::strncpy( info.Name, name.UTF8String, len );
	[device release];
	info.Id = 0;
	info.Type = MET_Metal;
	return true;
}

//----------------------------------------------------------------------------------------------------------------------------

// Not using STL in headers
class CMutex : public std::mutex {
};

//----------------------------------------------------------------------------------------------------------------------------

const int MetalMemoryAlignment = 16;

CMetalMathEngine::CMetalMathEngine( size_t memoryLimit ) :
	queue( new CMetalCommandQueue() ),
	memoryPool( new CMemoryPool( MIN( memoryLimit == 0 ? SIZE_MAX : memoryLimit, defineMemoryLimit() ), this, false ) ),
	deviceStackAllocator( new CDeviceStackAllocator( *memoryPool, MetalMemoryAlignment ) ),
	mutex( new CMutex() )
{
	ASSERT_EXPR( queue->Create() );
}

CMetalMathEngine::~CMetalMathEngine()
{
}

void CMetalMathEngine::SetReuseMemoryMode( bool enable )
{
	std::lock_guard<std::mutex> lock( *mutex );
	memoryPool->SetReuseMemoryMode( enable );
}

CMemoryHandle CMetalMathEngine::HeapAlloc( size_t size )
{
	std::lock_guard<CMutex> lock( *mutex );
	CMemoryHandle result = memoryPool->Alloc( size );
	if( result.IsNull() ) {
		THROW_MEMORY_EXCEPTION;
	}

	return result;
}

void CMetalMathEngine::HeapFree( const CMemoryHandle& handle )
{
	ASSERT_EXPR( handle.GetMathEngine() == this );

	std::lock_guard<CMutex> lock( *mutex );
	return memoryPool->Free( handle );
}

CMemoryHandle CMetalMathEngine::StackAlloc( size_t size )
{
	std::lock_guard<std::mutex> lock( *mutex );
	CMemoryHandle result = deviceStackAllocator->Alloc( size );
	if( result.IsNull() ) {
		THROW_MEMORY_EXCEPTION;
	}
	return result;
}

void CMetalMathEngine::StackFree( const CMemoryHandle& ptr )
{
	std::lock_guard<std::mutex> lock( *mutex );
	deviceStackAllocator->Free( ptr );
}

size_t CMetalMathEngine::GetFreeMemorySize() const
{
	std::lock_guard<CMutex> lock( *mutex );
	return memoryPool->GetFreeMemorySize();
}

size_t CMetalMathEngine::GetPeakMemoryUsage() const
{
	std::lock_guard<std::mutex> lock( *mutex );
	return memoryPool->GetPeakMemoryUsage();
}

void CMetalMathEngine::CleanUp()
{
	std::lock_guard<CMutex> lock( *mutex );
	deviceStackAllocator->CleanUp();
	memoryPool->CleanUp();
}

static void* getBufferPtr( void* buffer, ptrdiff_t offset )
{
	id<MTLBuffer> metalBuffer = (id)buffer;
	return (char *)[metalBuffer contents] + offset;
}

void* CMetalMathEngine::GetBuffer( const CMemoryHandle& handle, size_t pos, size_t /*size*/ )
{
	ASSERT_EXPR( handle.GetMathEngine() == this );

	queue->WaitCommandBuffer();
	return getBufferPtr( GetRawAllocation( handle ), GetRawOffset( handle ) + pos );
}

void CMetalMathEngine::ReleaseBuffer( const CMemoryHandle&, void*, bool )
{
	// Do nothing
}

void CMetalMathEngine::DataExchangeRaw( const CMemoryHandle& handle, const void* data, size_t size )
{
	ASSERT_EXPR( handle.GetMathEngine() == this );

	queue->WaitCommandBuffer();
	void* buf = getBufferPtr( GetRawAllocation( handle ), GetRawOffset( handle ) );
	memcpy( buf, data, size );
}

void CMetalMathEngine::DataExchangeRaw( void* data, const CMemoryHandle& handle, size_t size )
{
	ASSERT_EXPR( handle.GetMathEngine() == this );

	queue->WaitCommandBuffer();
	void* buf = getBufferPtr( GetRawAllocation( handle ), GetRawOffset( handle ) );
	memcpy( data, buf, size );
}

CMemoryHandle CMetalMathEngine::CopyFrom( const CMemoryHandle& handle, size_t size )
{
	CMemoryHandle result = HeapAlloc( size );

	IMathEngine* otherMathEngine = handle.GetMathEngine();
	void* ptr = otherMathEngine->GetBuffer( handle, 0, size );

	DataExchangeRaw( result, ptr, size );

	otherMathEngine->ReleaseBuffer( handle, ptr, false );

	return result;
}

void CMetalMathEngine::VectorCopy( const CFloatHandle& first, const CConstFloatHandle& second, int vectorSize )
{
	ASSERT_EXPR( first.GetMathEngine() == this );
	ASSERT_EXPR( second.GetMathEngine() == this );

	std::unique_ptr<CMetalCommandBuffer> commandBuffer( queue->CreateCommandBuffer() );
	ASSERT_EXPR( commandBuffer->GetHandle() != nil );

	id<MTLBlitCommandEncoder> blitCommandEncoder = [[commandBuffer->GetHandle() blitCommandEncoder] retain];
	ASSERT_EXPR( blitCommandEncoder != nil );

    [blitCommandEncoder copyFromBuffer: (id)GetRawAllocation(second)
                          sourceOffset: GetRawOffset( second )
                              toBuffer: (id)GetRawAllocation( first )
                     destinationOffset: GetRawOffset( first )
                                  size: vectorSize * sizeof(float) ];
    [blitCommandEncoder endEncoding];

    queue->CommitCommandBuffer( commandBuffer.get() );

    [blitCommandEncoder release];
}

void CMetalMathEngine::VectorCopy( const CIntHandle& first, const CConstIntHandle& second, int vectorSize )
{
	ASSERT_EXPR( first.GetMathEngine() == this );
	ASSERT_EXPR( second.GetMathEngine() == this );

	std::unique_ptr<CMetalCommandBuffer> commandBuffer( queue->CreateCommandBuffer() );
	ASSERT_EXPR( commandBuffer->GetHandle() != nil );

	id<MTLBlitCommandEncoder> blitCommandEncoder = [[commandBuffer->GetHandle() blitCommandEncoder] retain];
	ASSERT_EXPR( blitCommandEncoder != nil );

    [blitCommandEncoder copyFromBuffer: (id)GetRawAllocation( second )
                          sourceOffset: GetRawOffset( second )
                              toBuffer: (id)GetRawAllocation( first )
                     destinationOffset: GetRawOffset( first )
                                  size: vectorSize * sizeof(int) ];
    [blitCommandEncoder endEncoding];

    queue->CommitCommandBuffer( commandBuffer.get() );

    [blitCommandEncoder release];
}

CMemoryHandle CMetalMathEngine::Alloc( size_t size )
{
	// Use the common memory because it's easier than using MTLResourceStorageModePrivate and the performance doesn't differ
	id<MTLBuffer> buffer = [queue->GetDevice() newBufferWithLength: size options: MTLResourceStorageModeShared];
	if( buffer == nil ) {
		return CMemoryHandle();
	}
	return CMemoryHandleInternal::CreateMemoryHandle( &mathEngine(), buffer );
}

void CMetalMathEngine::Free( const CMemoryHandle& handle )
{
	id<MTLBuffer> buffer = (id)CMemoryHandleInternal::GetRawAllocation( handle );
	[buffer release];
}

void CMetalMathEngine::GetMathEngineInfo( CMathEngineInfo& info ) const
{
	LoadMetalEngineInfo( info );
}

} // namespace NeoML

#endif // NEOML_USE_METAL
