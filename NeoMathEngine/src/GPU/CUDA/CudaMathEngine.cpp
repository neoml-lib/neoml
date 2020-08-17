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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_CUDA

#include <CudaMathEngine.h>
#include <cublas.h>
#include <cusparse.h>
#include <MathEngineAllocator.h>
#include <MathEngineCommon.h>
#include <MemoryHandleInternal.h>
#include <MathEngineDeviceStackAllocator.h>
#include <MathEngineHostStackAllocator.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <CudaDevice.h>
#include <CudaAssert.h>

namespace NeoML {

void CCudaMathEngine::CleanUp()
{
	std::lock_guard<std::mutex> lock( mutex );
	deviceStackRunTime->CleanUp();
	hostStackRunTime->CleanUp();
	memoryPool->CleanUp();
}

size_t CCudaMathEngine::GetFreeMemorySize() const
{
	std::lock_guard<std::mutex> lock( mutex );
	return memoryPool->GetFreeMemorySize();
}

size_t CCudaMathEngine::GetPeakMemoryUsage() const
{
	std::lock_guard<std::mutex> lock( mutex );
	return memoryPool->GetPeakMemoryUsage();
}

void CCudaMathEngine::SetReuseMemoryMode( bool )
{
	// Always true, because allocation is sync
}

CMemoryHandle CCudaMathEngine::HeapAlloc( size_t size )
{
	std::lock_guard<std::mutex> lock( mutex );
	CMemoryHandle result = memoryPool->Alloc( size );
	if( result.IsNull() ) {
		THROW_MEMORY_EXCEPTION;
	}
	return result;
}

void CCudaMathEngine::HeapFree( const CMemoryHandle& handle )
{
	ASSERT_EXPR( handle.GetMathEngine() == this );

	std::lock_guard<std::mutex> lock( mutex );
	return memoryPool->Free( handle );
}

CMemoryHandle CCudaMathEngine::StackAlloc( size_t size )
{
	ASSERT_EXPR( deviceStackRunTime != 0 );

	std::lock_guard<std::mutex> lock( mutex );
	CMemoryHandle result = deviceStackRunTime->Alloc( size );
	if( result.IsNull() ) {
		THROW_MEMORY_EXCEPTION;
	}
	return result;
}

void CCudaMathEngine::StackFree( const CMemoryHandle& ptr )
{
	ASSERT_EXPR(ptr.GetMathEngine() == this);

	std::lock_guard<std::mutex> lock( mutex );
	deviceStackRunTime->Free( ptr );
}

void* CCudaMathEngine::GetBuffer( const CMemoryHandle& handle, size_t pos, size_t size )
{
	ASSERT_EXPR(handle.GetMathEngine() == this);

	size_t realSize = size + 16;
	char* result = reinterpret_cast<char*>( hostStackRunTime->Alloc( realSize ) );
	size_t* posPtr = reinterpret_cast<size_t*>( result );
	*posPtr = pos;
	size_t* sizePtr = reinterpret_cast<size_t*>( result ) + 1;
	*sizePtr = size;
	DataExchangeRaw( result + 16, handle, size );
	return result + 16;
}

void CCudaMathEngine::ReleaseBuffer( const CMemoryHandle& handle, void* ptr, bool exchange )
{
	ASSERT_EXPR(handle.GetMathEngine() == this);

	if( exchange ) {
		size_t* posPtr = reinterpret_cast<size_t*>( reinterpret_cast<char*>( ptr ) - 16 );
		size_t pos = *posPtr;
		size_t* sizePtr = posPtr + 1;
		size_t size = *sizePtr;

		DataExchangeRaw( CTypedMemoryHandle<char>( handle ) + pos, ptr, size );
	}

	hostStackRunTime->Free( reinterpret_cast<char*>( ptr ) - 16 );
}

void CCudaMathEngine::DataExchangeRaw(const CMemoryHandle& handle, const void* data, size_t size)
{
	ASSERT_EXPR(handle.GetMathEngine() == this);
	ASSERT_CUDA(cudaMemcpy(GetRaw(handle), data, size, cudaMemcpyHostToDevice));
}

void CCudaMathEngine::DataExchangeRaw(void* data, const CMemoryHandle& handle, size_t size)
{
	ASSERT_EXPR(handle.GetMathEngine() == this);
	ASSERT_CUDA(cudaMemcpy(data, GetRaw(handle), size, cudaMemcpyDeviceToHost));
}

CMemoryHandle CCudaMathEngine::CopyFrom( const CMemoryHandle& handle, size_t size )
{
	CMemoryHandle result = HeapAlloc( size );

	IMathEngine* otherMathEngine = handle.GetMathEngine();
	void* ptr = otherMathEngine->GetBuffer( handle, 0, size );

	DataExchangeRaw( result, ptr, size );

	otherMathEngine->ReleaseBuffer( handle, ptr, false );

	return result;
}

CMemoryHandle CCudaMathEngine::Alloc( size_t size )
{
	cudaSetDevice( device->DeviceNumber );
	void* ptr;
	cudaError_t mallocError = cudaMalloc(&ptr, size);
	if( mallocError != 0 ) {
		return CMemoryHandle();
	}
	return CMemoryHandleInternal::CreateMemoryHandle( this, ptr );
}

void CCudaMathEngine::Free( const CMemoryHandle& handle )
{
	ASSERT_CUDA( cudaFree( GetRaw( CTypedMemoryHandle<char>( handle ) ) ) );
}

void CCudaMathEngine::GetMathEngineInfo( CMathEngineInfo& info ) const
{
	cudaDeviceProp devProp;
	cudaGetDeviceProperties( &devProp, device->DeviceNumber );
	info.Type = MET_Cuda;
	info.Id = device->DeviceNumber;
	info.AvailableMemory = devProp.totalGlobalMem;
	::memset( info.Name, 0, sizeof( info.Name ) );
	::strcpy( info.Name, devProp.name );
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
