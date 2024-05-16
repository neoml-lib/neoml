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
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <CudaDevice.h>
#include <CudaAssert.h>
#include <CudaCommon.h>

namespace NeoML {

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

CMemoryHandle CCudaMathEngine::Alloc( size_t size )
{
	SetCudaDevice( device->DeviceNumber );
	void* ptr;
	cudaError_t mallocError = cudaMalloc(&ptr, size);
	if( mallocError != 0 ) {
		return CMemoryHandle();
	}
	return CMemoryHandleInternal::CreateMemoryHandle( this, ptr );
}

void CCudaMathEngine::Free( const CMemoryHandle& handle )
{
	cudaFree( GetRaw( CTypedMemoryHandle<char>( handle ) ) );
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

void CCudaMathEngine::AllReduce( const CFloatHandle& handle, int size )
{
	ASSERT_EXPR( handle.GetMathEngine() == this );
	ASSERT_EXPR( size >= 0 );
#ifdef NEOML_USE_NCCL
	if( ncclCommunicator != nullptr ){
		ncclCommunicator->AllReduce( handle, size );
	}
#endif //NEOML_USE_NCCL
}

void CCudaMathEngine::AbortDistributed()
{
#ifdef NEOML_USE_NCCL
	if( ncclCommunicator != nullptr ){
		ncclCommunicator->Abort();
	}
#endif //NEOML_USE_NCCL
}

void CCudaMathEngine::Broadcast( const CFloatHandle& handle, int size, int root )
{
	ASSERT_EXPR( handle.GetMathEngine() == this );
	ASSERT_EXPR( size >= 0 );
	ASSERT_EXPR( root >= 0 );
#ifdef NEOML_USE_NCCL
	if( ncclCommunicator != nullptr ){
		ncclCommunicator->Broadcast( handle, size, root );
	}
#endif //NEOML_USE_NCCL
}

#ifdef NEOML_USE_NCCL
void CCudaMathEngine::SetDistributedCommunicator( const ncclUniqueId& uniqueId, const CMathEngineDistributedInfo& info,
	std::shared_ptr<std::atomic<bool>> isAbort )
{
	ncclCommunicator = std::make_unique<CCudaDistributedCommunicator>( uniqueId, info, isAbort );
	distributedInfo = info;
}
#endif //NEOML_USE_NCCL

} // namespace NeoML

#endif // NEOML_USE_CUDA
