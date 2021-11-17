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

#ifdef NEOML_USE_NCCL

#include "cuda_runtime.h"
#include <CudaMathEngine.h>
#include <NeoMathEngine/NeoMathEngineException.h>
#include <CudaMathEngineDnnDistributed.h>
#include <MemoryHandleInternal.h>
#include <CudaCommon.h>
#include <DllLoader.h>

namespace NeoML {

#define ASSERT_NCCL( nccl, expr ) \
	do { \
		int _err_ = ( int ) ( expr ); \
		if( _err_ != ncclSuccess ) { \
			GetMathEngineExceptionHandler()->OnAssert( nccl->GetErrorString( static_cast<ncclResult_t>( _err_ ) ), __UNICODEFILE__, __LINE__, _err_ ); \
		} \
	} while(0)

CCudaDistributedCommunicator::CCudaDistributedCommunicator( const ncclUniqueId& uniqueId, const CMathEngineDistributedInfo& info )
{
    CDllLoader::Load(CDllLoader::NCCL_DLL);
    nccl = CDllLoader::ncclDll->GetFunctions();
    ASSERT_NCCL( nccl, nccl->CommInitRank( &comm, info.Threads, uniqueId, info.Thread ) );
}

CCudaDistributedCommunicator::~CCudaDistributedCommunicator()
{
    ASSERT_NCCL( nccl, nccl->CommDestroy( comm ) );
    CDllLoader::Free(CDllLoader::NCCL_DLL);
}

void CCudaDistributedCommunicator::AllReduce( const CFloatHandle& handle, int size )
{
    ASSERT_NCCL( nccl, nccl->AllReduce( ( const void* )GetRaw( handle ), (void*)GetRaw( handle ), size,
        ncclFloat, ncclAvg, comm, 0 ) );
}

void CCudaDistributedCommunicator::Broadcast( const CFloatHandle& handle, int size, int root )
{
    ASSERT_NCCL( nccl, nccl->Broadcast( ( const void* )GetRaw( handle ), (void*)GetRaw( handle ), size,
        ncclFloat, root, comm, 0 ) );
}

void CreateDistributedCudaMathEnginesNccl( IMathEngine** mathEngines, int devsCount, const int* cudaDevs )
{
    std::unique_ptr<IGpuMathEngineManager> gpuManager( CreateGpuMathEngineManager() );
    const CNccl* nccl = CDllLoader::ncclDll->GetFunctions();
    if( nccl == nullptr ){
        return;
    }
    ncclUniqueId id;
    nccl->GetUniqueId( &id );
    ASSERT_NCCL( nccl, nccl->GroupStart() );
    for( int i = 0; i < devsCount; i++ ){
        mathEngines[i]  = gpuManager->CreateMathEngine( cudaDevs[i], 0u );
        SetCudaDevice( cudaDevs[i] );
        static_cast<CCudaMathEngine*>( mathEngines[i] )->SetDistributedCommunicator( id, {i, devsCount} );
    }
    ASSERT_NCCL( nccl, nccl->GroupEnd() );
}

} // namespace NeoML

#endif