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
#include "cuda_runtime.h"
#pragma hdrstop

#include <NeoMathEngine/NeoMathEngineException.h>
#include <CudaMathEngineDnnDistributed.h>
#include <MemoryHandleInternal.h>

#ifdef NEOML_USE_NCCL

namespace NeoML {

#define ASSERT_NCCL( expr ) \
	do { \
		int _err_ = ( int ) ( expr ); \
		if( _err_ != ncclSuccess ) { \
			GetMathEngineExceptionHandler()->OnAssert( ncclGetErrorString( static_cast<ncclResult_t>( _err_ ) ), __UNICODEFILE__, __LINE__, _err_ ); \
		} \
	} while(0)

CCudaDistributedCommunicator::~CCudaDistributedCommunicator()
{
    ASSERT_NCCL( ncclCommDestroy( comm ) );
}

void CCudaDistributedCommunicator::AllReduce( const CFloatHandle& handle, int size )
{
    ASSERT_NCCL( ncclAllReduce( ( const void* )GetRaw( handle ), (void*)GetRaw( handle ), size,
        ncclFloat, ncclAvg, comm, 0 ) );
}

void CreateDistributedCudaMathEngines( std::vector<std::unique_ptr<IMathEngine>>& mathEngines, int count, std::initializer_list<int> devs )
{
    std::unique_ptr<IGpuMathEngineManager> gpuManager( CreateGpuMathEngineManager() );
    mathEngines.resize( devs.size() );
    vector<ncclComm_t> comms( devs.size() );
    ASSERT_NCCL( ncclCommInitAll( comms.data(), count, devs.begin() ) );
    int ind = 0;
    for( int dev : devs ){
        mathEngines[ind].reset( gpuManager->CreateMathEngine( dev, 0u ) );
        auto comm = make_shared<CCudaDistributedCommunicator>( comms[ind] );
        mathEngines[ind]->SetDistributedCommunicator( comm, {ind, count} );
        ind++;
    }
}

} // namespace NeoML

#endif