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

#ifdef NEOML_USE_NCCL

#include <nccl.h>
#include <atomic>
#include <NeoMathEngine/NeoMathEngine.h>
#include <NcclFunctions.h>

namespace NeoML {

class CCudaDistributedCommunicator {
public:
    CCudaDistributedCommunicator( const ncclUniqueId& uniqueId, const CMathEngineDistributedInfo& info, std::shared_ptr<std::atomic<bool>> isAbort );
    void AllReduce( const CFloatHandle& handle, int size );
    void Broadcast( const CFloatHandle& handle, int size, int root );
    void Abort();
    ~CCudaDistributedCommunicator();
private:
    ncclComm_t comm;
    const CNccl* nccl;
    std::shared_ptr<std::atomic<bool>> isAbort;
    CDllLoader ncclLoader;

    void ncclStreamSynchronize( cudaStream_t stream );
};

void CreateDistributedCudaMathEnginesNccl( IMathEngine** mathEngines, int devsCount, const int* cudaDevs );

} // namespace NeoML

#endif