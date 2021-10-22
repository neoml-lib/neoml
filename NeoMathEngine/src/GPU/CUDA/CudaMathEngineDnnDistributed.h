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

#include "nccl.h"
#include <NeoMathEngine/NeoMathEngine.h>
#include <CudaMathEngine.h>
#include <vector>

namespace NeoML {

class CCudaDistributedCommunicator : public IDistributedCommunicator {
public:
    CCudaDistributedCommunicator( ncclComm_t _comm )
        : comm( _comm ) {};
    void AllReduce( const CFloatHandle& handle, int size ) override;
    ~CCudaDistributedCommunicator();
private:
    ncclComm_t comm;
};

void CreateDistributedCudaMathEngines( std::vector<std::unique_ptr<IMathEngine>>& mathEngines, int count, std::initializer_list<int> devs );

} // namespace NeoML

#endif