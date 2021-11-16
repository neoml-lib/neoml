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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_NCCL

#include <nccl.h>

namespace NeoML {

// The nccl functions used in CUDA implementation of the MathEngine
struct CNccl {
	// typedef for convenience
    typedef ncclResult_t( *TNcclCommInitAll ) ( ncclComm_t* comms, int ndev, const int* devlist );
    typedef ncclResult_t( *TNcclCommDestroy ) ( ncclComm_t comm );
    typedef ncclResult_t( *TNcclAllReduce ) ( const void* sendbuff, void* recvbuff, size_t count,
        ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream );
    typedef ncclResult_t( *TNcclBroadcast ) ( const void* sendbuff, void* recvbuff, size_t count,
        ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream );
    typedef ncclResult_t( *TNcclCommInitRank ) ( ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank );
    typedef ncclResult_t( *TNcclGroupStart ) ();
    typedef ncclResult_t( *TNcclGroupEnd ) ();
    typedef ncclResult_t( *TNcclGetUniqueId ) ( ncclUniqueId* uniqueId );
    typedef const char*( *TNcclGetErrorString ) (  ncclResult_t result );


    TNcclCommInitAll CommInitAll;
    TNcclCommDestroy CommDestroy;
    TNcclAllReduce AllReduce;
    TNcclBroadcast Broadcast;
    TNcclCommInitRank CommInitRank;
    TNcclGroupStart GroupStart;
    TNcclGroupEnd GroupEnd;
    TNcclGetUniqueId GetUniqueId;
    TNcclGetErrorString GetErrorString;
};

} // namespace NeoML

#endif // NEOML_USE_NCCL

