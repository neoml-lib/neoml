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

#include <CudaDevice.h>

namespace NeoML {

CCudaDevice::CCudaDevice( int deviceNumber, size_t memoryLimit ) :
	DeviceNumber( deviceNumber ),
	DeviceId( 0 ),
	MemoryLimit( 0 ),
	SharedMemoryLimit( 48 * 1024 ),
	ThreadMaxCount( 0 ),
	ThreadMax3DCount( 0, 0, 0 ),
	WarpSize( 0 )
{
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, DeviceNumber);

	DeviceId = devProp.pciBusID;

	if( memoryLimit <= 0 ) {
		MemoryLimit = devProp.totalGlobalMem;
	} else {
		MemoryLimit = min( memoryLimit, devProp.totalGlobalMem );
	}

	ThreadMaxCount = devProp.maxThreadsPerBlock;
	ThreadMax3DCount.x = devProp.maxThreadsDim[0];
	ThreadMax3DCount.y = devProp.maxThreadsDim[1];
	ThreadMax3DCount.z = devProp.maxThreadsDim[2];
	WarpSize = devProp.warpSize;

	for( int i = 0; i < CUDA_DEV_SLOT_COUNT; ++i ) {
		Handles[i] = 0;
	}
}

CCudaDevice::~CCudaDevice()
{
	for( int i = 0; i < CUDA_DEV_SLOT_COUNT; ++i ) {
		if( Handles[i] != 0 ) {
			::CloseHandle( Handles[i] );
		}
	}
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
