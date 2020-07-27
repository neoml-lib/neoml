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

#ifdef NEOML_USE_CUDA

#include <NeoMathEngine/CrtAllocatedObject.h>
#include <cuda_runtime.h>

namespace NeoML {

// The number of slots in the device memory
// Memory may only be blocked by whole slots
const int CUDA_DEV_SLOT_COUNT = 64;

// CUDA device descriptor
struct CCudaDevice : public CCrtAllocatedObject {
	int DeviceNumber;
	int DeviceId;
	size_t MemoryLimit;
	int SharedMemoryLimit;
	int ThreadMaxCount;
	dim3 ThreadMax3DCount;
	int WarpSize;
	void* Handle;

	CCudaDevice( int deviceNumber, size_t memoryLimit, void* handle );
	~CCudaDevice();

	CCudaDevice( const CCudaDevice& ) = delete;
	CCudaDevice& operator=( const CCudaDevice& ) = delete;
};

// Returns the amount of captured slots on this device.
int GetDeviceUsage( int deviceId );

// Captures slots and returns it's hadnle.
// Handle must be released after work (ReleaseDeviceSlots).
// Returns nullptr if device is too busy.
void* CaptureDeviceSlots( int deviceId, int slotCount );

// Releases slots.
void ReleaseDeviceSlots( void* handle );

} // namespace NeoML

#endif // NEOML_USE_CUDA
