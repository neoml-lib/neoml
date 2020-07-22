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
	void* Handles[CUDA_DEV_SLOT_COUNT];

	CCudaDevice( int deviceNumber, size_t memoryLimit );
	~CCudaDevice();
};

// Checks if device slot is free.
bool IsDeviceSlotFree( int deviceId, int slotIndex );

// Captures slot and returns it's hadnle.
// Handle must be released after work (ReleaseDeviceSlot).
// Returns nullptr if slot is busy.
void* CaptureDeviceSlot( int deviceId, int slotIndex );

// Releases device slot.
void ReleaseDeviceSlot( void* slot, int deviceId, int slotIndex );

} // namespace NeoML

#endif // NEOML_USE_CUDA
