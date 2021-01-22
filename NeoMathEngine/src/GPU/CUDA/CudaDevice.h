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

namespace NeoML {

// CUDA device descriptor
struct CCudaDevice : public CCrtAllocatedObject {
	int DeviceNumber;
	int DeviceId;
	size_t MemoryLimit;
	int SharedMemoryLimit;
	int ThreadMaxCount;
	unsigned int ThreadMax3DCountX;
	unsigned int ThreadMax3DCountY;
	unsigned int ThreadMax3DCountZ;
	int WarpSize;
	void* Handle;

	CCudaDevice() {}
	~CCudaDevice();
	CCudaDevice( const CCudaDevice& ) = delete;
	CCudaDevice& operator=( const CCudaDevice& ) = delete;
};

// Captures specified cuda deivice.
// If deviceIndex is less than 0, tries to get some CUDA device (with focus on the free memory size)
// If memoryLimit is 0, then creates device which consumes whole free space on the device
// Device should be deleted after use
CCudaDevice* CaptureCudaDevice( int deviceIndex, size_t memoryLimit );

} // namespace NeoML

#endif // NEOML_USE_CUDA
