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
#include <MathEngineAllocator.h>
#include <MathEngineCommon.h>

#include <sstream>

#if FINE_PLATFORM(FINE_LINUX)
#include <fcntl.h>
#include <sys/stat.h>
#include <semaphore.h>
#endif // FINE_PLATFORM(FINE_LINUX)

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
			ReleaseDeviceSlot( Handles[i], DeviceId, i );
		}
	}
}

#if FINE_PLATFORM(FINE_WINDOWS)

static std::string getCudaMutexName(int devNum, int slotNum)
{
	std::stringstream ss;
	ss << "Global\\AbbyyNeoMLCudaDev" << devNum << "_" << slotNum;
	return ss.str();
}

bool IsDeviceSlotFree( int deviceId, int slotIndex )
{
	HANDLE devHandle = ::OpenMutexA( SYNCHRONIZE, FALSE, getCudaMutexName(deviceId, slotIndex).c_str() );
	if( devHandle != 0 ) {
		::CloseHandle(devHandle);
		return false;
	}
	return true;
}

void* CaptureDeviceSlot( int deviceId, int slotIndex, bool reuse )
{
	void* handle = ::CreateMutexA( 0, FALSE, getCudaMutexName(deviceId, slotIndex).c_str() );
	if( handle != nullptr && GetLastError() == ERROR_ALREADY_EXISTS && !reuse ) {
		// Reusing slots is not allowed.
		ReleaseDeviceSlot( handle, deviceId, slotIndex );
		handle = nullptr;
	}
	return handle;
}

void ReleaseDeviceSlot( void* slot, int /*deviceId*/, int /*slotIndex*/ )
{
	::CloseHandle( slot );
}

#elif FINE_PLATFORM(FINE_LINUX)

static const int semInitValue = 255;

static std::string getCudaMutexName(int devNum, int slotNum)
{
	std::stringstream ss;
	ss << "/AbbyyNeoMLCudaDev" << devNum << "_" << slotNum;
	return ss.str();
}

bool IsDeviceSlotFree( int deviceId, int slotIndex )
{
	const std::string name = getCudaMutexName(deviceId, slotIndex);
	sem_t* semaphore = ::sem_open( name.c_str(), O_CREAT, 0666, semInitValue );
	if( semaphore != nullptr ) {
		// Checking its value.
		int value = 0;
		bool isFree = false;
		ASSERT_EXPR( ::sem_getvalue( semaphore, &value ) == 0 );
		isFree = ( value == semInitValue );
		::sem_close( semaphore );
		if( isFree ) {
			// Semaphore was free, removing it from the system.
			::sem_unlink( name.c_str() );
		}
		return isFree;
	}
	return false;
}

void* CaptureDeviceSlot( int deviceId, int slotIndex, bool reuse )
{
	sem_t* semaphore = ::sem_open( getCudaMutexName(deviceId, slotIndex).c_str(),
		O_CREAT, 0666, semInitValue );
	if( semaphore != nullptr ) {
		if( !reuse ) {
			// Checking its value.
			int value = 0;
			bool isFree = false;
			ASSERT_EXPR( ::sem_getvalue( semaphore, &value ) == 0 );
			isFree = ( value == semInitValue );
			if( !isFree ) {
				::sem_close( semaphore );
				// Semaphore isn't free. Skipping sem_unlink...
				return nullptr;
			}
		}
		ASSERT_EXPR( ::sem_wait( semaphore ) == 0 );
	}
	return semaphore;
}

void ReleaseDeviceSlot( void* slot, int deviceId, int slotIndex )
{
	sem_t* semaphore = static_cast<sem_t*>( slot );
	ASSERT_EXPR( ::sem_post( semaphore ) == 0 );
	int value = 0;
	ASSERT_EXPR( ::sem_getvalue( semaphore, &value ) == 0 );
	::sem_close( semaphore );
	if( value == semInitValue ) {
		// That was last sem_close. Removing semaphore from the system.
		::sem_unlink( getCudaMutexName(deviceId, slotIndex).c_str() );
	}
}

#else
#error "Platform is not supported!"
#endif

} // namespace NeoML

#endif // NEOML_USE_CUDA
