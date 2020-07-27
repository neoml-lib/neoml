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

#if FINE_PLATFORM(FINE_LINUX)
#include <unistd.h>
#include <sys/types.h>
#include <sys/file.h>
#include <signal.h>
#include <chrono>
#include <mutex>
#include <thread>
#endif // FINE_PLATFORM(FINE_LINUX)

namespace NeoML {

CCudaDevice::CCudaDevice( int deviceNumber, size_t memoryLimit, void* handle ) :
	DeviceNumber( deviceNumber ),
	DeviceId( 0 ),
	MemoryLimit( memoryLimit ),
	SharedMemoryLimit( 48 * 1024 ),
	ThreadMaxCount( 0 ),
	ThreadMax3DCount( 0, 0, 0 ),
	WarpSize( 0 ),
	Handle( handle )
{
	ASSERT_EXPR( Handle != nullptr );

	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, DeviceNumber);

	DeviceId = devProp.pciBusID;

	ThreadMaxCount = devProp.maxThreadsPerBlock;
	ThreadMax3DCount.x = devProp.maxThreadsDim[0];
	ThreadMax3DCount.y = devProp.maxThreadsDim[1];
	ThreadMax3DCount.z = devProp.maxThreadsDim[2];
	WarpSize = devProp.warpSize;
}

CCudaDevice::~CCudaDevice()
{
	ReleaseDeviceSlots( Handle );
}

#if FINE_PLATFORM(FINE_WINDOWS)

static std::string getCudaMutexName(int devNum, int slotNum)
{
	return "Global\\AbbyyNeoMLCudaDev" + std::to_string( devNum ) + "_" + std::to_string( slotNum );
}

int GetDeviceUsage( int deviceId )
{
	int result = 0;
	for( int slotIndex = 0; slotIndex < CUDA_DEV_SLOT_COUNT; ++slotIndex ) {
		HANDLE devHandle = ::OpenMutexA( SYNCHRONIZE, FALSE, getCudaMutexName(deviceId, slotIndex).c_str() );
		if( devHandle != 0 ) {
			::CloseHandle(devHandle);
			result += 1;
		}
	}
	return result;
}

typedef std::vector<void*> CSlotsHandle;

void* CaptureDeviceSlots( int deviceId, int slotCount )
{
	std::unique_ptr<CSlotsHandle> result( new CSlotsHandle() );
	result->reserve( slotCount );

	for( int slotIndex = 0; slotIndex < CUDA_DEV_SLOT_COUNT; ++slotIndex ) {
		void* handle = ::CreateMutexA( 0, FALSE, getCudaMutexName(deviceId, slotIndex).c_str() );
		if( handle != nullptr && GetLastError() == ERROR_ALREADY_EXISTS ) {
			// Reusing slots is not allowed.
			::CloseHandle( handle );
		}
		if( handle != nullptr ) {
			result->push_back( handle );
			if( static_cast<int>( result->size() ) == slotCount ) {
				return result.reelase();
			}
		}
	}

	ASSERT_EXPT( static_cast<int>( result->size() ) < slotCount );
	for( size_t handleIndex = 0; handleIndex < result->size(); ++handleIndex ) {
		::CloseHandle( ( *handles )[handleIndex] );
	}
	return nullptr;
}

void ReleaseDeviceSlots( void* handle )
{
	CSlotsHandle* handles = static_cast<CSlotsHandle*>( handle );
	for( size_t handleIndex = 0; handleIndex < static_cast<inresult->size(); ++handleIndex ) {
		::CloseHandle( ( *handles )[handleIndex] );
	}
	delete handles;
}

#elif FINE_PLATFORM(FINE_LINUX)

//---------------------------------------------------------------------------------------------------------------------
// Static part

static std::timed_mutex mutex;

// Returns process start time.
static unsigned long long getProcessStartTime( int pid )
{
	const std::string fileName = "/proc/" + std::to_string( pid ) + "/stat";
	FILE* fp = ::fopen( fileName.data(), "r" );
	if( fp == nullptr ) {
		return 0;
	}

	// In order to get start time we have to parse all the preceding fields in /proc/<pid>/stat file.
	int currPid = 0;
	char exec[256];
	char state;
	int parentPid = 0;
	int processGroupId = 0;
	int sessionId = 0;
	int terminal = 0;
	int terminalProcessGroupId = 0;
	unsigned int flags = 0;
	unsigned long int minorFaults = 0;
	unsigned long int childrenMinorFaults = 0;
	unsigned long int majorFaults = 0;
	unsigned long int childrenMajorFaults = 0;
	unsigned long int userTime = 0;
	unsigned long int kernelTime = 0;
	long int childrenUserTime = 0;
	long int childrenKernelTime = 0;
	long int priority = 0;
	long int nice = 0;
	long int numThreads = 0;
	long int itRealValue = 0;
	unsigned long long startTime = 0;

	int parsed = ::fscanf( fp, "%d %s %c %d %d %d %d %d %u %lu %lu %lu %lu %lu %lu %ld %ld %ld %ld %ld %ld %llu",
		&currPid, exec, &state, &parentPid, &processGroupId, &sessionId, &terminal, &terminalProcessGroupId,
		&flags, &minorFaults, &childrenMinorFaults, &majorFaults, &childrenMajorFaults, &userTime, &kernelTime,
		&childrenUserTime, &childrenKernelTime, &priority, &nice, &numThreads, &itRealValue, &startTime );
	::fclose( fp );

	return parsed == 22 ? startTime : 0;
}

static std::string getCudaDeviceFileName( int devNum )
{
	return "/var/lock/AbbyyNeoMLCudaDev" + std::to_string( devNum );
}

//---------------------------------------------------------------------------------------------------------------------
// File, containing info about slot acquisition for a single device.
class CDeviceFile {
public:
	explicit CDeviceFile( int deviceNum );
	~CDeviceFile();

	CDeviceFile( const CDeviceFile& ) = delete;
	CDeviceFile& operator=( const CDeviceFile& ) = delete;

	// Opens a file and acquires all necessary locks.
	// Returns true if succeeded.
	bool Open();

	// Checks if slot is free.
	// Returns true if slot is free.
	bool IsSlotFree( int slotIndex );

	// Captures slot if it's free.
	// Returns true if succeeded.
	bool CaptureSlot( int slotIndex );

	// Releases previously captured slot.
	void ReleaseSlot( int slotIndex );

private:
	static const int slotEntrySize = 12; // pid (%d) + start time (%ull)
	int device; // device Id
	int fd; // file descriptor (-1 if closed)
};

//---------------------------------------------------------------------------------------------------------------------
// CDeviceFile's implementation

CDeviceFile::CDeviceFile( int deviceNum ) :
	device( deviceNum ),
	fd( -1 )
{
}

CDeviceFile::~CDeviceFile()
{
	if( fd != -1 ) {
		::flock( fd, LOCK_UN );
		mutex.unlock();
		::close( fd );
	}
}

bool CDeviceFile::Open()
{
	ASSERT_EXPR( fd == -1 );
	int localFd = ::open( getCudaDeviceFileName( device ).data(), O_CREAT | O_RDWR, 0666 );
	if( localFd == -1 ) {
		return false;
	}

	const int maxTimeoutMs = 5000;
	auto cnow = std::chrono::steady_clock::now().time_since_epoch();
	auto lockStart = std::chrono::duration_cast<std::chrono::milliseconds>(cnow).count();

	// First lock: mutex for sync between threads.
	if( !mutex.try_lock_for( std::chrono::milliseconds( maxTimeoutMs ) ) ) {
		::close( localFd );
		return false;
	}

	// Second lock: flock for sync between processes.
	while( ::flock( localFd, LOCK_EX | LOCK_NB ) == -1 ) {
		cnow = std::chrono::steady_clock::now().time_since_epoch();
		auto now = std::chrono::duration_cast<std::chrono::milliseconds>(cnow).count();
		if( now - lockStart > maxTimeoutMs ) {
			// Failed to acquire lock for a device file in time.
			mutex.unlock();
			::close( localFd );
			return false;
		}
		std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
	}

	int64_t fileLength = static_cast<int64_t>( ::lseek( localFd, 0, SEEK_END ) );
	if( fileLength != CUDA_DEV_SLOT_COUNT * slotEntrySize ) {
		std::vector<char> buff( CUDA_DEV_SLOT_COUNT * slotEntrySize, 0 );
		ASSERT_EXPR( ::lseek( localFd, 0, SEEK_SET ) == 0 );
		::write( localFd, buff.data(), buff.size() );
		// If original file is bigger than needed.
		if( fileLength > static_cast<int64_t>( buff.size() ) ) {
			ASSERT_EXPR( ::lseek( localFd, 0, SEEK_SET ) == 0 );
			::ftruncate( localFd, buff.size() );
		}
	}

	fd = localFd;
	return true;
}

bool CDeviceFile::IsSlotFree( int slotIndex )
{
	ASSERT_EXPR( fd != -1 );
	::lseek( fd, slotIndex * slotEntrySize, SEEK_SET );

	// Lets check slot content.
	pid_t pid;
	::read( fd, &pid, sizeof( pid ) );
	bool result = false;
	if( pid != 0 ) {
		// Slot is empty if such process doesn't exist.
		result = ::kill( pid, 0 ) == -1;
		if( !result ) {
			// Process pid is still alive.
			// Let's check its start time.
			unsigned long long actualStartTime = getProcessStartTime( pid );
			unsigned long long storedStartTime = 0;
			::read( fd, &storedStartTime, sizeof( storedStartTime ) );
			// The mismatch between actual and stored start times means
			// that pid was reused and the original process has already finished.
			result = ( actualStartTime != storedStartTime );
		}
	} else {
		// Slot is filled with zeroes. That means its free.
		result = true;
	}

	return result;
}

bool CDeviceFile::CaptureSlot( int slotIndex )
{
	ASSERT_EXPR( fd != -1 );
	if( !IsSlotFree( slotIndex ) ) {
		return false;
	}

	::lseek( fd, slotIndex * slotEntrySize, SEEK_SET );
	// Write current pid and process start time.
	pid_t pid = ::getpid();
	::write( fd, &pid, sizeof( pid ) );
	int64_t startTime = getProcessStartTime( pid );
	::write( fd, &startTime, sizeof( startTime ) );
	return true;
}

void CDeviceFile::ReleaseSlot( int slotIndex )
{
	ASSERT_EXPR( fd != -1 );
	// Write zeroes over current slot entry.
	::lseek( fd, slotIndex * slotEntrySize, SEEK_SET );
	std::vector<char> buff( slotEntrySize, 0 );
	::write( fd, buff.data(), buff.size() );
}

//---------------------------------------------------------------------------------------------------------------------
// Functions from CudaDevice.h

int GetDeviceUsage( int deviceId )
{
	CDeviceFile file( deviceId );
	if( !file.Open() ) {
		return CUDA_DEV_SLOT_COUNT;
	}

	int result = 0;
	for( int slotIndex = 0; slotIndex < CUDA_DEV_SLOT_COUNT; ++slotIndex ) {
		if( !file.IsSlotFree( slotIndex ) ) {
			++result;
		}
	}
	return result;
}

typedef std::vector<int64_t> CSlotsHandle;

void* CaptureDeviceSlots( int deviceId, int slotCount )
{
	CDeviceFile file( deviceId );
	if( !file.Open() ) {
		return nullptr;
	}
	
	// Check if we still have required slots.
	int freeSlots = 0;
	for( int slotIndex = 0; slotIndex < CUDA_DEV_SLOT_COUNT; ++slotIndex ) {
		if( file.IsSlotFree( slotIndex ) ) {
			++freeSlots;
		}
	}
	if( freeSlots < slotCount ) {
		return nullptr;
	}

	// Capturing slots
	std::unique_ptr<CSlotsHandle> slots( new CSlotsHandle() );
	slots->reserve( slotCount );
	for( int slotIndex = 0; slotIndex < CUDA_DEV_SLOT_COUNT; ++slotIndex ) {
		if( file.CaptureSlot( slotIndex ) ) {
			slots->push_back( deviceId * CUDA_DEV_SLOT_COUNT + slotIndex );
			if( static_cast<int>( slots->size() ) == slotCount ) {
				break;
			}
		}
	}

	ASSERT_EXPR( static_cast<int>( slots->size() ) == slotCount );
	return slots.release();
}

void ReleaseDeviceSlots( void* handle )
{
	CSlotsHandle* handles = static_cast<CSlotsHandle*>( handle );\

	if( !handles->empty() ) {
		const int device = ( *handles )[0] / CUDA_DEV_SLOT_COUNT;
		CDeviceFile file( device );
		if( file.Open() ) {
			for( size_t handleIndex = 0; handleIndex < handles->size(); ++handleIndex ) {
				const int slotIndex = ( *handles )[handleIndex] % CUDA_DEV_SLOT_COUNT;
				file.ReleaseSlot( slotIndex );
			}
		}
	}

	delete handles;
}

#else
#error "Platform is not supported!"
#endif

} // namespace NeoML

#endif // NEOML_USE_CUDA
