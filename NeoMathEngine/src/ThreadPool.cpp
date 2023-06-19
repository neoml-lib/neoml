/* Copyright © 2017-2022 ABBYY Production LLC

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

#include <NeoMathEngine/ThreadPool.h>
#include <NeoMathEngine/OpenMP.h>

#include <condition_variable>
#include <mutex>
#include <thread>
#include <queue>
#include <iostream>

#if FINE_PLATFORM( FINE_LINUX )
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif // _GNU_SOURCE
#include <fstream>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif // FINE_PLATFORM( FINE_LINUX )

namespace NeoML {

#if FINE_PLATFORM( FINE_LINUX )
// Checks if we're running inside of docker or k8s
static bool isInDocker()
{
	{
		// First method: existence of .dockerenv
		struct stat buffer;
		const int ret = ::stat( "/.dockerenv", &buffer );
		if( ret == 0 ) {
			::printf( "dockerenv detected!\n" );
			return true;
		} else {
			::printf( "stat( \"/.dockerenv\") returned %d\n", ret );
		}
	}

	// Second method: checking the contents of cgroup file
	std::ifstream cgroupFile( "/proc/self/cgroup" );
	if( cgroupFile.good() ) {
		std::string data;
		::printf( "cgroup contents:\n" );
		while( cgroupFile >> data ) {
			::printf( "\t%s\n", data.c_str() );
			if( data.find( "docker" ) != std::string::npos ) {
				::printf( "\"docker\" found\n" );
				return true;
			} else if( data.find( "kubepods" ) != std::string::npos ) {
				::printf( "\"kubepods\" found\n" );
				return true;
			}
		}
	}

	::printf( "Not in docker!\n" );
	return false;
}

// Reads integer from file
// Returns -1 if something goes wrong
static int readIntFromFile( const char* name )
{
	std::ifstream stream( name );
	int result = -1;
	if( stream.good() && ( stream >> result ) ) {
		::printf( "read value '%d' from %s\n", result, name );
		return result;
	}
	::printf( "read from %s failed\n", name );
	return -1;
}

#endif // FINE_PLATFORM( FINE_LINUX )

// Returns number of CPU cores available in 
static int getAvailableCpuCoreNum()
{
#if FINE_PLATFORM( FINE_LINUX )
	if( isInDocker() ) {
		// Case #1: linux Docker with --cpus value set
		// In this case the only way to get number of cores is to read quotas
		// When working under cgroups without quotas cfs_quota_us contains -1
		const int quota = readIntFromFile( "/sys/fs/cgroup/cpu/cpu.cfs_quota_us" );
		const int period = readIntFromFile( "/sys/fs/cgroup/cpu/cpu.cfs_period_us" );
		if( quota > 0 && period > 0 ) {
			// Using ceil because --cpus 0.1 is a valid scenario in docker (0.1 means quota * 10 == period)
			::printf( "quota is %d\n", ( quota + period - 1 ) / period );
			return ( quota + period - 1 ) / period;
		}

		// Case #2: linux Docker with --cpuset-cpus
		cpu_set_t cpuSet;
		CPU_ZERO( &cpuSet );
		const int ret = ::pthread_getaffinity_np( ::pthread_self(), sizeof( cpu_set_t ), &cpuSet );
		if( ret == 0 ) {
			::printf( "CPU_COUNT is %d\n", static_cast<int>( CPU_COUNT( &cpuSet ) ) );
			return static_cast<int>( CPU_COUNT( &cpuSet ) );	
		} else {
			::printf( "pthread_getaffinity_np returned %d\n", ret );
		}
	}
#endif // FINE_PLATFORM( FINE_LINUX )

	::printf( "std::thread::hardware_concurrency() is %d\n", std::thread::hardware_concurrency() );
	// hardware_concurrency may return 0 if the value is not well defined or not computable
	return std::max( static_cast<int>( std::thread::hardware_concurrency() ), 1 );
}

struct CTask {
	IThreadPool::TFunction Function;
	void* Params;
};

struct CThreadParams {
	int Count; // Number of threads in the pool.
	int Index; // Thread index in the pool.
	std::condition_variable ConditionVariable;
	std::mutex Mutex;
	std::queue<CTask> Queue;
	bool Stopped;
};

static void threadEntry( CThreadParams* parameters )
{
	CThreadParams& params = *parameters;
	std::unique_lock<std::mutex> lock(params.Mutex);

	while(!params.Stopped) {
		if( !params.Queue.empty() ) {
			CTask task = params.Queue.front();
			lock.unlock();

			try {
				task.Function(params.Index, task.Params);
			} catch(...) {
				ASSERT_EXPR(false); // Better than nothing
			}

			lock.lock();
			params.Queue.pop();
			params.ConditionVariable.notify_all();
		}

		params.ConditionVariable.wait(lock);
	}
}

//------------------------------------------------------------------------------------------------------------

class CThreadPool : public IThreadPool {
public:
	explicit CThreadPool(int threadCount);
	~CThreadPool() override;

	// IThreadPool:
	int Size() const override { return static_cast<int>(threads.size()); }
	bool AddTask( int threadIndex, TFunction function, void* params ) override;
	void WaitAllTask() override;
	void StopAndWait() override;

private:
	std::vector<std::thread*> threads; // CPointerArray isn't available in neoml.
	std::vector<CThreadParams*> params;

	CThreadPool( const CThreadPool& );
	CThreadPool& operator=( const CThreadPool& );
};

CThreadPool::CThreadPool( int threadCount )
{
	std::cout << "Initial call was with " << threadCount << " threads\n";
	std::cout << "C++ detects " << std::thread::hardware_concurrency() << " threads\n";
	std::cout << "OMP detects " << OmpGetMaxThreadCount() << " threads\n";
	if( threadCount <= 0 ) {
		threadCount = getAvailableCpuCoreNum();
	}
	std::cout << "Creating pool with " << threadCount << " threads\n";
	ASSERT_EXPR( threadCount > 0 );
	for( int i = 0; i < threadCount; i++ ) {
		CThreadParams* param = new CThreadParams();
		param->Count = threadCount;
		param->Index = i;
		param->Stopped = false;
		params.push_back(param);

		std::thread* thread = new std::thread(threadEntry, param);
		threads.push_back(thread);
	}
}

CThreadPool::~CThreadPool()
{
	StopAndWait();
	for( auto t : threads ) {
		delete t;
	}
	for( auto p : params ) {
		delete p;
	}
}

bool CThreadPool::AddTask( int threadIndex, TFunction function, void* functionParams )
{
	assert(0 <= threadIndex && threadIndex < static_cast<int>(params.size()));

	std::unique_lock<std::mutex> lock(params[threadIndex]->Mutex);
	params[threadIndex]->Queue.push({function, functionParams});
	params[threadIndex]->ConditionVariable.notify_all();

	return !params[threadIndex]->Stopped;
}

void CThreadPool::WaitAllTask()
{
	for( size_t i = 0; i < params.size(); i++ ) {
		std::unique_lock<std::mutex> lock(params[i]->Mutex);
		while(!params[i]->Queue.empty()) {
			params[i]->ConditionVariable.wait(lock);
		}
	}
}

void CThreadPool::StopAndWait()
{
	for( size_t i = 0; i < threads.size(); i++ ) {
		{
			std::unique_lock<std::mutex> lock(params[i]->Mutex);
			params[i]->Stopped = true;
			params[i]->ConditionVariable.notify_all();
		}
		threads[i]->join();
	}
}

IThreadPool* CreateThreadPool(int threadCount)
{
	return new CThreadPool(threadCount);
}

} // namespace NeoML
