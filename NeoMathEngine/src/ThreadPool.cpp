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

#include <condition_variable>
#include <mutex>
#include <thread>
#include <queue>

namespace NeoML {

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
