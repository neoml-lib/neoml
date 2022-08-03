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

#pragma once

#include <NeoMathEngine/NeoMathEngineDefs.h>
#include <NeoMathEngine/CrtAllocatedObject.h>
#include <NeoMathEngine/NeoMathEngineException.h>

namespace NeoML {

// The class provides thread pool functionality.
class NEOMATHENGINE_API IThreadPool : public CCrtAllocatedObject {
public:
	// Interface for pool task.
	typedef void(*TFunction)(int threadIndex, void* params);

	virtual ~IThreadPool();

	// Returns the number of threads in the pool.
	virtual int Size() const = 0;

	// Adds a task with parameters for the given thread.
	virtual bool AddTask( int threadIndex, TFunction function, void* params ) = 0;

	// Waits for all tasks to complete.
	virtual void WaitAllTask() = 0;

	// Stops all threads and waits for them to complete.
	virtual void StopAndWait() = 0;
};

// Creates a thread pool containing the given number of threads.
NEOMATHENGINE_API IThreadPool* CreateThreadPool(int threadCount);

//------------------------------------------------------------------------------------------------------------

inline void ExecuteTasks(IThreadPool& threadPool, void* params, IThreadPool::TFunction func)
{
	int threadCount = threadPool.Size();
	if( threadCount == 1 ) {
		func(0, params);
		return;
	}
	for (int i = 0; i < threadCount; i++) {
		threadPool.AddTask(i, func, params);
	}
	threadPool.WaitAllTask();
}

#define NEOML_NUM_THREADS(_threadPool, _params, _func) { ExecuteTasks(_threadPool, _params, _func);}

//------------------------------------------------------------------------------------------------------------

// Splits the specified number of tasks [0, fullCount) into intervals for each thread
// Returns the index of the first task and the number of tasks per thread
// IMPORTANT: may return 0 as the number of tasks
// The return value is true if count != 0 (there are some tasks), otherwise false
inline bool GetTaskIndexAndCount( int threadCount, int threadIndex, int fullCount, int align, int& index, int& count )
{
	if( threadCount > 1 ) {
		int countPerThread = (fullCount + threadCount - 1) / threadCount;
		countPerThread = (countPerThread + align - 1) / align * align;

		index = countPerThread * threadIndex;
		count = countPerThread;
		if( index + count > fullCount ) {
			count = fullCount - index;
			if( count < 0 ) {
				count = 0;
			}
		}
	} else {
		index = 0;
		count = fullCount;
	}

	return count != 0;
}

inline bool GetTaskIndexAndCount( int threadCount, int threadIndex, int fullCount, int& index, int& count )
{
	return GetTaskIndexAndCount(threadCount, threadIndex, fullCount, 1, index, count);
}

} // namespace NeoML
