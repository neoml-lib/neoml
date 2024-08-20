/* Copyright © 2023 ABBYY

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

namespace NeoML {

// Forward declaration
class IThreadPool;

// Task which processes in multiple threads
struct IGradientBoostThreadTask {
	virtual ~IGradientBoostThreadTask() {}

	// Run in a single thread or in parallel, corresponding to `ParallelizeSize()`
	void ParallelRun();
protected:
	IGradientBoostThreadTask( IThreadPool& threadPool ) : ThreadPool( threadPool ) {}

	// Get way of split the task into sub-tasks
	void RunSplittedByThreads( int threadIndex );
	// Run the process in a separate thread
	virtual void Run( int threadIndex, int startIndex, int count ) = 0;
	// The size of parallelization, max number of elements to perform
	virtual int ParallelizeSize() const = 0;

	static constexpr int MultiThreadMinTasksCount = 2;
	IThreadPool& ThreadPool; // Executors
};

} // namespace NeoML
