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

#include <common.h>
#pragma hdrstop

#include <GradientBoostThreadTask.h>
#include <NeoMathEngine/ThreadPool.h>

namespace NeoML {

void IGradientBoostThreadTask::ParallelRun()
{
	if( ParallelizeSize() < MultiThreadMinTasksCount ) {
		// Run in a single thread
		Run( /*threadIndex*/0, /*index*/0, ParallelizeSize() );
		return;
	}
	// Run in parallel
	NEOML_NUM_THREADS( ThreadPool, this, []( int threadIndex, void* ptr ) {
		( ( IGradientBoostThreadTask* )ptr )->RunSplittedByThreads( threadIndex );
	} );
}

void IGradientBoostThreadTask::RunSplittedByThreads( int threadIndex )
{
	int index = 0;
	int count = 0;
	// 1 dimensional split
	if( GetTaskIndexAndCount( ThreadPool.Size(), threadIndex, ParallelizeSize(), index, count ) ) {
		Run( threadIndex, index, count );
	}
}

} // namespace NeoML
