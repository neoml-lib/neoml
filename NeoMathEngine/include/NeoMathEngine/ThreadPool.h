/* Copyright © 2017-2023 ABBYY

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

namespace NeoML {

// The class provides thread pool functionality.
class NEOMATHENGINE_API IThreadPool : public CCrtAllocatedObject {
public:
	// Interface for pool task.
	typedef void( *TFunction )( int threadIndex, void* params );

	IThreadPool() = default;
	virtual ~IThreadPool();
	// Forbidden to copy this class, or any children
	IThreadPool( const IThreadPool& ) = delete;
	IThreadPool& operator=( const IThreadPool& ) = delete;

	// Returns the number of threads in the pool.
	virtual int Size() const = 0;
	// Adds a task with parameters for the given thread.
	virtual bool AddTask( int threadIndex, TFunction function, void* params ) = 0;
	// Waits for all tasks to complete.
	virtual void WaitAllTask() = 0;
};

// Number of available CPU cores in current environment (e.g. inside container)
NEOMATHENGINE_API int GetAvailableCpuCores();

// RAM limit in current environment (e.g. inside container)
// Returns SIZE_MAX if no limit
NEOMATHENGINE_API size_t GetRamLimit();

// Creates a thread pool containing the given number of threads.
// If threadCount is 0 or less then creates a pool with GetAvailableCpuCores() threads
NEOMATHENGINE_API IThreadPool* CreateThreadPool( int threadCount );

//------------------------------------------------------------------------------------------------------------

inline void ExecuteTasks( IThreadPool& threadPool, void* params, IThreadPool::TFunction func )
{
	const int threadCount = threadPool.Size();
	if( threadCount == 1 ) {
		func( 0, params );
		return;
	}
	for( int i = 0; i < threadCount; ++i ) {
		threadPool.AddTask( i, func, params );
	}
	threadPool.WaitAllTask();
}

#define NEOML_NUM_THREADS(_threadPool, _params, _func) {ExecuteTasks(_threadPool, _params, _func);}

#define NEOML_THPOOL_MAX(x, y)    (((x) > (y)) ? (x) : (y))
#define NEOML_THPOOL_MIN(x, y)    (((x) < (y)) ? (x) : (y))

//------------------------------------------------------------------------------------------------------------

// Splits the specified number of tasks [0, fullCount) into intervals for each thread
// Returns the index of the first task and the number of tasks per thread
// IMPORTANT: may return 0 as the number of tasks
// The return value is true if count != 0 (there are some tasks), otherwise false
inline bool GetTaskIndexAndCount( int threadCount, int threadIndex, int fullCount, int align, int& index, int& count )
{
	if( threadCount > 1 ) {
		int countPerThread = ( fullCount + threadCount - 1 ) / threadCount;
		countPerThread = ( align > 1 ) ? ( ( countPerThread + align - 1 ) / align * align ) : countPerThread;

		index = countPerThread * threadIndex;
		count = countPerThread;
		count = NEOML_THPOOL_MIN( count, fullCount - index );
		count = NEOML_THPOOL_MAX( count, 0 );
	} else {
		index = 0;
		count = fullCount;
	}
	return count != 0;
}

inline bool GetTaskIndexAndCount( int threadCount, int threadIndex, int fullCount, int& index, int& count )
{
	return GetTaskIndexAndCount( threadCount, threadIndex, fullCount, /*align*/1, index, count );
}

//------------------------------------------------------------------------------------------------------------

inline int GetTaskGreatestCommonFactorWithAlign( int m, int mAlign, int n )
{
	if( ( m % mAlign ) == 0 ) {
		// If "m" was aligned to start with, preserve the alignment
		m /= mAlign;
	}

	// Euclidean algorithm
	while( true ) {
		const int k = m % n;
		if( k == 0 ) {
			return n;
		}
		m = n;
		n = k;
	}
}

// Similar to GetTaskIndexAndCount, only for a 3D "cube" of tasks to be split among the threads
inline bool GetTaskIndexAndCount3D( int fullCountX, int alignX, int fullCountY, int alignY, int fullCountZ, int alignZ,
	int& indexX, int& countX, int& indexY, int& countY, int& indexZ, int& countZ, int threadCount, int threadIndex )
{
	if( threadCount == 1 ) {
		indexX = 0;
		countX = fullCountX;
		indexY = 0;
		countY = fullCountY;
		indexZ = 0;
		countZ = fullCountZ;
	} else {
		// Calculate the thread block size: = mulX x mulY x mulZ
		// Attempt to divide without a remainder if possible
		int mulX = GetTaskGreatestCommonFactorWithAlign( fullCountX, alignX, threadCount );
		threadCount /= mulX;
		int mulY = GetTaskGreatestCommonFactorWithAlign( fullCountY, alignY, threadCount );
		threadCount /= mulY;
		int mulZ = GetTaskGreatestCommonFactorWithAlign( fullCountZ, alignZ, threadCount );
		threadCount /= mulZ;

		countX = fullCountX / mulX;
		countY = fullCountY / mulY;
		countZ = fullCountZ / mulZ;

		// Find the maximum dimension and divide by it, with remainder if necessary
		int* maxMul = &mulX;
		int* maxCount = &countX;
		const int* align = &alignX;

		if( countY / alignY > *maxCount / *align ) {
			maxMul = &mulY;
			maxCount = &countY;
			align = &alignY;
		}
		if( countZ / alignZ > *maxCount / *align ) {
			maxMul = &mulZ;
			maxCount = &countZ;
		}

		*maxCount = ( *maxCount + threadCount - 1 ) / threadCount;
		*maxMul *= threadCount;

		// Align the block size
		countX = ( countX + alignX - 1 ) / alignX * alignX;
		countY = ( countY + alignY - 1 ) / alignY * alignY;
		countZ = ( countZ + alignZ - 1 ) / alignZ * alignZ;

		// Calculate the coordinate for the given thread in a block
		int threadIndexX = threadIndex % mulX;
		int threadIndexY = threadIndex / mulX;
		int threadIndexZ = threadIndexY / mulY;
		threadIndexY %= mulY;

		// Calculate indeces
		indexX = countX * threadIndexX;
		countX = NEOML_THPOOL_MIN( countX, fullCountX - indexX );
		countX = NEOML_THPOOL_MAX( countX, 0 );

		indexY = countY * threadIndexY;
		countY = NEOML_THPOOL_MIN( countY, fullCountY - indexY );
		countY = NEOML_THPOOL_MAX( countY, 0 );

		indexZ = countZ * threadIndexZ;
		countZ = NEOML_THPOOL_MIN( countZ, fullCountZ - indexZ );
		countZ = NEOML_THPOOL_MAX( countZ, 0 );
	}
	return countX != 0 && countY != 0 && countZ != 0;
}

inline bool GetTaskIndexAndCount3D( int fullCountX, int fullCountY, int fullCountZ,
	int& indexX, int& countX, int& indexY, int& countY, int& indexZ, int& countZ, int threadCount, int threadIndex )
{
	return GetTaskIndexAndCount3D( fullCountX, 1, fullCountY, 1, fullCountZ, 1,
		indexX, countX, indexY, countY, indexZ, countZ, threadCount, threadIndex );
}

inline bool GetTaskIndexAndCount2D( int fullCountX, int alignX, int fullCountY, int alignY,
	int& indexX, int& countX, int& indexY, int& countY, int threadCount, int threadIndex )
{
	int indexZ = 0;
	int countZ = 0;
	return GetTaskIndexAndCount3D( fullCountX, alignX, fullCountY, alignY, 1, 1,
		indexX, countX, indexY, countY, indexZ, countZ, threadCount, threadIndex );
}

inline bool GetTaskIndexAndCount2D( int fullCountX, int fullCountY,
	int& indexX, int& countX, int& indexY, int& countY, int threadCount, int threadIndex )
{
	return GetTaskIndexAndCount2D( fullCountX, 1, fullCountY, 1,
		indexX, countX, indexY, countY, threadCount, threadIndex );
}


} // namespace NeoML
