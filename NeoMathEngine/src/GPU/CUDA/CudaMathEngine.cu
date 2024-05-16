/* Copyright © 2017-2024 ABBYY

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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_CUDA

#include <CudaMathEngine.h>
#include <CudaCommon.h>
#include <CublasFunctions.h>
#include <CusparseFunctions.h>
#include <CudaDevice.h>
#include <CudaAssert.h>
#include <MathEngineCommon.h>
#include <MemoryHandleInternal.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

namespace NeoML {

static __constant__ const float ZeroDev = 0;
static __constant__ const float OneDev = 1;

const int CudaMemoryAlignment = 4;

//------------------------------------------------------------------------------------------------------------

CCudaMathEngine::CCudaMathEngine( const CCusparse* _cusparse, const CCublas* _cublas,
		std::unique_ptr<CCudaDevice>& _device, int flags ) :
	loader( CDllLoader::CUDA_DLL ),
	cusparse( _cusparse ),
	cublas( _cublas ),
	cublasHandle( 0 ),
	cusparseHandle( 0 )
{
	device.swap( _device );

	// CUDA
	ASSERT_EXPR( device != 0 );
	SetCudaDevice( device->DeviceNumber );

	// Cublas.
	ASSERT_CUBLAS( cublas->Create( &cublasHandle ) );
	cublasMath_t cublasMath = CUBLAS_DEFAULT_MATH;
	if( ( flags & GpuMathEngineCublasUseTensorCoresTF32Flag ) != 0 ) {
		cublasMath = CUBLAS_TF32_TENSOR_OP_MATH;
	} else if( ( flags & GpuMathEngineCublasUseTensorCoresHalfFlag ) != 0 ) {
		cublasMath = CUBLAS_TENSOR_OP_MATH;
	}
	ASSERT_CUBLAS( cublas->SetMathMode( cublasHandle, cublasMath ) );
	ASSERT_CUBLAS( cublas->SetAtomicsMode( cublasHandle, CUBLAS_ATOMICS_ALLOWED ) );
	ASSERT_CUBLAS( cublas->SetPointerMode( cublasHandle, CUBLAS_POINTER_MODE_DEVICE ) );

	// Cusparse.
	ASSERT_CUSPARSE( cusparse->Create( &cusparseHandle ) );

	// Constants
	ASSERT_CUDA( cudaGetSymbolAddress((void**)&cudaConstZero, ZeroDev) );
	ASSERT_CUDA( cudaGetSymbolAddress((void**)&cudaConstOne, OneDev) );

	InitializeMemory( this, device->MemoryLimit, CudaMemoryAlignment, /*reuse*/true, /*hostStack*/true );
}

CCudaMathEngine::~CCudaMathEngine()
{
	HostStackAllocator.reset();
	DeviceStackAllocator.reset();
	MemoryPool.reset();

	cusparse->Destroy( cusparseHandle );
	cublas->Destroy( cublasHandle );
}

//---------------------------------------------------------------------------------------------------------------------

static inline void cudaFixGeom( int& minVal, int maxVal, unsigned int& geom )
{
	if(minVal > maxVal) {
		minVal = maxVal;
	}

	if( minVal > static_cast<int>( geom ) ) {
		minVal = static_cast<int>( geom );
	}

	if( static_cast<int>( geom ) > maxVal ) {
		geom = maxVal;
	}
}

// The largest 2^N number smaller than this one (returns 1 for input 1)
static inline int getMax2ExpLess( int value )
{
	const int startExp = 16;
	int expStep = startExp >> 1;

	int candidate = 1 << startExp;
	while(expStep > 0) {
		if(candidate >= value) {
			candidate >>= expStep;
		} else {
			candidate <<= expStep;
		}
		expStep >>= 1;
	}

	if(candidate >= value) {
		candidate >>= 1;
	}

	return candidate;
}

static inline void cudaFixMinVals(int& minX, int& minY, int& minZ, int maxThreadCount,
	int gridMinX, int gridMinY, int gridMinZ)
{
	int nextMin = 0;
	int lastReduce = 0;
	while(minX * minY * minZ > maxThreadCount || nextMin >= lastReduce + 3) {
		int candidate = nextMin++ % 3;
		switch(candidate) {
			case 0:
			{
				int newMinX = getMax2ExpLess( minX );
				if( newMinX >= gridMinX ) {
					minX = newMinX;
					lastReduce = nextMin;
				}
				break;
			}
			case 1:
			{
				int newMinY = getMax2ExpLess( minY );
				if( newMinY >= gridMinY ) {
					minY = newMinY;
					lastReduce = nextMin;
				}
				break;
			}
			case 2:
			{
				int newMinZ = getMax2ExpLess( minZ );
				if( newMinZ >= gridMinZ ) {
					minZ = newMinZ;
					lastReduce = nextMin;
				}
				break;
			}
		}
	}
}

static inline int cudaGridMinBlockSize( int taskNum, int maxGridSize )
{
	return static_cast<int>( ( static_cast<int64_t>( taskNum ) + maxGridSize - 1 ) / maxGridSize );
}

static inline uint64_t cudaCalculateBlock( int height, int width, int batchSize,
	int minX, int minY, int minZ, int maxThreadCount, const CCudaDevice& device, const dim3& geom,
	dim3& threadCount, dim3& blockCount )
{
	uint64_t optimalGridSize = ULLONG_MAX;

	dim3 currentGeom;
	unsigned int zLimit = min( geom.z * 2, static_cast<unsigned>( maxThreadCount ) + 1 );
	for(currentGeom.z = minZ; currentGeom.z < zLimit; currentGeom.z *= 2) {
		unsigned int zBlock = min(currentGeom.z, geom.z);
		unsigned int zBlockCount = (batchSize + zBlock - 1) / zBlock;

		unsigned int xyMaxThreadCount = maxThreadCount / currentGeom.z;
		unsigned int yLimit = min(geom.y * 2, xyMaxThreadCount + 1);

		for(currentGeom.y = minY; currentGeom.y < yLimit; currentGeom.y *= 2) {

			currentGeom.x = xyMaxThreadCount / currentGeom.y;
			if((int)currentGeom.x < minX) {
				continue;
			}

			unsigned int yBlock = min(currentGeom.y, geom.y);
			unsigned int yBlockCount = (height + yBlock - 1) / yBlock;

			unsigned int xBlock = min(currentGeom.x, geom.x);
			unsigned int xBlockCount = (width + xBlock - 1) / xBlock;

			uint64_t gridSize = static_cast<uint64_t>( xBlockCount ) * yBlockCount * zBlockCount;
			if(gridSize < optimalGridSize) {
				optimalGridSize = gridSize;
				threadCount = dim3(xBlock, yBlock, zBlock);
				blockCount = dim3(xBlockCount, yBlockCount, zBlockCount);
			}
		}
	}

	return optimalGridSize;
}

//---------------------------------------------------------------------------------------------------------------------

int CCudaMathEngine::alignXSizeForWarp( int xSize ) const
{
	// Align the size so it is either large than warp or smaller or equal and could be presented as 2^N 
	// Required for reduction with warps
	int candidate = device->WarpSize;
	if( xSize >= candidate ) {
		return ( ( xSize + candidate - 1 ) / candidate ) * candidate;
	}

	int next = candidate;
	do {
		candidate = next;
		next = next >> 1;
	} while(xSize <= next);

	return candidate;
}

int CCudaMathEngine::getCudaTempMatrixMaxHeight( int matrixHeight, int matrixWidth ) const
{
	const int maxTempMatrixSizeConst = 256 * 1024 * 1024;
	const int maxPossibleMatrixHeight = min( maxTempMatrixSizeConst,
		static_cast<int>( max( static_cast<size_t>( 1 ), GetFreeMemorySize() / ( 2 * sizeof( float ) * matrixWidth ) ) ) );
	return min( matrixHeight, maxPossibleMatrixHeight );
}

void CCudaMathEngine::getCudaTaskGrid( int& blockCount, int& threadCount, int taskCount, int combineCount ) const
{
	ASSERT_EXPR( taskCount > 0 );
	ASSERT_EXPR( combineCount > 0 );
	const int runCount = (taskCount + combineCount - 1) / combineCount;
	threadCount = device->ThreadMaxCount;

	if(threadCount > runCount) {
		threadCount = runCount;
	}

	blockCount = (runCount + threadCount - 1) / threadCount;
}

void CCudaMathEngine::getCudaTaskGrid2D(dim3& blockCount, dim3& threadCount,
	int height, int width, int maxThreadCount) const
{
	getCudaTaskGrid3DMinZYX(1, 1, 1, blockCount, threadCount, 1, height, width, maxThreadCount);
}

void CCudaMathEngine::getCudaTaskGrid3D(dim3& blockCount, dim3& threadCount,
	int batchSize, int height, int width, int maxThreadCount) const
{
	getCudaTaskGrid3DMinZYX(1, 1, 1, blockCount, threadCount, batchSize, height, width, maxThreadCount);
}

void CCudaMathEngine::getCudaTaskGrid2DMinYX(int minY, int minX, dim3& blockCount, dim3& threadCount,
	int height, int width, int maxThreadCount) const
{
	getCudaTaskGrid3DMinZYX(1, minY, minX, blockCount, threadCount, 1, height, width, maxThreadCount);
}

void CCudaMathEngine::getCudaTaskGrid3DMinZYX(int minZ, int minY, int minX, dim3& blockCount, dim3& threadCount,
	int batchSize, int height, int width, int _maxThreadCount) const
{
	const int maxThreadCount = min( device->ThreadMaxCount, _maxThreadCount );

	ASSERT_EXPR(maxThreadCount >= 1);
	ASSERT_EXPR(minZ > 0 && minY > 0 && minX > 0);
	ASSERT_EXPR(batchSize > 0 && height > 0 && width > 0);

	dim3 geom( device->ThreadMax3DCountX, device->ThreadMax3DCountY, device->ThreadMax3DCountZ );
	cudaFixGeom( minX, width, geom.x );
	cudaFixGeom( minY, height, geom.y );
	cudaFixGeom( minZ, batchSize, geom.z );

	const int gridBlockMinX = cudaGridMinBlockSize( width, device->MaxGridSizeX );
	const int gridBlockMinY = cudaGridMinBlockSize( height, device->MaxGridSizeY );
	const int gridBlockMinZ = cudaGridMinBlockSize( batchSize, device->MaxGridSizeZ );
	ASSERT_EXPR( static_cast<uint64_t>( gridBlockMinX ) * gridBlockMinY * gridBlockMinZ
		<= static_cast<uint64_t>( maxThreadCount ) );

	// We cannot violate grid limits (otherwise device won't be able to execute the task)
	minX = max( gridBlockMinX, minX );
	minY = max( gridBlockMinY, minY );
	minZ = max( gridBlockMinZ, minZ );

	cudaFixMinVals( minX, minY, minZ, maxThreadCount, gridBlockMinX, gridBlockMinY, gridBlockMinZ );

	threadCount = dim3(1, 1, 1);
	blockCount = dim3(width, height, batchSize);

	uint64_t optimalBlockSize = ULLONG_MAX;
	if( static_cast<uint64_t>( minX ) * minY * minZ <= static_cast<uint64_t>( maxThreadCount ) ) {
		optimalBlockSize = cudaCalculateBlock( height, width, batchSize, minX, minY, minZ, maxThreadCount, *device,
			geom, threadCount, blockCount );
	}
	if( optimalBlockSize == ULLONG_MAX ) {
		// Ignore min* and try to find the block which fits the grid (gridBlockMin*)
		optimalBlockSize = cudaCalculateBlock( height, width, batchSize, gridBlockMinX, gridBlockMinY, gridBlockMinZ,
			maxThreadCount, *device, geom, threadCount, blockCount );
	}

	ASSERT_EXPR(optimalBlockSize != ULLONG_MAX);
	ASSERT_EXPR(blockCount.x <= device->MaxGridSizeX);
	ASSERT_EXPR(blockCount.y <= device->MaxGridSizeY);
	ASSERT_EXPR(blockCount.z <= device->MaxGridSizeZ);
	ASSERT_EXPR(threadCount.x <= device->ThreadMax3DCountX);
	ASSERT_EXPR(threadCount.y <= device->ThreadMax3DCountY);
	ASSERT_EXPR(threadCount.z <= device->ThreadMax3DCountZ);
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
