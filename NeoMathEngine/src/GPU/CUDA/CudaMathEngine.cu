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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_CUDA

#include <DllLoader.h>
#include <CudaMathEngine.h>
#include <CudaCommon.h>
#include <CublasFunctions.h>
#include <CusparseFunctions.h>
#include <CudaDevice.h>
#include <CudaAssert.h>
#include <MathEngineCommon.h>
#include <MemoryHandleInternal.h>
#include <MathEngineDeviceStackAllocator.h>
#include <MathEngineHostStackAllocator.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

namespace NeoML {

static __constant__ const float ZeroDev = 0;
static __constant__ const float OneDev = 1;

const float* CCudaConst::Zero;
const float* CCudaConst::One;

const int CudaMemoryAlignment = 4;

//------------------------------------------------------------------------------------------------------------

CCudaMathEngine::CCudaMathEngine( const CCusparse* _cusparse, const CCublas* _cublas, std::unique_ptr<CCudaDevice>& _device ) :
	cusparse( _cusparse ),
	cublas( _cublas ),
	cudaStream( 0 ),
	cublasHandle( 0 ),
	cusparseHandle( 0 )
{
	device.swap( _device );

	// CUDA
	ASSERT_EXPR( device != 0 );
	ASSERT_CUDA( cudaSetDevice( device->DeviceNumber ) );

	// CUDA stream.
	ASSERT_CUDA( cudaStreamCreate( &cudaStream ) );

	// Cublas.
	ASSERT_CUBLAS( cublas->Create( &cublasHandle ) );
	ASSERT_CUBLAS( cublas->SetAtomicsMode( cublasHandle, CUBLAS_ATOMICS_ALLOWED ) );
	ASSERT_CUBLAS( cublas->SetPointerMode( cublasHandle, CUBLAS_POINTER_MODE_DEVICE ) );
	ASSERT_CUBLAS( cublas->SetStream( cublasHandle, cudaStream ) );

	// Cusparse.
	ASSERT_CUSPARSE( cusparse->Create( &cusparseHandle ) );
	ASSERT_CUSPARSE( cusparse->SetStream( cusparseHandle, cudaStream ) );

	// Constants
	ASSERT_CUDA( cudaGetSymbolAddress((void**)&CCudaConst::Zero, ZeroDev) );
	ASSERT_CUDA( cudaGetSymbolAddress((void**)&CCudaConst::One, OneDev) );

	memoryPool = std::unique_ptr<CMemoryPool>( new CMemoryPool( device->MemoryLimit, this, true ) );
	deviceStackRunTime = std::unique_ptr<CDeviceStackAllocator>( new CDeviceStackAllocator( *memoryPool, CudaMemoryAlignment ) );
	hostStackRunTime = std::unique_ptr<CHostStackAllocator>( new CHostStackAllocator( CudaMemoryAlignment ) );

	CDllLoader::Load(CDllLoader::CUDA_DLL);
}

CCudaMathEngine::~CCudaMathEngine()
{
	hostStackRunTime.reset();
	deviceStackRunTime.reset();
	memoryPool.reset();

	cudaStreamDestroy( cudaStream );

	cusparse->Destroy( cusparseHandle );
	cublas->Destroy( cublasHandle );

	CDllLoader::Free(CDllLoader::CUDA_DLL);
}

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

static inline void CudaFixGeom(int& minVal, int maxVal, unsigned int& geom)
{
	if(minVal > maxVal) {
		minVal = maxVal;
	}

	if(minVal > (int)geom) {
		minVal = (int)geom;
	}

	if((int)geom > maxVal) {
		geom = maxVal;
	}
}

// The largest 2^N number smaller than this one (returns 1 for input 1)
static inline int GetMax2ExpLess(int value)
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

static inline void CudaFixMinVals(int& minZ, int& minY, int& minX, int maxThreadCount)
{
	int nextMin = 0;
	while(minX * minY * minZ > maxThreadCount) {
		int candidate = nextMin++ % 3;
		switch(candidate) {
			case 0:
				minZ = GetMax2ExpLess(minZ);
				break;
			case 1:
				minY = GetMax2ExpLess(minY);
				break;
			case 2:
				minX = GetMax2ExpLess(minX);
				break;
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

int CCudaMathEngine::alignXSizeForWarp(int xSize)
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

void CCudaMathEngine::getCudaTaskGrid(int& blockCount, int& threadCount, int taskCount, int combineCount)
{
	ASSERT_EXPR( taskCount > 0 );
	ASSERT_EXPR( combineCount > 0 );
	int runCount = (taskCount + combineCount - 1) / combineCount;
	threadCount = device->ThreadMaxCount;

	if(threadCount > runCount) {
		threadCount = runCount;
	}

	blockCount = (runCount + threadCount - 1) / threadCount;
}

void CCudaMathEngine::getCudaTaskGrid2D(dim3& blockCount, dim3& threadCount,
	int height, int width, int maxThreadCount)
{
	getCudaTaskGrid3DMinZYX(1, 1, 1, blockCount, threadCount, 1, height, width, maxThreadCount);
}

void CCudaMathEngine::getCudaTaskGrid3D(dim3& blockCount, dim3& threadCount,
	int batchSize, int height, int width, int maxThreadCount)
{
	getCudaTaskGrid3DMinZYX(1, 1, 1, blockCount, threadCount, batchSize, height, width, maxThreadCount);
}

void CCudaMathEngine::getCudaTaskGrid2DMinYX(int minY, int minX, dim3& blockCount, dim3& threadCount,
	int height, int width, int maxThreadCount)
{
	getCudaTaskGrid3DMinZYX(1, minY, minX, blockCount, threadCount, 1, height, width, maxThreadCount);
}

void CCudaMathEngine::getCudaTaskGrid3DMinZYX(int minZ, int minY, int minX, dim3& blockCount, dim3& threadCount,
	int batchSize, int height, int width, int _maxThreadCount)
{
	int maxThreadCount = min( device->ThreadMaxCount, static_cast<unsigned int>( _maxThreadCount ) );

	ASSERT_EXPR(maxThreadCount >= 1);
	ASSERT_EXPR(minZ > 0 && minY > 0 && minX > 0);
	ASSERT_EXPR(batchSize > 0 && height > 0 && width > 0);

	dim3 geom( device->ThreadMax3DCountX, device->ThreadMax3DCountY, device->ThreadMax3DCountZ );
	CudaFixGeom(minX, width, geom.x);
	CudaFixGeom(minY, height, geom.y);
	CudaFixGeom(minZ, batchSize, geom.z);

	CudaFixMinVals(minX, minY, minZ, maxThreadCount);

	unsigned int optimalGridSize = INT_MAX;
	threadCount = dim3(1, 1, 1);
	blockCount = dim3(width, height, batchSize);

	dim3 currentGeom;
	unsigned int zLimit = min(geom.z * 2, maxThreadCount + 1);
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

			unsigned int gridSize = xBlockCount * yBlockCount * zBlockCount;
			if(gridSize < optimalGridSize) {
				optimalGridSize = gridSize;
				threadCount = dim3(xBlock, yBlock, zBlock);
				blockCount = dim3(xBlockCount, yBlockCount, zBlockCount);
			}
		}
	}
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
