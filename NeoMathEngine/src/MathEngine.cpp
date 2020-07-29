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

#include <NeoMathEngine/NeoMathEngine.h>
#include <MathEngineAllocator.h>
#include <CpuMathEngine.h>

#ifdef NEOML_USE_CUDA
#include <cuda_runtime.h>
#include <CudaMathEngine.h>
#include <CudaDevice.h>
#include <CublasDll.h>
#include <CusparseDll.h>
#include <MathEngineCommon.h>
#endif

#ifdef NEOML_USE_VULKAN
#include <VulkanMathEngine.h>
#include <VulkanDll.h>
#endif

#ifdef NEOML_USE_METAL
#include <MetalMathEngine.h>
#endif

#include <DllLoader.h>

#include <vector>

namespace NeoML {

static IMathEngineExceptionHandler* exceptionHandler = 0;

// Interface destructors
IVectorMathEngine::~IVectorMathEngine() {}
IBlasEngine::~IBlasEngine() {}
IDnnEngine::~IDnnEngine() {}
IMathEngine::~IMathEngine() {}
IMathEngineExceptionHandler::~IMathEngineExceptionHandler() {}
IGpuMathEngineManager::~IGpuMathEngineManager() {}

CTimeConvolutionDesc::~CTimeConvolutionDesc() {}
C3dConvolutionDesc::~C3dConvolutionDesc() {}
CConvolutionDesc::~CConvolutionDesc() {}
CChannelwiseConvolutionDesc::~CChannelwiseConvolutionDesc() {}
CRleConvolutionDesc::~CRleConvolutionDesc() {}
CDropoutDesc::~CDropoutDesc() {}
CGlobalMaxPoolingDesc::~CGlobalMaxPoolingDesc() {}
CMaxPoolingDesc::~CMaxPoolingDesc() {}
CMeanPoolingDesc::~CMeanPoolingDesc() {}
C3dMaxPoolingDesc::~C3dMaxPoolingDesc() {}
C3dMeanPoolingDesc::~C3dMeanPoolingDesc() {}
CGlobalMaxOverTimePoolingDesc::~CGlobalMaxOverTimePoolingDesc() {}
CMaxOverTimePoolingDesc::~CMaxOverTimePoolingDesc() {}

// GPU manager implementation
class CGpuMathEngineManager : public IGpuMathEngineManager {
public:
	CGpuMathEngineManager();

	// IGpuMathEngineManager interface methods
	virtual int GetMathEngineCount() const { return static_cast<int>( info.size() ); }
	virtual void GetMathEngineInfo( int index, CMathEngineInfo& info ) const;
	virtual IMathEngine* CreateMathEngine( int index, size_t memoryLimit ) const;

private:
	CDllLoader loader;
	std::vector< CMathEngineInfo, CrtAllocator<CMathEngineInfo> > info;
};

CGpuMathEngineManager::CGpuMathEngineManager() :
	loader()
{
#ifdef NEOML_USE_VULKAN
	if( loader.IsLoaded( CDllLoader::VULKAN_DLL ) ) {
		LoadVulkanEngineInfo( *CDllLoader::vulkanDll, info );
	}
#endif

#ifdef NEOML_USE_CUDA
	if( loader.IsLoaded( CDllLoader::CUDA_DLL ) ) {
		int deviceCount = 0;
		if( cudaGetDeviceCount( &deviceCount ) == cudaSuccess ) {
			for( int i = 0; i < deviceCount; i++ ) {
				cudaDeviceProp devProp;
				if( cudaGetDeviceProperties( &devProp, i ) == cudaSuccess ) {
					CMathEngineInfo deviceInfo;
					deviceInfo.Type = MET_Cuda;
					deviceInfo.Id = i;
					deviceInfo.AvailableMemory = devProp.totalGlobalMem;
					::memset( deviceInfo.Name, 0, sizeof( deviceInfo.Name ) );
					::strcpy( deviceInfo.Name, devProp.name );
					info.push_back( deviceInfo );
				}
			}
		}
	}
#endif

#ifdef NEOML_USE_METAL
	CMathEngineInfo deviceInfo;
	if( LoadMetalEngineInfo( deviceInfo ) ) {
		info.push_back( deviceInfo );
	}
#endif
}

void CGpuMathEngineManager::GetMathEngineInfo( int index, CMathEngineInfo& result ) const
{
	if( 0 <= index && index < static_cast<int>( info.size() ) ) {
		result = info[index];
	} else {
		result = CMathEngineInfo();
	}
}

#ifdef NEOML_USE_CUDA

struct CCudaDevUsage {
	int DevNum;
	size_t FreeMemory;
};

static CCudaDevice* captureSpecifiedCudaDevice( int deviceNumber, size_t deviceMemoryLimit )
{
	cudaDeviceProp devProp;
	ASSERT_ERROR_CODE( cudaGetDeviceProperties(&devProp, deviceNumber) );

	if( deviceMemoryLimit <= 0 ) {
		deviceMemoryLimit = devProp.totalGlobalMem;
	} else if( deviceMemoryLimit > devProp.totalGlobalMem ) {
		return nullptr;
	}

	size_t slotSize = devProp.totalGlobalMem / CUDA_DEV_SLOT_COUNT;
	int slotCount = static_cast<int>( ( deviceMemoryLimit + slotSize - 1 ) / slotSize );
	void* handle = CaptureDeviceSlots( devProp.pciBusID, slotCount );

	if( handle == nullptr ) {
		return nullptr;
	}

	return new CCudaDevice( deviceNumber, deviceMemoryLimit, handle );
}

// Captures the CUDA device
static CCudaDevice* captureCudaDevice( int deviceNumber, size_t deviceMemoryLimit )
{
	if( deviceNumber >= 0 ) {
		return captureSpecifiedCudaDevice( deviceNumber, deviceMemoryLimit );
	}

	int deviceCount = 0;
	ASSERT_ERROR_CODE( cudaGetDeviceCount( &deviceCount ) );

	// Detect the devices and their processing load
	vector<CCudaDevUsage> devs;
	for( int i = 0; i < deviceCount; ++i ) {
		cudaDeviceProp devProp;
		ASSERT_ERROR_CODE( cudaGetDeviceProperties( &devProp, i ) );

		CCudaDevUsage dev;
		dev.DevNum = i;
		dev.FreeMemory = ( CUDA_DEV_SLOT_COUNT - GetDeviceUsage( devProp.pciBusID ) ) * ( devProp.totalGlobalMem / CUDA_DEV_SLOT_COUNT );
		devs.push_back(dev);
	}
	// Sort the devices in decreasing free memory
	std::sort( devs.begin(), devs.end(), []( const CCudaDevUsage& a, const CCudaDevUsage& b ) { return a.FreeMemory > b.FreeMemory; } );

	for( size_t i = 0; i < devs.size(); ++i ) {
		CCudaDevice* result = captureSpecifiedCudaDevice( devs[i].DevNum, deviceMemoryLimit );
		if( result != nullptr ) {
			return result;
		}
	}

	return nullptr;
}

#endif // NEOML_USE_CUDA

IMathEngine* CGpuMathEngineManager::CreateMathEngine( int index, size_t memoryLimit ) const
{
	auto size = static_cast<int>(info.size());
	if( size == 0 || index >= size ) {
		return nullptr;
	}
	switch(info[index >= 0 ? index : 0].Type) {
#ifdef NEOML_USE_CUDA
	case MET_Cuda:
	{
		std::unique_ptr<CCudaDevice> device( captureCudaDevice( index >= 0 ? info[index].Id : -1, memoryLimit ) );
		if( device == nullptr ) {
			return nullptr;
		}
		return new CCudaMathEngine( CDllLoader::cusparseDll->GetFunctions(), CDllLoader::cublasDll->GetFunctions(), device );
	}
#endif
#ifdef NEOML_USE_VULKAN
	case MET_Vulkan:
		return new CVulkanMathEngine( *CDllLoader::vulkanDll, index >= 0 ? info[index].Id : 0, memoryLimit );
#endif
#ifdef NEOML_USE_METAL
	case MET_Metal:
		return new CMetalMathEngine( memoryLimit );
#endif
	case MET_Undefined:
	default:
	{
		memoryLimit;
		return nullptr;
	}
	}
}

//------------------------------------------------------------------------------------------------------------

void SetMathEngineExceptionHandler( IMathEngineExceptionHandler* newExceptionHandler )
{
	exceptionHandler = newExceptionHandler;
}

IMathEngineExceptionHandler* GetMathEngineExceptionHandler()
{
	return exceptionHandler;
}

IMathEngine* CreateCpuMathEngine( int threadCount, size_t memoryLimit )
{
	return new CCpuMathEngine( threadCount, memoryLimit );
}

IMathEngine* CreateGpuMathEngine( size_t memoryLimit )
{
	CGpuMathEngineManager manager;
	return manager.CreateMathEngine(-1, memoryLimit);
}

IGpuMathEngineManager* CreateGpuMathEngineManager()
{
	return new CGpuMathEngineManager();
}

} // namespace NeoML
