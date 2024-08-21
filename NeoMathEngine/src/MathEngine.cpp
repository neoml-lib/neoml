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

#include <common.h>
#pragma hdrstop

#include <NeoMathEngine/NeoMathEngine.h>
#include <MathEngineAllocator.h>
#include <CpuMathEngine.h>
#include <DllLoader.h>

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

// Interface destructors
IVectorMathEngine::~IVectorMathEngine() = default;
IBlasEngine::~IBlasEngine() = default;
IDnnEngine::~IDnnEngine() = default;
IMathEngine::~IMathEngine() = default;
IMathEngineExceptionHandler::~IMathEngineExceptionHandler() = default;
IGpuMathEngineManager::~IGpuMathEngineManager() = default;
IThreadPool::~IThreadPool() = default;

CTimeConvolutionDesc::~CTimeConvolutionDesc() = default;
C3dConvolutionDesc::~C3dConvolutionDesc() = default;
CConvolutionDesc::~CConvolutionDesc() = default;
CChannelwiseConvolutionDesc::~CChannelwiseConvolutionDesc() = default;
CRleConvolutionDesc::~CRleConvolutionDesc() = default;
CDropoutDesc::~CDropoutDesc() = default;
CGlobalMaxPoolingDesc::~CGlobalMaxPoolingDesc() = default;
CMaxPoolingDesc::~CMaxPoolingDesc() = default;
CMeanPoolingDesc::~CMeanPoolingDesc() = default;
C3dMaxPoolingDesc::~C3dMaxPoolingDesc() = default;
C3dMeanPoolingDesc::~C3dMeanPoolingDesc() = default;
CGlobalMaxOverTimePoolingDesc::~CGlobalMaxOverTimePoolingDesc() = default;
CMaxOverTimePoolingDesc::~CMaxOverTimePoolingDesc() = default;
CLrnDesc::~CLrnDesc() = default;
CLstmDesc::~CLstmDesc() = default;
CRowwiseOperationDesc::~CRowwiseOperationDesc() = default;

//------------------------------------------------------------------------------------------------------------

// GPU manager implementation
class CGpuMathEngineManager : public IGpuMathEngineManager {
public:
	CGpuMathEngineManager();

	// IGpuMathEngineManager interface methods
	int GetMathEngineCount() const override { return static_cast<int>( info.size() ); }
	void GetMathEngineInfo( int index, CMathEngineInfo& info ) const override;
	IMathEngine* CreateMathEngine( int index, size_t memoryLimit, int flags = 0 ) const override;

private:
	CDllLoader loader;
	std::vector< CMathEngineInfo, CrtAllocator<CMathEngineInfo> > info;
};

CGpuMathEngineManager::CGpuMathEngineManager() :
	loader( CDllLoader::CUDA_DLL | CDllLoader::VULKAN_DLL )
{
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
#endif // NEOML_USE_CUDA

#ifdef NEOML_USE_VULKAN
	if (loader.IsLoaded(CDllLoader::VULKAN_DLL)) {
		LoadVulkanEngineInfo(*CDllLoader::vulkanDll, info);
	}
#endif // NEOML_USE_VULKAN

#ifdef NEOML_USE_METAL
	CMathEngineInfo deviceInfo;
	if( LoadMetalEngineInfo( deviceInfo ) ) {
		info.push_back( deviceInfo );
	}
#endif // NEOML_USE_METAL
}

void CGpuMathEngineManager::GetMathEngineInfo( int index, CMathEngineInfo& result ) const
{
	if( 0 <= index && index < static_cast<int>( info.size() ) ) {
		result = info[index];
	} else {
		result = CMathEngineInfo();
	}
}

IMathEngine* CGpuMathEngineManager::CreateMathEngine( int index, size_t memoryLimit, int flags ) const
{
	(void)flags; // Avoiding unused variable warning when NEOML_USE_CUDA is not defined
	auto size = static_cast<int>(info.size());
	if( size == 0 || index >= size ) {
		return nullptr;
	}
	IMathEngine* mathEngine = nullptr;
	switch(info[index >= 0 ? index : 0].Type) {
#ifdef NEOML_USE_CUDA
	case MET_Cuda:
	{
		std::unique_ptr<CCudaDevice> device( CaptureCudaDevice( index >= 0 ? info[index].Id : -1, memoryLimit ) );
		if( device != nullptr ) {
			mathEngine = new CCudaMathEngine( CDllLoader::cusparseDll->GetFunctions(), CDllLoader::cublasDll->GetFunctions(), device, flags );
			break;
		}
		return nullptr;
	}
#endif //NEOML_USE_CUDA
#ifdef NEOML_USE_VULKAN
	case MET_Vulkan:
	{
		const auto& deviceInfo = loader.vulkanDll->GetDevices()[index >= 0 ? info[index].Id : 0];
		std::unique_ptr<const CVulkanDevice> device (loader.vulkanDll->CreateDevice( deviceInfo ) );
		if( device != nullptr ) {
			mathEngine = new CVulkanMathEngine( device, memoryLimit );
			break;
		}
		return nullptr;
	}
#endif //NEOML_USE_VULKAN
#ifdef NEOML_USE_METAL
	case MET_Metal:
	{
		mathEngine = new CMetalMathEngine( memoryLimit );
		break;
	}
#endif //NEOML_USE_METAL
	case MET_Undefined:
	default:
	{
		(void)memoryLimit;
		return nullptr;
	}
	}
	ASSERT_EXPR( mathEngine && mathEngine->IsInitialized() ); // Fails, if no call CMemoryEngineMixin::InitializeMemory in some child ctor
	return mathEngine;
}

//------------------------------------------------------------------------------------------------------------

class CDefaultMathEngineExceptionHandler : public IMathEngineExceptionHandler {
public:
	~CDefaultMathEngineExceptionHandler() override = default;

	void OnAssert( const char* message, const wchar_t*, int, int ) override
	{
		throw std::logic_error( message );
	}

	void OnMemoryError() override
	{
		throw std::bad_alloc();
	}

	static IMathEngineExceptionHandler* GetInstance()
	{
		static CDefaultMathEngineExceptionHandler instance;
		return &instance;
	}

private:
	CDefaultMathEngineExceptionHandler() = default;
};

static IMathEngineExceptionHandler* exceptionHandler = CDefaultMathEngineExceptionHandler::GetInstance();

void SetMathEngineExceptionHandler( IMathEngineExceptionHandler* newExceptionHandler )
{
	exceptionHandler = newExceptionHandler == nullptr ? CDefaultMathEngineExceptionHandler::GetInstance() : newExceptionHandler;
}

IMathEngineExceptionHandler* GetMathEngineExceptionHandler()
{
	return exceptionHandler;
}

//------------------------------------------------------------------------------------------------------------

IMathEngine* CreateCpuMathEngine( size_t memoryLimit )
{
	IMathEngine *mathEngine = new CCpuMathEngine( memoryLimit );
	ASSERT_EXPR( mathEngine && mathEngine->IsInitialized() ); // Fails, if no call CMemoryEngineMixin::InitializeMemory in some child ctor
	return mathEngine;
}

// deprecated
IMathEngine* CreateCpuMathEngine( int /*deprecated*/, size_t memoryLimit )
{
	return CreateCpuMathEngine( memoryLimit );
}

//------------------------------------------------------------------------------------------------------------

IMathEngine* CreateGpuMathEngine( size_t memoryLimit, int flags )
{
	CGpuMathEngineManager manager;
	return manager.CreateMathEngine(-1, memoryLimit, flags);
}

IGpuMathEngineManager* CreateGpuMathEngineManager()
{
	return new CGpuMathEngineManager();
}

//------------------------------------------------------------------------------------------------------------

void CreateDistributedCudaMathEngines( IMathEngine** mathEngines, int devsCount, const int* cudaDevs, size_t memoryLimit )
{
	ASSERT_EXPR( mathEngines != nullptr );
	ASSERT_EXPR( devsCount > 0 );
	ASSERT_EXPR( cudaDevs != nullptr );
#ifdef NEOML_USE_NCCL
	CreateDistributedCudaMathEnginesNccl( mathEngines, devsCount, cudaDevs, memoryLimit );
#else  // !NEOML_USE_NCCL
	( void ) memoryLimit;
	ASSERT_EXPR( false );
#endif // !NEOML_USE_NCCL
}

} // namespace NeoML
