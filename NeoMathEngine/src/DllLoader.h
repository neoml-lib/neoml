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

#pragma once

#include <common.h>
#pragma hdrstop

#include <NeoMathEngine/CrtAllocatedObject.h>

#ifdef NEOML_USE_CUDA
#include <cuda_runtime.h>
#include <CudaMathEngine.h>
#include <CublasDll.h>
#include <CusparseDll.h>
#endif

#ifdef NEOML_USE_NCCL
#include <NcclDll.h>
#endif

#ifdef NEOML_USE_VULKAN
#include <VulkanDll.h>
#endif

#ifdef NEOML_USE_AVX
#include <AvxDll.h>
#endif

namespace NeoML {

// DLL loading control
class CDllLoader : public CCrtAllocatedObject {
public:
#ifdef NEOML_USE_CUDA
	static CCusparseDll* cusparseDll;
	static CCublasDll* cublasDll;
	static int cudaDllLinkCount;
	static constexpr int CUDA_DLL = 0x1;
#else
	static constexpr int CUDA_DLL = 0x0;
#endif

#ifdef NEOML_USE_NCCL
	static CNcclDll* ncclDll;
	static int ncclDllLinkCount;
	static constexpr int NCCL_DLL = 0x8;
#else
	static constexpr int NCCL_DLL = 0x0;
#endif

#ifdef NEOML_USE_VULKAN
	static CVulkanDll* vulkanDll;
	static int vulkanDllLinkCount;
	static constexpr int VULKAN_DLL = 0x2;
#else
	static constexpr int VULKAN_DLL = 0x0;
#endif

#ifdef NEOML_USE_AVX
	static CAvxDll* avxDll;
	static int avxDllLinkCount;
	static constexpr int AVX_DLL = 0x4;
#else
	static constexpr int AVX_DLL = 0x0;
#endif

	explicit CDllLoader( int dll ) : loadedDlls( loadDlls( dll ) ) {}
	~CDllLoader() { freeDlls( loadedDlls ); }

	bool IsLoaded( int dll ) const { return ( loadedDlls & dll ) != 0; }

private:
	int loadedDlls;

	static int loadDlls( int dll );
	static void freeDlls( int dll );
};

} // namespace NeoML
