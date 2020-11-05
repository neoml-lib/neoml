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

#include <DllLoader.h>

#include <mutex>

namespace NeoML {

#ifdef NEOML_USE_CUDA
CCusparseDll* CDllLoader::cusparseDll = nullptr;
CCublasDll* CDllLoader::cublasDll = nullptr;
int CDllLoader::cudaDllLinkCount = 0;
#endif

#ifdef NEOML_USE_VULKAN
CVulkanDll* CDllLoader::vulkanDll = nullptr;
int CDllLoader::vulkanDllLinkCount = 0;
#endif

#if FINE_PLATFORM(FINE_WINDOWS) || FINE_PLATFORM(FINE_LINUX) || FINE_PLATFORM(FINE_DARWIN)
CAvxDll* CDllLoader::avxDll = nullptr;
int CDllLoader::avxDllLinkCount = 0;
#endif

static std::mutex mutex;

int CDllLoader::Load( int dll )
{
	int result = 0;
	if( (dll & ALL_DLL) != 0 ) {
		std::lock_guard<std::mutex> lock(mutex);
#ifdef NEOML_USE_VULKAN
		if( (dll & VULKAN_DLL) != 0 ) {
			if( vulkanDll == nullptr ) {
				vulkanDll = new CVulkanDll();
			}

			if( !vulkanDll->Load() ) {
				delete vulkanDll;
				vulkanDll = nullptr;
			} else {
				result |= VULKAN_DLL;
				vulkanDllLinkCount++;
			}
		}
#endif

#ifdef NEOML_USE_CUDA
		if( (dll & CUDA_DLL) != 0 ) {
			if( cusparseDll == nullptr ) {
				cusparseDll = new CCusparseDll();
				cublasDll = new CCublasDll();
			}

			if( !cusparseDll->Load() || !cublasDll->Load() ) {
				cusparseDll->Free();
				delete cusparseDll;
				cusparseDll = nullptr;
				cublasDll->Free();
				delete cublasDll;
				cublasDll = nullptr;
			} else {
				result |= CUDA_DLL;
				cudaDllLinkCount++;
			}
		}
#endif

#if FINE_PLATFORM(FINE_WINDOWS) || FINE_PLATFORM(FINE_LINUX) || FINE_PLATFORM(FINE_DARWIN)
		if( ( dll & AVX_DLL ) != 0 ) {
			if( avxDll == nullptr ) {
				avxDll = new CAvxDll();
			}

			if( !avxDll->Load()) {
				delete avxDll;
				avxDll = nullptr;
			} else {
				result |= AVX_DLL;
				avxDllLinkCount++;
			}
		}
#endif
	}
	return result;
}

void CDllLoader::Free( int dll )
{
	if( (dll & ALL_DLL) != 0 ) {
		std::lock_guard<std::mutex> lock( mutex );
#ifdef NEOML_USE_VULKAN
		if( (dll & VULKAN_DLL) != 0 && vulkanDllLinkCount > 0 ) {
			vulkanDllLinkCount--;
			if( vulkanDllLinkCount <= 0 ) {
				delete vulkanDll;
				vulkanDll = nullptr;
			}
		}
#endif
#ifdef NEOML_USE_CUDA
		if( (dll & CUDA_DLL) != 0 && cudaDllLinkCount > 0 ) {
			cudaDllLinkCount--;
			if( cudaDllLinkCount <= 0 ) {
				delete cusparseDll;
				cusparseDll = nullptr;
				delete cublasDll;
				cublasDll = nullptr;
			}
		}
#endif

#if FINE_PLATFORM(FINE_WINDOWS) || FINE_PLATFORM(FINE_LINUX) || FINE_PLATFORM(FINE_DARWIN)
		if( ( dll & AVX_DLL ) != 0 && avxDllLinkCount > 0 ) {
			avxDllLinkCount--;
			if( avxDllLinkCount <= 0 ) {
				delete avxDll;
				avxDll = nullptr;
			}
		}
#endif
	}
}

} // namespace NeoML
