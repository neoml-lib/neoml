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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_VULKAN

#include <vector>

#include <vulkan/vulkan.h>

#include <MathEngineAllocator.h>
#include <VulkanCommon.h>

namespace NeoML {

// Vulkan device type
enum TVulkanDeviceType {
	VDT_Undefined = 0,
	VDT_Regular,		// a regular device, use the default algorithms
						// In particular, a pre-Bifrost Mali and all unknown devices will be detected as Regular
	VDT_Adreno,			// Adreno mobile GPU
	VDT_MaliBifrost,	// Mali mobile GPU with Bifrost+ architecture
	VDT_Nvidia,			// Nvidia discrete device
	VDT_Intel			// Intel integrated device
};

// The information about a vulkan device
struct CVulkanDeviceInfo {
	TVulkanDeviceType Type;
	int Family;
	size_t AvailableMemory;
	VkPhysicalDevice PhysicalDevice;
	VkPhysicalDeviceMemoryProperties MemoryProperties;
	VkPhysicalDeviceProperties Properties;
};

//------------------------------------------------------------------------------------------------------------

// Vulkan device
class CVulkanDevice: public CResource<VkDevice, vkDestroyDevice> {
public:
	CVulkanDevice( const CVulkanDeviceInfo& info );

	bool IsImageBased() const {
		if( info.Type == VDT_MaliBifrost || info.Type == VDT_Nvidia || info.Type == VDT_Intel ) {
			return false;
		}
		return true;
	}

	TVulkanDeviceType Type() const { return info.Type; }

	const VkPhysicalDeviceProperties& Properties() const { return info.Properties; }

	const VkPhysicalDeviceMemoryProperties& MemoryProperties() const { return info.MemoryProperties; }

	int Family() const { return info.Family; }
private:
	const CVulkanDeviceInfo& info;
};

//------------------------------------------------------------------------------------------------------------

// Vulkan instance
class CVulkanInstance: public CResource<VkInstance, vkDestroyInstance> {
public:
	CVulkanInstance();

	// Gets the information about available devices
	const std::vector< CVulkanDeviceInfo, CrtAllocator<CVulkanDeviceInfo> >& GetDevices() const { return devices; }
	
private:
	std::vector<CVulkanDeviceInfo, CrtAllocator<CVulkanDeviceInfo>> devices; // available devices

	bool enumDevices();
};

} // namespace NeoML

#endif // NEOML_USE_VULKAN
