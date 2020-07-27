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
#include <MathEngineDll.h>
#include <MathEngineAllocator.h>

namespace NeoML {

// The macro checks if a vulkanAPI function call was successful
#define vkSucceded( functionCall ) { VkResult temp = functionCall; temp; assert( temp == VK_SUCCESS ); }

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
struct CVulkanDevice {
	virtual ~CVulkanDevice() {}

	VkDevice Handle;
	int Family;
	bool IsImageBased;
	TVulkanDeviceType Type;
	VkPhysicalDeviceMemoryProperties MemoryProperties;
	VkPhysicalDeviceProperties Properties;

	// The functions loaded for this device
	PFN_vkDestroyDevice vkDestroyDevice;
	PFN_vkGetDeviceQueue vkGetDeviceQueue;
	PFN_vkCreateBuffer vkCreateBuffer;
	PFN_vkCreateImage vkCreateImage;
	PFN_vkCreateImageView vkCreateImageView;
	PFN_vkCreateSampler vkCreateSampler;
	PFN_vkDestroyBuffer vkDestroyBuffer;
	PFN_vkDestroyImage vkDestroyImage;
	PFN_vkDestroyImageView vkDestroyImageView;
	PFN_vkDestroySampler vkDestroySampler;
	PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements;
	PFN_vkGetImageMemoryRequirements vkGetImageMemoryRequirements;
	PFN_vkAllocateMemory vkAllocateMemory;
	PFN_vkFreeMemory vkFreeMemory;
	PFN_vkBindBufferMemory vkBindBufferMemory;
	PFN_vkBindImageMemory vkBindImageMemory;
	PFN_vkCreateCommandPool vkCreateCommandPool;
	PFN_vkDestroyCommandPool vkDestroyCommandPool;
	PFN_vkCreateComputePipelines vkCreateComputePipelines;
	PFN_vkDestroyPipeline vkDestroyPipeline;
	PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers;
	PFN_vkFreeCommandBuffers vkFreeCommandBuffers;
	PFN_vkCreateFence vkCreateFence;
	PFN_vkDestroyFence vkDestroyFence;
	PFN_vkBeginCommandBuffer vkBeginCommandBuffer;
	PFN_vkEndCommandBuffer vkEndCommandBuffer;
	PFN_vkQueueSubmit vkQueueSubmit;
	PFN_vkWaitForFences vkWaitForFences;
	PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier;
	PFN_vkCmdCopyBuffer vkCmdCopyBuffer;
	PFN_vkResetFences vkResetFences;
	PFN_vkCmdUpdateBuffer vkCmdUpdateBuffer;
	PFN_vkMapMemory vkMapMemory;
	PFN_vkUnmapMemory vkUnmapMemory;
	PFN_vkCmdFillBuffer	vkCmdFillBuffer;
	PFN_vkCreateDescriptorPool vkCreateDescriptorPool;
	PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool;
	PFN_vkCmdBindPipeline vkCmdBindPipeline;
	PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets;
	PFN_vkCmdDispatch vkCmdDispatch;
	PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets;
	PFN_vkFreeDescriptorSets vkFreeDescriptorSets;
	PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout;
	PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout;
	PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets;
	PFN_vkCreatePipelineLayout vkCreatePipelineLayout;
	PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout;
	PFN_vkCreateShaderModule vkCreateShaderModule;
	PFN_vkDestroyShaderModule vkDestroyShaderModule;
	PFN_vkCmdPushConstants vkCmdPushConstants;
	PFN_vkQueueWaitIdle vkQueueWaitIdle;
};

//------------------------------------------------------------------------------------------------------------

// The dynamic link vulkan library
class CVulkanDll : public CDll {
public:
	CVulkanDll();
	~CVulkanDll();

	// Loads the library
	bool Load();

	// Checks if the library has been loaded already
	bool IsLoaded() const { return CDll::IsLoaded(); }

	// Gets the information about available devices
	const std::vector< CVulkanDeviceInfo, CrtAllocator<CVulkanDeviceInfo> >& GetDevices() const { return devices; }

	// Creates a device
	CVulkanDevice* CreateDevice( const CVulkanDeviceInfo& info ) const;

	// Unloads the library
	void Free();

private:
	VkInstance instance; // vulkan instance.
	std::vector< CVulkanDeviceInfo, CrtAllocator<CVulkanDeviceInfo> > devices; // available devices
	// The general functions loaded from dll
	PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
	PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr;
	PFN_vkCreateInstance vkCreateInstance;
	PFN_vkDestroyInstance vkDestroyInstance;
	PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices;
	PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties;
	PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties;
	PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties;
	PFN_vkCreateDevice vkCreateDevice;

	bool loadFunctions();
	bool enumDevices();
};

} // namespace NeoML

#endif // NEOML_USE_VULKAN
