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
#include <MathEngineCommon.h>
#include <MathEngineAllocator.h>

namespace NeoML {

// The macro checks if a vulkanAPI function call was successful
#define vkSucceded( functionCall ) { VkResult temp = functionCall; temp; ASSERT_EXPR( temp == VK_SUCCESS ); }

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
	int DeviceID;
	int Family;
	size_t AvailableMemory;
	VkPhysicalDevice PhysicalDevice;
	VkPhysicalDeviceMemoryProperties MemoryProperties;
	VkPhysicalDeviceProperties Properties;
};

template <typename T>
struct DeviceFunction;

template <typename R, typename... Args>
struct DeviceFunction<R(VKAPI_PTR*)(VkDevice, Args...)>
{
	using PointerType = R(VKAPI_PTR*)( VkDevice, Args... );

	DeviceFunction(): DeviceFunction( nullptr, nullptr ) {}
	DeviceFunction( VkDevice device_, PointerType ptr_ ) :
		device( device_ ),
		ptr( ptr_ )
	{}

	R operator()( Args... args ) const { return ptr( device, args... ); }

private:
	VkDevice device;
	PointerType ptr;
};


// Vulkan device
struct CVulkanDevice
{
	int Family;
	bool IsImageBased;
	TVulkanDeviceType Type;
	VkPhysicalDeviceMemoryProperties MemoryProperties;
	VkPhysicalDeviceProperties Properties;
	std::size_t AvailableMemory;

	// The functions loaded for this device
	DeviceFunction<PFN_vkGetDeviceQueue> vkGetDeviceQueue;
	DeviceFunction<PFN_vkCreateBuffer> vkCreateBuffer;
	DeviceFunction<PFN_vkCreateImage> vkCreateImage;
	DeviceFunction<PFN_vkCreateImageView> vkCreateImageView;
	DeviceFunction<PFN_vkCreateSampler> vkCreateSampler;
	DeviceFunction<PFN_vkDestroyBuffer> vkDestroyBuffer;
	DeviceFunction<PFN_vkDestroyImage> vkDestroyImage;
	DeviceFunction<PFN_vkDestroyImageView> vkDestroyImageView;
	DeviceFunction<PFN_vkDestroySampler> vkDestroySampler;
	DeviceFunction<PFN_vkGetBufferMemoryRequirements> vkGetBufferMemoryRequirements;
	DeviceFunction<PFN_vkGetImageMemoryRequirements> vkGetImageMemoryRequirements;
	DeviceFunction<PFN_vkAllocateMemory> vkAllocateMemory;
	DeviceFunction<PFN_vkFreeMemory> vkFreeMemory;
	DeviceFunction<PFN_vkBindBufferMemory> vkBindBufferMemory;
	DeviceFunction<PFN_vkBindImageMemory> vkBindImageMemory;
	DeviceFunction<PFN_vkCreateCommandPool> vkCreateCommandPool;
	DeviceFunction<PFN_vkDestroyCommandPool> vkDestroyCommandPool;
	DeviceFunction<PFN_vkCreateComputePipelines> vkCreateComputePipelines;
	DeviceFunction<PFN_vkDestroyPipeline> vkDestroyPipeline;
	DeviceFunction<PFN_vkAllocateCommandBuffers> vkAllocateCommandBuffers;
	DeviceFunction<PFN_vkFreeCommandBuffers> vkFreeCommandBuffers;
	DeviceFunction<PFN_vkResetCommandPool> vkResetCommandPool;
	DeviceFunction<PFN_vkCreateFence> vkCreateFence;
	DeviceFunction<PFN_vkDestroyFence> vkDestroyFence;
	PFN_vkBeginCommandBuffer vkBeginCommandBuffer;
	PFN_vkEndCommandBuffer vkEndCommandBuffer;
	PFN_vkQueueSubmit vkQueueSubmit;
	DeviceFunction<PFN_vkWaitForFences> vkWaitForFences;
	PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier;
	PFN_vkCmdCopyBuffer vkCmdCopyBuffer;
	DeviceFunction<PFN_vkResetFences> vkResetFences;
	PFN_vkCmdUpdateBuffer vkCmdUpdateBuffer;
	DeviceFunction<PFN_vkMapMemory> vkMapMemory;
	DeviceFunction<PFN_vkUnmapMemory> vkUnmapMemory;
	PFN_vkCmdFillBuffer	vkCmdFillBuffer;
	DeviceFunction<PFN_vkCreateDescriptorPool> vkCreateDescriptorPool;
	DeviceFunction<PFN_vkDestroyDescriptorPool> vkDestroyDescriptorPool;
	PFN_vkCmdBindPipeline vkCmdBindPipeline;
	PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets;
	PFN_vkCmdDispatch vkCmdDispatch;
	DeviceFunction<PFN_vkAllocateDescriptorSets> vkAllocateDescriptorSets;
	DeviceFunction<PFN_vkFreeDescriptorSets> vkFreeDescriptorSets;
	DeviceFunction<PFN_vkCreateDescriptorSetLayout> vkCreateDescriptorSetLayout;
	DeviceFunction<PFN_vkDestroyDescriptorSetLayout> vkDestroyDescriptorSetLayout;
	DeviceFunction<PFN_vkUpdateDescriptorSets> vkUpdateDescriptorSets;
	DeviceFunction<PFN_vkCreatePipelineLayout> vkCreatePipelineLayout;
	DeviceFunction<PFN_vkDestroyPipelineLayout> vkDestroyPipelineLayout;
	DeviceFunction<PFN_vkCreateShaderModule> vkCreateShaderModule;
	DeviceFunction<PFN_vkDestroyShaderModule> vkDestroyShaderModule;
	PFN_vkCmdPushConstants vkCmdPushConstants;
	PFN_vkQueueWaitIdle vkQueueWaitIdle;

	friend class CVulkanDll;

	CVulkanDevice( const CVulkanDevice& ) = delete;
	CVulkanDevice& operator=( const CVulkanDevice& ) = delete;

	~CVulkanDevice() noexcept;

	const CVulkanDeviceInfo& Info() const { return info; }

private:
	VkDevice device;
	PFN_vkDestroyDevice vkDestroyDevice;
	const CVulkanDeviceInfo& info;

	CVulkanDevice( VkDevice device_, const CVulkanDeviceInfo& info_ ) :
		Family( info_.Family ),
		IsImageBased( ( info_.Type == VDT_MaliBifrost || info_.Type == VDT_Nvidia || info_.Type == VDT_Intel ) ? 
			false : true ),
		Type( info_.Type ),
		MemoryProperties( info_.MemoryProperties ),
		Properties( info_.Properties ),
		AvailableMemory( info_.AvailableMemory ),
		vkBeginCommandBuffer( nullptr ),
		vkEndCommandBuffer( nullptr ),
		vkQueueSubmit( nullptr ),
		vkCmdPipelineBarrier( nullptr ),
		vkCmdCopyBuffer( nullptr ),
		vkCmdUpdateBuffer( nullptr ),
		vkCmdFillBuffer( nullptr ),
		vkCmdBindPipeline( nullptr ),
		vkCmdBindDescriptorSets( nullptr ),
		vkCmdDispatch( nullptr ),
		vkCmdPushConstants( nullptr ),
		vkQueueWaitIdle( nullptr ),
		device(device_),
		vkDestroyDevice( nullptr ),
		info( info_ )
	{}
};

inline CVulkanDevice::~CVulkanDevice() noexcept
{
	if( device != nullptr ) {
		vkDestroyDevice( device, 0 );
		device = nullptr;
	}
}

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
	const CVulkanDevice* CreateDevice(const CVulkanDeviceInfo& info) const;

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

#ifdef ENABLE_VALIDATION
	PFN_vkEnumerateInstanceLayerProperties vkEnumerateInstanceLayerProperties;
	PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties;
	PFN_vkCreateDebugReportCallbackEXT vkCreateDebugReportCallbackEXT;
	PFN_vkDestroyDebugReportCallbackEXT vkDestroyDebugReportCallbackEXT;

	VkDebugReportCallbackEXT callback;

	bool checkLayersSupport() const;
#endif

	bool loadFunctions();
	bool enumDevices();
};

} // namespace NeoML

#endif // NEOML_USE_VULKAN
