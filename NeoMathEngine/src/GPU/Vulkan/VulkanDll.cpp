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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_VULKAN

#include <NeoMathEngine/CrtAllocatedObject.h>
#include <VulkanDll.h>
#include <string>
#include <memory>

namespace NeoML {

#define LOAD_VULKAN_FUNC_PROC_NAME(Type, Name, NameStr) if((Name = CDll::GetProcAddress<Type>(NameStr)) == 0) return false
#define LOAD_VULKAN_FUNC_PROC(Name) LOAD_VULKAN_FUNC_PROC_NAME(PFN_##Name, Name, #Name)

#define LOAD_VULKAN_INSTANCE_FUNC_PROC_NAME(Type, Name, NameStr) \
	if((Name = (Type)vkGetInstanceProcAddr(instance, NameStr)) == 0) return false
#define LOAD_VULKAN_INSTANCE_FUNC_PROC(Name) LOAD_VULKAN_INSTANCE_FUNC_PROC_NAME(PFN_##Name, Name, #Name)

#if FINE_PLATFORM(FINE_WINDOWS)
	static const char* VulkanDllName = "vulkan-1.dll";
#elif FINE_PLATFORM(FINE_ANDROID) || FINE_PLATFORM(FINE_LINUX)
	static const char* VulkanDllName = "libvulkan.so";
#else
	#error Platform not supported!
#endif

//------------------------------------------------------------------------------------------------------------

static VkApplicationInfo applicationInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO, 0, "NeoMachineLearning",
	VK_MAKE_VERSION(1, 0, 0), "NeoMathEngine", VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_0 };

static VkInstanceCreateInfo instanceCreateInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0, 0, &applicationInfo,
	0, 0, 0, 0 };

//------------------------------------------------------------------------------------------------------------

typedef basic_string<char, char_traits<char>, CrtAllocator<char> > fstring;

// Gets the vulkan device type
static TVulkanDeviceType defineDeviceType( const VkPhysicalDeviceProperties& props )
{
	fstring name( (const char*)props.deviceName );
	std::transform( name.begin(), name.end(), name.begin(), ::tolower );

	// Mali
	const char* MaliName = "mali";
	size_t pos = name.find( MaliName );
	if( pos != std::string::npos ) {
		size_t curPos = pos + strlen(MaliName);
		while( curPos < name.length() && !isalpha(name[curPos]) && !isdigit(name[curPos]) ) {
			++curPos;
		}
		if( curPos < name.length() && ( isdigit(name[curPos]) || name[curPos] == 't' ) ) {
			// Mali old version
			return VDT_Undefined;
		} else {
			// Mali new version
			return VDT_MaliBifrost;
		}

		return VDT_Regular;
	}

	// Adreno
	const char* AdrenoName = "adreno";
	pos = name.find( AdrenoName );
	if( pos != std::string::npos ) {
		return VDT_Adreno;
	}

	pos = name.find( "geforce" );
	if (pos != std::string::npos) {
		return VDT_Nvidia;
	}

	if(name.find( "intel" ) != std::string::npos ) {
		return VDT_Intel;
	}

	return VDT_Regular;
}

//------------------------------------------------------------------------------------------------------------

CVulkanDll::CVulkanDll() :
	instance( VK_NULL_HANDLE ),
	vkGetInstanceProcAddr( nullptr ),
	vkGetDeviceProcAddr( nullptr ),
	vkCreateInstance( nullptr ),
	vkDestroyInstance( nullptr ),
	vkEnumeratePhysicalDevices( nullptr ),
	vkGetPhysicalDeviceProperties( nullptr ),
	vkGetPhysicalDeviceQueueFamilyProperties( nullptr ),
	vkGetPhysicalDeviceMemoryProperties( nullptr ),
	vkCreateDevice( nullptr )
{
}

CVulkanDll::~CVulkanDll()
{
	Free();
}

bool CVulkanDll::Load()
{
	if( IsLoaded() ) {
		return true;
	}

	if( !CDll::Load( VulkanDllName ) ) {
		return false;
	}

	if( !loadFunctions() ) {
		CDll::Free();
		return false;
	}
	if( !enumDevices() ) {
		vkDestroyInstance( instance, 0 );
		instance = VK_NULL_HANDLE;
		CDll::Free();
		return false;
	}

	return true;
}

const CVulkanDevice* CVulkanDll::CreateDevice( const CVulkanDeviceInfo& info ) const
{
	VkDeviceQueueCreateInfo queueInfo = {};
	queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queueInfo.queueFamilyIndex = info.Family;
	queueInfo.queueCount = 1;
	queueInfo.flags = 0;
	float priority = 1;
	queueInfo.pQueuePriorities = &priority;

	VkPhysicalDeviceFeatures features = {}; // no special features needed

	VkDeviceCreateInfo deviceInfo = {};
	deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceInfo.pQueueCreateInfos = &queueInfo;
	deviceInfo.queueCreateInfoCount = 1;
	deviceInfo.pEnabledFeatures = &features;
	
	VkDevice handle;
	if( vkCreateDevice(info.PhysicalDevice, &deviceInfo, 0, &handle) != VK_SUCCESS ) {
		return nullptr;
	}

	std::unique_ptr<CVulkanDevice> result( new CVulkanDevice( handle, info ) );

#define LOAD_VULKAN_DEVICE_FUNC_PROC_NAME(Type, Name, NameStr) \
	if((result->Name = (Type)vkGetDeviceProcAddr(result->device, NameStr)) == 0) return nullptr
#define LOAD_VULKAN_DEVICE_FUNC_PROC(Name) LOAD_VULKAN_DEVICE_FUNC_PROC_NAME(PFN_##Name, Name, #Name)

#define LOAD_VULKAN_DEVICE_FUNC_PROC_NAME1(Type, Name, NameStr) \
	Type _##Name =  (Type)vkGetDeviceProcAddr(result->device, NameStr); \
	if((_##Name) == nullptr) return nullptr; \
	result->Name = {result->device, _##Name};
#define LOAD_VULKAN_DEVICE_FUNC_PROC1(Name) LOAD_VULKAN_DEVICE_FUNC_PROC_NAME1(PFN_##Name, Name, #Name)

	// Load functions
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkDestroyDevice);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkGetDeviceQueue);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkCreateBuffer);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkCreateImage);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkCreateImageView);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkCreateSampler);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkDestroyBuffer);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkDestroyImage);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkDestroyImageView);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkDestroySampler);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkGetBufferMemoryRequirements);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkGetImageMemoryRequirements);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkAllocateMemory);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkFreeMemory);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkBindBufferMemory);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkBindImageMemory);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkCreateCommandPool);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkDestroyCommandPool);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkCreateComputePipelines);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkDestroyPipeline);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkAllocateCommandBuffers);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkFreeCommandBuffers);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkCreateFence);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkDestroyFence);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkBeginCommandBuffer);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkEndCommandBuffer);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkQueueSubmit);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkWaitForFences);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCmdCopyBuffer);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCmdPipelineBarrier);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkResetFences);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCmdUpdateBuffer);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkMapMemory);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkUnmapMemory);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCmdFillBuffer);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkCreateDescriptorPool);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkDestroyDescriptorPool);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCmdBindPipeline);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCmdBindDescriptorSets);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCmdDispatch);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkAllocateDescriptorSets);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkFreeDescriptorSets);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkCreateDescriptorSetLayout);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkDestroyDescriptorSetLayout);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkUpdateDescriptorSets);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkCreatePipelineLayout);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkDestroyPipelineLayout);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkCreateShaderModule);
	LOAD_VULKAN_DEVICE_FUNC_PROC1(vkDestroyShaderModule);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCmdPushConstants);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkQueueWaitIdle);

	return result.release();
}

void CVulkanDll::Free()
{
	if( IsLoaded() ) {
		devices.clear();
		devices.shrink_to_fit();
		if( vkDestroyInstance != nullptr ) {
			vkDestroyInstance( instance, 0 );
		}
		instance = VK_NULL_HANDLE;
		CDll::Free();
	}
}

// Loads all necessary functions
bool CVulkanDll::loadFunctions()
{
	LOAD_VULKAN_FUNC_PROC(vkGetInstanceProcAddr);
	LOAD_VULKAN_FUNC_PROC(vkGetDeviceProcAddr);
	LOAD_VULKAN_INSTANCE_FUNC_PROC(vkCreateInstance);
	if( vkCreateInstance( &instanceCreateInfo, 0, &instance ) == VK_SUCCESS ) {
		LOAD_VULKAN_INSTANCE_FUNC_PROC(vkDestroyInstance);
		return true;
	}
	return false;
}

// Gets the information about available devices
bool CVulkanDll::enumDevices()
{
	LOAD_VULKAN_INSTANCE_FUNC_PROC(vkEnumeratePhysicalDevices);
	LOAD_VULKAN_INSTANCE_FUNC_PROC(vkGetPhysicalDeviceProperties);
	LOAD_VULKAN_INSTANCE_FUNC_PROC(vkGetPhysicalDeviceQueueFamilyProperties);
	LOAD_VULKAN_INSTANCE_FUNC_PROC(vkGetPhysicalDeviceMemoryProperties);
	LOAD_VULKAN_INSTANCE_FUNC_PROC(vkCreateDevice);

	uint32_t devCount = 0;
	if( vkEnumeratePhysicalDevices( instance, &devCount, 0 ) != VK_SUCCESS ) {
		return false;
	}

	std::vector< VkPhysicalDevice, CrtAllocator<VkPhysicalDevice> > physicalDevices;
	physicalDevices.resize( devCount );
	if( vkEnumeratePhysicalDevices( instance, &devCount, physicalDevices.data() ) != VK_SUCCESS ) {
		return false;
	}

	for( int i = 0; i < static_cast<int>( devCount ); i++ ) {
		VkPhysicalDeviceProperties props = {};
		vkGetPhysicalDeviceProperties( physicalDevices[i], &props );

		// Look for a suitable queue family
		uint32_t familyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties( physicalDevices[i], &familyCount, 0 );
		std::vector< VkQueueFamilyProperties, CrtAllocator<VkQueueFamilyProperties> > families;
		families.resize( familyCount );
		vkGetPhysicalDeviceQueueFamilyProperties( physicalDevices[i], &familyCount, families.data() );

		for( int familyNum = 0; familyNum < static_cast<int>( families.size() ); ++familyNum ) {
			if( families[familyNum].queueCount != 0
				&& ( families[familyNum].queueFlags & VK_QUEUE_COMPUTE_BIT ) != 0
				&& defineDeviceType( props ) != VDT_Undefined )
			{
				TVulkanDeviceType deviceType = defineDeviceType( props );
				if( deviceType != VDT_Undefined ) {
					CVulkanDeviceInfo info;
					info.Type = deviceType;
					info.PhysicalDevice = physicalDevices[i];
					info.DeviceID = static_cast<int>( devices.size() );
					info.Family = familyNum;
					info.AvailableMemory = 0;
					info.Properties = props;
					vkGetPhysicalDeviceMemoryProperties( physicalDevices[i], &info.MemoryProperties );
					for( int h = 0; h < static_cast<int>( info.MemoryProperties.memoryHeapCount ); ++h ) {
						if( ( info.MemoryProperties.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ) != 0 ) {
							info.AvailableMemory += static_cast<size_t>( info.MemoryProperties.memoryHeaps[h].size );
						}
					}

					devices.push_back( info );
					break;
				}
			}
		}
	}
	return true;
}

} // namespace NeoML

#endif // NEOML_USE_VULKAN
