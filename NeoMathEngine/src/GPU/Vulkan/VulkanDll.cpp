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

namespace NeoML {

#define LOAD_VULKAN_FUNC_PROC_NAME(Type, Name, NameStr) if((Name = (Type)CDll::GetProcAddress(NameStr)) == 0) return false
#define LOAD_VULKAN_FUNC_PROC(Name) LOAD_VULKAN_FUNC_PROC_NAME(PFN_##Name, Name, #Name)

#define LOAD_VULKAN_INSTANCE_FUNC_PROC_NAME(Type, Name, NameStr) \
	if((Name = (Type)vkGetInstanceProcAddr(instance, NameStr)) == 0) return false
#define LOAD_VULKAN_INSTANCE_FUNC_PROC(Name) LOAD_VULKAN_INSTANCE_FUNC_PROC_NAME(PFN_##Name, Name, #Name)

#define LOAD_VULKAN_DEVICE_FUNC_PROC_NAME(Type, Name, NameStr) \
	if((Name = (Type)vkGetDeviceProcAddr(Handle, NameStr)) == 0) return false
#define LOAD_VULKAN_DEVICE_FUNC_PROC(Name) LOAD_VULKAN_DEVICE_FUNC_PROC_NAME(PFN_##Name, Name, #Name)

#if FINE_PLATFORM(FINE_WINDOWS)
	static const char* VulkanDllName = "vulkan-1.dll";
#elif FINE_PLATFORM(FINE_ANDROID)
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

	return VDT_Regular;
}

//------------------------------------------------------------------------------------------------------------

// The implementation of a vulkan device
class CVulkanDeviceImpl: public CVulkanDevice, public CCrtAllocatedObject {
public:
	CVulkanDeviceImpl() { Handle = VK_NULL_HANDLE; }
	virtual ~CVulkanDeviceImpl() { DestroyDevice(); }

	// Tries to create the device
	bool CreateDevice( PFN_vkCreateDevice vkCreateDevice, PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr,
		const CVulkanDeviceInfo& info );

	// Destroys the device
	void DestroyDevice();

private:
	bool loadFunctions( PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr );
};

bool CVulkanDeviceImpl::CreateDevice( PFN_vkCreateDevice vkCreateDevice, PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr,
	const CVulkanDeviceInfo& info )
{
	if( Handle != VK_NULL_HANDLE ) {
		return false;
	}

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
	if( vkCreateDevice( info.PhysicalDevice, &deviceInfo, 0, &Handle ) != VK_SUCCESS ) {
		return false;
	}

	LOAD_VULKAN_DEVICE_FUNC_PROC(vkDestroyDevice);

	if( !loadFunctions( vkGetDeviceProcAddr ) ) {
		vkDestroyDevice( Handle, 0 );
		Handle = VK_NULL_HANDLE;
		return false;
	}

	Family = info.Family;
	IsImageBased = ( info.Type != VDT_MaliBifrost );
	Type = info.Type;
	MemoryProperties = info.MemoryProperties;
	Properties = info.Properties;
	return true;
}

void CVulkanDeviceImpl::DestroyDevice()
{
	if( Handle == VK_NULL_HANDLE ) {
		return;
	}
	vkDestroyDevice( Handle, 0 );
	Handle = VK_NULL_HANDLE;
}

// Loads all necessary functions
bool CVulkanDeviceImpl::loadFunctions( PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr )
{
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkDestroyDevice);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkGetDeviceQueue);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCreateBuffer);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCreateImage);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCreateImageView);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCreateSampler);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkDestroyBuffer);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkDestroyImage);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkDestroyImageView);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkDestroySampler);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkGetBufferMemoryRequirements);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkGetImageMemoryRequirements);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkAllocateMemory);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkFreeMemory);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkBindBufferMemory);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkBindImageMemory);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCreateCommandPool);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkDestroyCommandPool);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCreateComputePipelines);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkDestroyPipeline);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkAllocateCommandBuffers);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkFreeCommandBuffers);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCreateFence);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkDestroyFence);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkBeginCommandBuffer);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkEndCommandBuffer);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkQueueSubmit);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkWaitForFences);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCmdCopyBuffer);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCmdPipelineBarrier);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkResetFences);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCmdUpdateBuffer);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkMapMemory);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkUnmapMemory);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCmdFillBuffer);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCreateDescriptorPool);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkDestroyDescriptorPool);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCmdBindPipeline);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCmdBindDescriptorSets);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCmdDispatch);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkAllocateDescriptorSets);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkFreeDescriptorSets);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCreateDescriptorSetLayout);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkDestroyDescriptorSetLayout);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkUpdateDescriptorSets);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCreatePipelineLayout);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkDestroyPipelineLayout);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCreateShaderModule);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkDestroyShaderModule);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkCmdPushConstants);
	LOAD_VULKAN_DEVICE_FUNC_PROC(vkQueueWaitIdle);
	return true;
}

//------------------------------------------------------------------------------------------------------------

CVulkanDll::CVulkanDll() :
	instance( VK_NULL_HANDLE ),
	vkGetInstanceProcAddr( 0 ),
	vkGetDeviceProcAddr( 0 ),
	vkCreateInstance( 0 ),
	vkDestroyInstance( 0 ),
	vkEnumeratePhysicalDevices( 0 ),
	vkGetPhysicalDeviceProperties( 0 ),
	vkGetPhysicalDeviceQueueFamilyProperties( 0 ),
	vkGetPhysicalDeviceMemoryProperties( 0 ),
	vkCreateDevice( 0 )
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

CVulkanDevice* CVulkanDll::CreateDevice( const CVulkanDeviceInfo& info ) const
{
	CVulkanDeviceImpl* deviceImpl = new CVulkanDeviceImpl();

	if( deviceImpl->CreateDevice( vkCreateDevice, vkGetDeviceProcAddr, info ) ) {
		return deviceImpl;
	}

	delete deviceImpl;
	return 0;
}

void CVulkanDll::Free()
{
	if( IsLoaded() ) {
		devices.clear();
		devices.shrink_to_fit();
		if( vkDestroyInstance != 0 ) {
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
					info.Family = familyNum;
					info.AvailableMemory = 0;
					info.Properties = props;
					vkGetPhysicalDeviceMemoryProperties( physicalDevices[i], &info.MemoryProperties );
					for( int h = 0; h < static_cast<int>( info.MemoryProperties.memoryHeapCount ); h++ ) {
						if( ( info.MemoryProperties.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ) != 0 ) {
							info.AvailableMemory += info.MemoryProperties.memoryHeaps[h].size;
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
