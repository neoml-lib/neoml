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
#include <VulkanDevice.h>

#include <string>
#include <stdexcept>

namespace NeoML {

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

static inline VkDevice makeVulkanDevice( const CVulkanDeviceInfo& info )
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
	
	VkDevice device;
	if( vkCreateDevice( info.PhysicalDevice, &deviceInfo, 0, &device ) != VK_SUCCESS ) {
		throw std::runtime_error("Failed to create device!");
	}
	return device;
}

CVulkanDevice::CVulkanDevice( const CVulkanDeviceInfo& info_ ):
	CResourceType( makeVulkanDevice( info_ ) ),
	info( info_)
{
}
//------------------------------------------------------------------------------------------------------------

static inline VkInstance makeVulkanInstance( const VkInstanceCreateInfo& info )
{
	VkInstance instance;
	if( vkCreateInstance( &info, 0, &instance ) != VK_SUCCESS ) {
		throw std::runtime_error( "Failed to create instance!" );
	}
	return instance;
}

CVulkanInstance::CVulkanInstance():
	CResourceType( makeVulkanInstance( instanceCreateInfo ) )
{
	if( !enumDevices() ) {
		throw std::runtime_error( "Failed to get info about devices!" );
	}
}

// Gets the information about available devices
bool CVulkanInstance::enumDevices()
{
	uint32_t devCount = 0; 
	if( vkEnumeratePhysicalDevices( handle, &devCount, 0 ) != VK_SUCCESS ) {
		return false;
	}

	std::vector< VkPhysicalDevice, CrtAllocator<VkPhysicalDevice> > physicalDevices;
	physicalDevices.resize( devCount );
	if( vkEnumeratePhysicalDevices( handle, &devCount, physicalDevices.data() ) != VK_SUCCESS ) {
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
