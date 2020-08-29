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

#ifdef NEOML_USE_VULKAN

#include <VulkanMemory.h>
#include <VulkanDevice.h>

#include <stdexcept>

namespace NeoML {

static inline VkBuffer makeBuffer( VkDevice device, std::size_t _size, VkBufferUsageFlags _usage )
{
	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = _size;
	bufferInfo.usage = _usage;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VkBuffer result;
	VkCheck( vkCreateBuffer( device, &bufferInfo, nullptr, &result ) );
	return result;
}

static inline VkDeviceMemory allocMemory( const CVulkanDevice& device, VkBuffer buffer, VkMemoryPropertyFlags properties )
{
	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements( device, buffer, &memRequirements );

	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;

	bool isFound = false;
	for( uint32_t i = 0; i < device.MemoryProperties().memoryTypeCount; ++i ) {
		if( ( memRequirements.memoryTypeBits & (1u << i) ) != 0 &&
			( device.MemoryProperties().memoryTypes[i].propertyFlags & properties ) == properties ) {
			allocInfo.memoryTypeIndex = i;
			isFound = true;
			break;
		}
	}

	if( !isFound ) {
		throw std::runtime_error( "Failed to find suitable memory type!" );
	}

	VkDeviceMemory result;
	VkCheck( vkAllocateMemory( device, &allocInfo, nullptr, &result ) );
	return result;
}

CVulkanMemory::CVulkanMemory( const CVulkanDevice& _device, std::size_t _size, VkBufferUsageFlags _usage,
	VkMemoryPropertyFlags _properties ) :
	device( _device ),
	buffer( makeBuffer( device, _size, _usage ), device ),
	memory( allocMemory( device, buffer, _properties), device ),
	properties( _properties )
{
	VkCheck( vkBindBufferMemory( device, buffer, memory, 0 ) );
}
	
} // namespace NeoML

#endif // NEOML_USE_VULKAN
