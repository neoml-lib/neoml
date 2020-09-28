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

#include <NeoMathEngine/CrtAllocatedObject.h>
#include <VulkanDll.h>

#include <vulkan/vulkan.h>

namespace NeoML {

class CVulkanMemory : public CCrtAllocatedObject {
public:
	CVulkanMemory( const CVulkanDevice& _device, std::size_t _size,
		VkBufferUsageFlags _usage,
		VkMemoryPropertyFlags _properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT );
	
	~CVulkanMemory() noexcept { release(); }
	
	CVulkanMemory( const CVulkanMemory& ) = delete;
	CVulkanMemory& operator=( const CVulkanMemory& ) = delete;

	bool HostVisible() const { return ( properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ) != 0; }

	VkBuffer Buffer() const { return buffer; }

	VkDeviceMemory Memory() const { return memory; }
	
private:
	VkBuffer buffer;
	VkDeviceMemory memory;
	VkMemoryPropertyFlags properties;
	const CVulkanDevice& device;
	
	void release() noexcept
	{
		device.vkDestroyBuffer( buffer, nullptr );
		device.vkFreeMemory( memory, nullptr );
	}
};
} // namespace NeoML

#endif // NEOML_USE_VULKAN
