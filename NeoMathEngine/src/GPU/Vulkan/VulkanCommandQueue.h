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

#include <cassert>

#include <vector>
#include <mutex>

#include <vulkan/vulkan.h>

#include <MathEngineAllocator.h>
#include <NeoMathEngine/MemoryHandle.h>

namespace NeoML {

struct CVulkanShaderData;
struct CVulkanDevice;
class CVulkanImage;

// The maximum number of bytes that may be asynchronously copied to GPU memory
constexpr size_t VulkanMaxUpdateBufferSize = 65536;

// The number of descriptors in the pool
constexpr int VulkanMaxDescriptorSetPerPool = 128;

//------------------------------------------------------------------------------------------------------------
// The shader execution mechanism
// The executing resources are distributed on the stack allocator principle 
// (we noted substantial performance increase from this)
// The resources are freed by calling CleanUp()
class CVulkanCommandQueue : public CCrtAllocatedObject {
public:
	explicit CVulkanCommandQueue( const CVulkanDevice& device );
	~CVulkanCommandQueue();

	CVulkanCommandQueue( const CVulkanCommandQueue& ) = delete;
	CVulkanCommandQueue& operator=(const CVulkanCommandQueue&) = delete;

	// Add a shader to the compute queue
	void RunComputeShader( const CVulkanShaderData& shader, int countX, int countY, int countZ,
		const void* param, int paramSize,
		const CVulkanImage** images, int imageCount,
		const CVulkanImage** samplers, int samplerCount,
		const CMemoryHandle* dataBuffers, const size_t* dataSizes, int dataBufferCount );

	// Queue buffer update
	void RunUpdateBuffer( VkBuffer buffer, VkDeviceSize offset, const void* from, size_t size );

	// Queue buffer filling
	void RunFillBuffer( VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, int data );

	// Queue buffer copying
	void RunCopyBuffer( VkBuffer from, VkBuffer to, const VkBufferCopy& info );

	// Queue repacking an image into new layout
	void RunChangeLayoutForImage( const CVulkanImage* nativeImage, VkImageLayout oldLayout, VkImageLayout newLayout );

	// Wait for all commands in current thread is finished
	void Wait() { wait( getCurrentData() ); }

	// Release all temporary resources in current thread
	void CleanUp() { clean( getCurrentData() ); }

private:
	const CVulkanDevice& device; // the processing device

	std::mutex mutex;
	VkQueue queue; // the queue handle
	
	struct CData {
		VkCommandPool commandPool;
		VkFence fence;
		vector<VkCommandBuffer> commandBufferCache;
		int commandBufferCount;
		vector<VkDescriptorPool> descriptorPoolCache;
		vector<VkDescriptorSet> descriptorSets;

		CData() :
			commandPool( nullptr ),
			fence( nullptr ),
			commandBufferCount( 0 )
		{}
	};
	std::vector<CData> data;

	void wait( CData& data );
	void clean( CData& data );
	
	CData& getCurrentData();

	VkDescriptorSet getDescriptorSet( CData& data, const VkDescriptorSetLayout* layout );
	VkCommandBuffer getCommandBuffer( CData& data );
	void submitCommand( CData& data, VkCommandBuffer buffer );
};

} // namespace NeoML

#endif // NEOML_USE_VULKAN
