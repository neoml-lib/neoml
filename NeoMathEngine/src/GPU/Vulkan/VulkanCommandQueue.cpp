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
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>

#include <VulkanCommandQueue.h>
#include <VulkanDll.h>
#include <VulkanShader.h>
#include <VulkanImage.h>

#include <atomic>

#include <shaders/common/CommonStruct.h>

namespace NeoML {

static std::atomic<std::size_t> ThreadCounter;
constexpr std::size_t MaxThreads= 16;

//------------------------------------------------------------------------------------------------------------

CVulkanCommandQueue::CVulkanCommandQueue( const CVulkanDevice& vulkanDevice ) :
	device( vulkanDevice ),
	queue( nullptr ),
	data( MaxThreads )
{
	device.vkGetDeviceQueue( device.Family, 0, &queue );
}

CVulkanCommandQueue::~CVulkanCommandQueue()
{
	for( auto& d : data ) {
		if( d.commandPool != nullptr ) {
			clean( d );
		}
	}
	// The queue does not need to be closed with this API
}

void CVulkanCommandQueue::RunComputeShader( const CVulkanShaderData& shader, int countX, int countY, int countZ,
	const void* paramsBuffer, int paramsSize, const CVulkanImage** images, int imageCount,
	const CVulkanImage** samplers, int samplerCount, const CMemoryHandle* dataBuffers, 
	const size_t* dataSizes, int dataBufferCount )
{
	assert( imageCount <= VulkanMaxBindingCount );
	assert( samplerCount <= VulkanMaxBindingCount );
	assert( dataBufferCount <= VulkanMaxBindingCount );

	// Set the buffers for desc
	// The zero binding always contains the parameters of the call
	auto& data = getCurrentData();
	auto commandBuffer = getCommandBuffer( data );
	auto descriptorSet = getDescriptorSet( data, &shader.DescLayout );
	for( int i = 0; i < dataBufferCount; ++i ) {
		VkDescriptorBufferInfo descBufferInfo = {};
		descBufferInfo.buffer = GetRawAllocation( dataBuffers[i] )->Buffer();
		descBufferInfo.offset = GetRawOffset( dataBuffers[i] );
		descBufferInfo.range = dataSizes[i];

		VkWriteDescriptorSet writeDesc = {};
		writeDesc.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDesc.dstSet = descriptorSet;
		writeDesc.dstBinding = i + 1;
		writeDesc.descriptorCount = 1;
		writeDesc.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writeDesc.pBufferInfo = &descBufferInfo;
		device.vkUpdateDescriptorSets( 1, &writeDesc, 0, 0 );
	}

	for( int i = 0; i < imageCount; ++i ) {
		images[i]->UpdateDescriptorSet( commandBuffer, descriptorSet, shader.Layout, i, false );
	}

	for( int i = 0; i < samplerCount; ++i ) {
		samplers[i]->UpdateDescriptorSet( commandBuffer, descriptorSet, shader.Layout, i, true );
	}

	if( paramsSize > 0 ) {
		device.vkCmdPushConstants( commandBuffer, shader.Layout, VK_SHADER_STAGE_COMPUTE_BIT, 
			PUSH_CONSTANT_PARAM_OFFSET, paramsSize, paramsBuffer );
	}
	device.vkCmdBindPipeline( commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, shader.Pipeline );
	device.vkCmdBindDescriptorSets( commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, shader.Layout, 0, 1,
		&(descriptorSet), 0, 0 );
	device.vkCmdDispatch( commandBuffer, countX, countY, countZ );

	submitCommand( data, commandBuffer );
}

void CVulkanCommandQueue::RunUpdateBuffer( VkBuffer buffer, VkDeviceSize offset, const void* from, size_t size )
{
	assert( size <= VulkanMaxUpdateBufferSize );
	
	auto& data = getCurrentData();
	auto commandBuffer = getCommandBuffer( data );
	
	device.vkCmdUpdateBuffer( commandBuffer, buffer, offset, size, from );

	submitCommand( data, commandBuffer );
}

void CVulkanCommandQueue::RunFillBuffer( VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, int value )
{
	auto& data = getCurrentData();
	auto commandBuffer = getCommandBuffer( data );
	
	device.vkCmdFillBuffer( commandBuffer, buffer, offset, size, value );
	
	submitCommand( data, commandBuffer );
}

void CVulkanCommandQueue::RunCopyBuffer( VkBuffer from, VkBuffer to, const VkBufferCopy& info )
{
	auto& data = getCurrentData();
	auto commandBuffer = getCommandBuffer( data );

	device.vkCmdCopyBuffer( commandBuffer, from, to, 1, &info );

	submitCommand( data, commandBuffer );
}

void CVulkanCommandQueue::wait( CData& data )
{
	if( data.commandBufferCount == 0 ) {
		return;
	}

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &data.commandBufferCache[data.commandBufferCount - 1];
	{
		std::lock_guard<std::mutex> lock( mutex );
		ASSERT_ERROR_CODE( device.vkQueueSubmit( queue, 1, &submitInfo, data.fence ) );
	}
	// Wait until all commands complete
	ASSERT_ERROR_CODE( device.vkWaitForFences( 1, &data.fence, VK_TRUE, uint64_t(-1) ) );

	auto& descriptorSets = data.descriptorSets;
	for( std::size_t i = 0, j = 0; i < descriptorSets.size(); i += VulkanMaxDescriptorSetPerPool, ++j ) {
		auto descriptorPool = data.descriptorPoolCache[j];
		uint32_t setCount = (std::min<uint32_t>)( static_cast<uint32_t>( descriptorSets.size() - i ), VulkanMaxDescriptorSetPerPool );
		vkSucceded( device.vkFreeDescriptorSets( descriptorPool, setCount, &( descriptorSets[i] ) ) );
	}
	descriptorSets.clear();

	vkSucceded( device.vkResetFences( 1, &data.fence ) );

	data.commandBufferCount = 0;
}

void CVulkanCommandQueue::clean( CData& data )
{
	assert( data.commandPool != nullptr );

	wait( data );

	auto& commandBuffers = data.commandBufferCache;
	if( !commandBuffers.empty() ) {
		device.vkFreeCommandBuffers( data.commandPool, static_cast<uint32_t>( commandBuffers.size() ), commandBuffers.data() );
		commandBuffers.clear();
	}

	for( auto descriptorPool: data.descriptorPoolCache ) {
		device.vkDestroyDescriptorPool( descriptorPool, nullptr );
	}
	data.descriptorPoolCache.clear();

	device.vkDestroyFence( data.fence, nullptr );
	data.fence = nullptr;

	device.vkDestroyCommandPool( data.commandPool, nullptr );
	data.commandPool = nullptr;
}

void CVulkanCommandQueue::RunChangeLayoutForImage( const CVulkanImage* nativeImage, VkImageLayout oldLayout, VkImageLayout newLayout )
{
	VkImageMemoryBarrier info = {};
	info.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	info.srcAccessMask = info.dstAccessMask =
		VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	info.oldLayout = oldLayout;
	info.newLayout = newLayout;
	info.srcQueueFamilyIndex = device.Family;
	info.dstQueueFamilyIndex = device.Family;
	info.image = nativeImage->GetVkImage();
	info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	info.subresourceRange.baseMipLevel = 0;
	info.subresourceRange.levelCount = 1;
	info.subresourceRange.baseArrayLayer = 0;
	info.subresourceRange.layerCount = 1;

	auto& data = getCurrentData();
	VkCommandBuffer cmdBuf = getCommandBuffer( data );
	device.vkCmdPipelineBarrier( cmdBuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, 0, 0, 0, 1, &info );
	
	submitCommand( data, cmdBuf );
}

//------------------------------------------------------------------------------------------------------------
// private methods

// Gets the descriptors pool
VkDescriptorSet CVulkanCommandQueue::getDescriptorSet( CData& data, const VkDescriptorSetLayout* layout )
{
	VkDescriptorPool descriptorPool = nullptr;
	// No special improvements when reusing (as with the buffer), but save the space for the sake of similarity
	if( data.descriptorPoolCache.size() * VulkanMaxDescriptorSetPerPool > data.descriptorSets.size() ) {
		descriptorPool = data.descriptorPoolCache[ data.descriptorSets.size() / VulkanMaxDescriptorSetPerPool ];
	} else {
		// Create a new descriptor pool
		VkDescriptorPoolSize descPoolSize[4] = { {}, {}, {}, {} };
		descPoolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descPoolSize[0].descriptorCount = VulkanMaxBindingCount * VulkanMaxDescriptorSetPerPool;
		if (!device.IsImageBased) {
			descPoolSize[0].descriptorCount += VulkanMaxBindingCount * VulkanMaxDescriptorSetPerPool;
		}
		descPoolSize[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descPoolSize[1].descriptorCount = VulkanMaxBindingCount * VulkanMaxDescriptorSetPerPool;
		if (device.IsImageBased) {
			descPoolSize[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descPoolSize[2].descriptorCount = VulkanMaxBindingCount * VulkanMaxDescriptorSetPerPool;
			descPoolSize[3].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			descPoolSize[3].descriptorCount = VulkanMaxBindingCount * VulkanMaxDescriptorSetPerPool;
		}

		VkDescriptorPoolCreateInfo descPoolInfo = {};
		descPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		descPoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
		descPoolInfo.maxSets = VulkanMaxDescriptorSetPerPool;
		descPoolInfo.poolSizeCount = device.IsImageBased ? 4 : 2;
		descPoolInfo.pPoolSizes = descPoolSize;

		vkSucceded(device.vkCreateDescriptorPool( &descPoolInfo, 0, &descriptorPool ) );

		data.descriptorPoolCache.push_back( descriptorPool );
	}
	
	// Allocate descriptors set
	VkDescriptorSet result = nullptr;

	VkDescriptorSetAllocateInfo allocateInfo{};
	allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocateInfo.descriptorPool = descriptorPool;
	allocateInfo.descriptorSetCount = 1;
	allocateInfo.pSetLayouts = layout;
	
	vkSucceded( device.vkAllocateDescriptorSets(  &allocateInfo, &( result ) ) );

	data.descriptorSets.push_back( result );
	return result;
}

CVulkanCommandQueue::CData& CVulkanCommandQueue::getCurrentData()
{
	thread_local std::size_t id = ThreadCounter++;
	assert( id < MaxThreads );

	auto& result = data[id];
	
	if ( result.commandPool == nullptr ) {
		// Create command pool
		VkCommandPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = device.Family;

		vkSucceded( device.vkCreateCommandPool( &poolInfo, 0, &result.commandPool ) );

		// Create fence
		const VkFenceCreateInfo fenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, {} };
		vkSucceded( device.vkCreateFence( &fenceInfo, nullptr, &result.fence ) );
	}
	return result;
}

// Gets the commands buffer
VkCommandBuffer CVulkanCommandQueue::getCommandBuffer( CData& data )
{
	auto commandPool = data.commandPool;
	auto& commandBuffers = data.commandBufferCache;
	auto& commandBufferCount = data.commandBufferCount;
	// The first use of a buffer takes a lot of time in our tests,
	// so we'll reuse buffers when possible
	VkCommandBuffer result = nullptr;
	if( static_cast<int>( commandBuffers.size() ) > commandBufferCount ) {
		result = commandBuffers[commandBufferCount];
	} else {
		VkCommandBufferAllocateInfo cmdBufferInfo = {};
		cmdBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cmdBufferInfo.commandPool = commandPool;
		cmdBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cmdBufferInfo.commandBufferCount = 1;
		vkSucceded( device.vkAllocateCommandBuffers( &cmdBufferInfo, &result ) );
		commandBuffers.push_back( result );	
	}
	++commandBufferCount;

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkSucceded( device.vkBeginCommandBuffer( result, &beginInfo ) );

	return result;
}

// Queues one command
void CVulkanCommandQueue::submitCommand( CData& data, VkCommandBuffer commandBuffer )
{
	vkSucceded( device.vkEndCommandBuffer( commandBuffer ) );

	if( data.commandBufferCount > 1 ) {
		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &data.commandBufferCache[data.commandBufferCount - 2];

		std::lock_guard<std::mutex> lock( mutex );
		ASSERT_ERROR_CODE( device.vkQueueSubmit( queue, 1, &submitInfo, nullptr ) );
	}
}

} // namespace NeoML

#endif // NEOML_USE_VULKAN
