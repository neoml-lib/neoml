﻿/* Copyright © 2017-2020 ABBYY Production LLC

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

#include <shaders/common/CommonStruct.h>

namespace NeoML {

// A command in the queue
struct CCommand : public CCrtAllocatedObject {
	VkCommandBuffer Buffer;
	VkDescriptorPool DescriptorPool; // only for shaders
	VkDescriptorSet DescriptionSet; // only for shaders
	CCommand* Next;

	CCommand() : Buffer( 0 ), DescriptorPool( 0 ), DescriptionSet( 0 ), Next( 0 ) {}
};

//------------------------------------------------------------------------------------------------------------

CVulkanCommandQueue::CVulkanCommandQueue( const CVulkanDevice& vulkanDevice ) :
	device( vulkanDevice ),
	queue( VK_NULL_HANDLE ),
	commandPool( VK_NULL_HANDLE ),
	descriptionSetCount( 0 ),
	commandBufferCount( 0 ),
	commands( 0 )
{
	VkCommandPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	poolInfo.queueFamilyIndex = device.Family;
	vkSucceded( device.vkCreateCommandPool( &poolInfo, 0, &commandPool ) );

	device.vkGetDeviceQueue( device.Family, 0, &queue );
}

CVulkanCommandQueue::~CVulkanCommandQueue()
{
	CleanUp();

	device.vkDestroyCommandPool( commandPool, 0 );

	// The queue does not need to be closed with this API
}

void CVulkanCommandQueue::RunComputeShader( const CVulkanShaderData& shader, int countX, int countY, int countZ,
	const void* paramsBuffer, int paramsSize, const CVulkanImage** images, int imageCount,
	const CVulkanImage** samplers, int samplerCount, const CMemoryHandle* dataBuffers, const size_t* dataSizes, int dataBufferCount )
{
	assert( imageCount <= VulkanMaxBindingCount );
	assert( samplerCount <= VulkanMaxBindingCount );
	assert( dataBufferCount <= VulkanMaxBindingCount );

	CCommand* command = new CCommand();
	command->Buffer = getCommandBuffer();
	command->DescriptorPool = getDescriptorPool();

	// Allocate descriptors
	VkDescriptorSetAllocateInfo allocateInfo = {};
	allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocateInfo.descriptorPool = command->DescriptorPool;
	allocateInfo.descriptorSetCount = 1;
	allocateInfo.pSetLayouts = &shader.DescLayout;
	vkSucceded( device.vkAllocateDescriptorSets( &allocateInfo, &(command->DescriptionSet) ) );

	// Set the buffers for desc
	// The zero binding always contains the parameters of the call
	for( int i = 0; i < dataBufferCount; ++i ) {
		VkDescriptorBufferInfo descBufferInfo = {};
		descBufferInfo.buffer = GetRawAllocation( dataBuffers[i] )->Buffer();
		descBufferInfo.offset = GetRawOffset( dataBuffers[i] );
		descBufferInfo.range = dataSizes[i];

		VkWriteDescriptorSet writeDesc = {};
		writeDesc.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDesc.dstSet = command->DescriptionSet;
		writeDesc.dstBinding = i + 1;
		writeDesc.descriptorCount = 1;
		writeDesc.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writeDesc.pBufferInfo = &descBufferInfo;
		device.vkUpdateDescriptorSets( 1, &writeDesc, 0, 0 );
	}

	for( int i = 0; i < imageCount; ++i ) {
		images[i]->UpdateDescriptorSet( command->Buffer, command->DescriptionSet, shader.Layout, i, false );
	}

	for( int i = 0; i < samplerCount; ++i ) {
		samplers[i]->UpdateDescriptorSet( command->Buffer, command->DescriptionSet, shader.Layout, i, true );
	}

	if( paramsSize > 0 ) {
		device.vkCmdPushConstants( command->Buffer, shader.Layout, VK_SHADER_STAGE_COMPUTE_BIT, PUSH_CONSTANT_PARAM_OFFSET, paramsSize, paramsBuffer );
	}
	device.vkCmdBindPipeline( command->Buffer, VK_PIPELINE_BIND_POINT_COMPUTE, shader.Pipeline );
	device.vkCmdBindDescriptorSets( command->Buffer, VK_PIPELINE_BIND_POINT_COMPUTE, shader.Layout, 0, 1,
		&(command->DescriptionSet), 0, 0 );
	device.vkCmdDispatch( command->Buffer, countX, countY, countZ );

	submitCommand( command );
}

void CVulkanCommandQueue::RunUpdateBuffer( VkBuffer buffer, VkDeviceSize offset, const void* from, size_t size )
{
	assert( size <= VulkanMaxUpdateBufferSize );

	CCommand* command = new CCommand();
	command->Buffer = getCommandBuffer();

	device.vkCmdUpdateBuffer( command->Buffer, buffer, offset, size, from );

	submitCommand( command );
}

void CVulkanCommandQueue::RunFillBuffer( VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, int data )
{
	CCommand* command = new CCommand();
	command->Buffer = getCommandBuffer();

	device.vkCmdFillBuffer( command->Buffer, buffer, offset, size, data );

	submitCommand( command );
}

void CVulkanCommandQueue::RunCopyBuffer( VkBuffer from, VkBuffer to, const VkBufferCopy& info )
{
	CCommand* command = new CCommand();
	command->Buffer = getCommandBuffer();

	device.vkCmdCopyBuffer( command->Buffer, from, to, 1, &info );

	submitCommand( command );
}

void CVulkanCommandQueue::Wait()
{
	// Wait until all commands complete
	ASSERT_ERROR_CODE( device.vkQueueWaitIdle( queue ) );

	while( commands != 0 ) {
		CCommand* toDestroy = commands;

		if( toDestroy->DescriptorPool != VK_NULL_HANDLE && toDestroy->DescriptionSet != VK_NULL_HANDLE ) {
			vkSucceded( device.vkFreeDescriptorSets( toDestroy->DescriptorPool, 1, &(toDestroy->DescriptionSet) ) );
		}

		commands = commands->Next;
		delete toDestroy;
	}
	commandBufferCount = 0;
	descriptionSetCount = 0;
}

void CVulkanCommandQueue::CleanUp()
{
	Wait();

	if( commandBuffers.size() > 0 ) {
		device.vkFreeCommandBuffers( commandPool, static_cast<int>( commandBuffers.size() ), commandBuffers.data() );
		commandBuffers.clear();
	}

	if( descriptorPools.size() > 0 ) {
		for( size_t i = 0; i < descriptorPools.size(); ++i ) {
			device.vkDestroyDescriptorPool( descriptorPools[i], 0 );
		}
		descriptorPools.clear();
	}
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

	VkCommandBuffer cmdBuf = getCommandBuffer();
	device.vkCmdPipelineBarrier( cmdBuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, 0, 0, 0, 1, &info );
	vkSucceded( device.vkEndCommandBuffer( cmdBuf ) );

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cmdBuf;
	vkSucceded( device.vkQueueSubmit( queue, 1, &submitInfo, VK_NULL_HANDLE ) );
}

//------------------------------------------------------------------------------------------------------------
// private methods

// Gets the descriptors pool
VkDescriptorPool CVulkanCommandQueue::getDescriptorPool()
{
	// No special improvements when reusing (as with the buffer), but save the space for the sake of similarity
	if( static_cast<int>( descriptorPools.size() ) * VulkanMaxDescriptorSetPerPool > descriptionSetCount ) {
		descriptionSetCount++;
		return descriptorPools[( descriptionSetCount - 1 ) / VulkanMaxDescriptorSetPerPool];
	}

	// Create a new descriptor pool
	VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
	VkDescriptorPoolSize descPoolSize[4] = { {}, {}, {}, {} };
	descPoolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	descPoolSize[0].descriptorCount = VulkanMaxBindingCount * VulkanMaxDescriptorSetPerPool;
	if( !device.IsImageBased ) {
		descPoolSize[0].descriptorCount += VulkanMaxBindingCount * VulkanMaxDescriptorSetPerPool;
	}
	descPoolSize[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	descPoolSize[1].descriptorCount = VulkanMaxBindingCount * VulkanMaxDescriptorSetPerPool;
	if( device.IsImageBased ) {
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

	vkSucceded( device.vkCreateDescriptorPool( &descPoolInfo, 0, &descriptorPool ) );

	descriptorPools.push_back( descriptorPool );
	descriptionSetCount++;
	return descriptorPool;
}

// Gets the commands buffer
VkCommandBuffer CVulkanCommandQueue::getCommandBuffer()
{
	// The first use of a buffer takes a lot of time in our tests,
	// so we'll reuse buffers when possible
	VkCommandBuffer result = VK_NULL_HANDLE;
	if( static_cast<int>( commandBuffers.size() ) > commandBufferCount ) {
		commandBufferCount++;
		result = commandBuffers[commandBufferCount - 1];
	} else {
		VkCommandBufferAllocateInfo cmdBufferInfo = {};
		cmdBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cmdBufferInfo.commandPool = commandPool;
		cmdBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cmdBufferInfo.commandBufferCount = 1;
		vkSucceded( device.vkAllocateCommandBuffers( &cmdBufferInfo, &result ) );
		commandBuffers.push_back( result );
		commandBufferCount++;
	}

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkSucceded( device.vkBeginCommandBuffer( result, &beginInfo ) );

	return result;
}

// Queues a command
void CVulkanCommandQueue::submitCommand( CCommand* command )
{
	vkSucceded( device.vkEndCommandBuffer( command->Buffer ) );

	command->Next = commands;
	commands = command;

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &(command->Buffer);

	vkSucceded( device.vkQueueSubmit( queue, 1, &submitInfo, VK_NULL_HANDLE ) );
}

} // namespace NeoML

#endif // NEOML_USE_VULKAN
