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

#include <VulkanImage.h>
#include <VulkanDll.h>

namespace NeoML {

CVulkanImage::CVulkanImage( const CVulkanDevice& vulkanDevice, int _width, int _height ) :
	device( vulkanDevice ),
	width( _width ),
	height( _height ),
	image( VK_NULL_HANDLE ),
	imageView( VK_NULL_HANDLE ),
	imageMemory( VK_NULL_HANDLE ),
	sampler( VK_NULL_HANDLE )
{
	VkImageCreateInfo imageInfo = {};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	imageInfo.extent = { (uint32_t)width, (uint32_t)height, (uint32_t)1 };
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	vkSucceded( device.vkCreateImage( &imageInfo, 0, &image ) );

	VkMemoryRequirements req = {};
	device.vkGetImageMemoryRequirements( image, &req );

	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = req.size;

	// Look for the suitable index in the memory
	bool isFound = false;
	const int flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	for( int i = 0; i < (int)device.MemoryProperties.memoryTypeCount; ++i ) {
		if( (req.memoryTypeBits & (1 << i)) != 0 && (int)(device.MemoryProperties.memoryTypes[i].propertyFlags & flags) == flags ) {
			allocInfo.memoryTypeIndex = i;
			isFound = true;
			break;
		}
	}
	ASSERT_EXPR(isFound); 
	vkSucceded( device.vkAllocateMemory( &allocInfo, 0, &imageMemory ) );
	vkSucceded( device.vkBindImageMemory( image, imageMemory, 0 ) );

	VkImageViewCreateInfo imageViewInfo = {};
	imageViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	imageViewInfo.image = image;
	imageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	imageViewInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	imageViewInfo.components =
		{ VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
	imageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	imageViewInfo.subresourceRange.baseMipLevel = 0;
	imageViewInfo.subresourceRange.levelCount = 1;
	imageViewInfo.subresourceRange.baseArrayLayer = 0;
	imageViewInfo.subresourceRange.layerCount = 1;
	vkSucceded( device.vkCreateImageView( &imageViewInfo, 0, &imageView ) );

	VkSamplerCreateInfo samplerInfo = {};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = VK_FILTER_NEAREST;
	samplerInfo.minFilter = VK_FILTER_NEAREST;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.mipLodBias = 0.f;
	samplerInfo.compareOp = VK_COMPARE_OP_NEVER;
	samplerInfo.minLod = 0.f;
	samplerInfo.maxLod = 0.f;
	samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;
	vkSucceded( device.vkCreateSampler( &samplerInfo, 0, &sampler ) );
}

CVulkanImage::~CVulkanImage()
{
	if( sampler != VK_NULL_HANDLE ) {
		device.vkDestroySampler( sampler, 0 );
	}

	if( imageMemory != VK_NULL_HANDLE ) {
		device.vkFreeMemory( imageMemory, 0 );
	}

	if( imageView != VK_NULL_HANDLE ) {
		device.vkDestroyImageView( imageView, 0 );
	}

	if( image != VK_NULL_HANDLE ) {
		device.vkDestroyImage( image, 0 );
	}
}

bool CVulkanImage::IsImageFit(int& newWidth, int& newHeight) const
{
	if( newWidth <= width && newHeight <= height ) {
		return true;
	}

	newWidth = max(newWidth, width);
	newHeight = max(newHeight, height);

	return false;
}

void CVulkanImage::SetWorkingArea(int workingWidth, int workingHeight)
{
	ASSERT_EXPR(workingWidth <= width && workingHeight <= height);
	workingWidth;
	workingHeight;
	// no more actions needed
}

void CVulkanImage::UpdateDescriptorSet( VkCommandBuffer, VkDescriptorSet desc, VkPipelineLayout, int index, bool isSampled) const
{
	VkDescriptorImageInfo imageInfo = {};
	imageInfo.imageView = imageView;
	imageInfo.sampler = sampler;
	imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkWriteDescriptorSet writeDesc = {};
	writeDesc.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeDesc.dstSet = desc;
	writeDesc.dstBinding = isSampled ? SAMPLER_BINDING_NUM( index ) : IMAGE_BINDING_NUM( index );
	writeDesc.descriptorCount = 1;
	writeDesc.descriptorType = isSampled ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	writeDesc.pImageInfo = &imageInfo;
	device.vkUpdateDescriptorSets( 1, &writeDesc, 0, 0 );
}

const VkImage& CVulkanImage::GetVkImage() const
{
	return image;
} 

} // namespace NeoML

#endif // NEOML_USE_VULKAN
