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

#include <shaders/common/CommonStruct.h>
#include <VulkanShader.h>
#include <VulkanDll.h>

namespace NeoML {

// The number of threads run for MaliBifrost. We selected this number experimentally for best performance
const int VulkanMaliBifrostThreadCount1D = 32;
const int VulkanMaliBifrostThreadCount2D_X = 4;
const int VulkanMaliBifrostThreadCount2D_Y = 8;
// The number of threads run for Adreno and Regular GPU
const int VulkanAdrenoRegularThreadCount1D = 64;
const int VulkanAdrenoRegularThreadCount2D_X = 8;
const int VulkanAdrenoRegularThreadCount2D_Y = 8;

CVulkanShaderLoader::CVulkanShaderLoader( CVulkanDevice& vulkanDevice ) :
	device( vulkanDevice )
{
	shaders.insert( shaders.begin(), SH_Count, 0 );
}

CVulkanShaderLoader::~CVulkanShaderLoader()
{
	for( size_t i = 0; i < shaders.size(); ++i ) {
		if( shaders[i] == 0 ) {
			continue;
		}

		if( shaders[i]->Module != VK_NULL_HANDLE ) {
			device.vkDestroyShaderModule( device.Handle, shaders[i]->Module, 0 );
		}
		if( shaders[i]->Pipeline != VK_NULL_HANDLE ) {
			device.vkDestroyPipeline( device.Handle, shaders[i]->Pipeline, 0 );
		}
		if( shaders[i]->Layout != VK_NULL_HANDLE ) {
			device.vkDestroyPipelineLayout( device.Handle, shaders[i]->Layout, 0 );
		}
		if( shaders[i]->DescLayout != VK_NULL_HANDLE ) {
			device.vkDestroyDescriptorSetLayout( device.Handle, shaders[i]->DescLayout, 0 );
		}

		delete shaders[i];
	}
}

const CVulkanShaderData& CVulkanShaderLoader::GetShaderData(TShader id, bool isIB, const uint32_t* code, int codeLen,
		size_t paramSize, int imageCount, int samplerCount, int bufferCount, int dimensions)
{
	assert( imageCount <= IMAGE_MAX_COUNT );
	assert( samplerCount <= SAMPLER_MAX_COUNT );
	assert( dimensions >= 1 );
	assert( dimensions <= 3 );

	if( shaders[id] == 0 ) {
		// Create the shader data
		shaders[id] = new CVulkanShaderData();

		shaders[id]->IsImageBased = isIB && device.IsImageBased;

		VkShaderModuleCreateInfo shaderInfo = {};
		shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		shaderInfo.pCode = code;
		shaderInfo.codeSize = codeLen;

		vkSucceded( device.vkCreateShaderModule( device.Handle, &shaderInfo, 0, &shaders[id]->Module ) );

		std::vector< VkDescriptorSetLayoutBinding, CrtAllocator<VkDescriptorSetLayoutBinding> > bindingInfo;

		bindingInfo.reserve( 2 * VulkanMaxBindingCount + 1 );
		int curBinding = 1;
		for( int i = 0; i < bufferCount; ++i ) {
			VkDescriptorSetLayoutBinding info = {};
			info.binding = curBinding++;
			info.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			info.descriptorCount = 1;
			info.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			bindingInfo.push_back( info );
		}
		for( int i = 0; i < imageCount; ++i ) {
			VkDescriptorSetLayoutBinding info = {};
			info.binding = IMAGE_BINDING_NUM(i);
			info.descriptorType =
				shaders[id]->IsImageBased ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			info.descriptorCount = 1;
			info.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			bindingInfo.push_back( info );
		}
		for( int i = 0; i < samplerCount; ++i ) {
			VkDescriptorSetLayoutBinding info = {};
			info.binding = SAMPLER_BINDING_NUM(i);
			info.descriptorType =
				shaders[id]->IsImageBased ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			info.descriptorCount = 1;
			info.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			bindingInfo.push_back( info );
		}
		VkDescriptorSetLayoutCreateInfo descInfo = {};
		descInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		descInfo.bindingCount = static_cast<int>( bindingInfo.size() );
		descInfo.pBindings = bindingInfo.data();
		vkSucceded( device.vkCreateDescriptorSetLayout( device.Handle, &descInfo, 0, &shaders[id]->DescLayout ) );

		VkPipelineLayoutCreateInfo layoutInfo = {};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = 1;
		layoutInfo.pSetLayouts = &shaders[id]->DescLayout;
		if( paramSize > 0 || samplerCount > 0 || imageCount > 0 ) {
			size_t pushConstantSize = PUSH_CONSTANT_PARAM_OFFSET + paramSize;
			assert( pushConstantSize <= device.Properties.limits.maxPushConstantsSize );
			VkPushConstantRange pushConstantRange = { VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>( pushConstantSize ) };
			layoutInfo.pushConstantRangeCount = 1;
			layoutInfo.pPushConstantRanges = &pushConstantRange;
		}

		vkSucceded( device.vkCreatePipelineLayout( device.Handle, &layoutInfo, 0, &shaders[id]->Layout ) );

		int threadGroupSizeX;
		int threadGroupSizeY;
		int threadGroupSizeZ;
		calculateThreadGroupSize( dimensions, threadGroupSizeX, threadGroupSizeY, threadGroupSizeZ );

		std::vector<VkSpecializationMapEntry> specializationMapEntries( 3 );
        specializationMapEntries[0].constantID = 0;
        specializationMapEntries[0].offset = 0 * sizeof(int);
        specializationMapEntries[0].size = sizeof(int);

        specializationMapEntries[1].constantID = 1;
        specializationMapEntries[1].offset = 1 * sizeof(int);
        specializationMapEntries[1].size = sizeof(int);

        specializationMapEntries[2].constantID = 2;
        specializationMapEntries[2].offset = 2 * sizeof(int);
        specializationMapEntries[2].size = sizeof(int);

        int specializationData[3] = { threadGroupSizeX, threadGroupSizeY, threadGroupSizeZ };

		VkSpecializationInfo specializationInfo;
	    specializationInfo.mapEntryCount = 3;
	    specializationInfo.pMapEntries = &specializationMapEntries[0];
	    specializationInfo.dataSize = 3 * sizeof(int);
	    specializationInfo.pData = specializationData;

		VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo;
	    pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	    pipelineShaderStageCreateInfo.pNext = 0;
	    pipelineShaderStageCreateInfo.flags = 0;
	    pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	    pipelineShaderStageCreateInfo.module = shaders[id]->Module;
	    pipelineShaderStageCreateInfo.pName = "main";
	    pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;

		VkComputePipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		pipelineInfo.stage = pipelineShaderStageCreateInfo;
		pipelineInfo.layout = shaders[id]->Layout;
		vkSucceded( device.vkCreateComputePipelines( device.Handle, VK_NULL_HANDLE, 1, &pipelineInfo, 0, &shaders[id]->Pipeline ) );

		shaders[id]->GroupSizeX = threadGroupSizeX;
		shaders[id]->GroupSizeY = threadGroupSizeY;
		shaders[id]->GroupSizeZ = threadGroupSizeZ;
	}

	return *shaders[id];
}

void CVulkanShaderLoader::calculateThreadGroupSize(int dimensions, 
	int& threadGroupSizeX, int& threadGroupSizeY, int& threadGroupSizeZ) const
{
	assert( dimensions >= 1 );
	assert( dimensions <= 3 );

	switch( dimensions ) {
		case 1: {
			threadGroupSizeX = 
				( device.Type == VDT_MaliBifrost ) ? VulkanMaliBifrostThreadCount1D : VulkanAdrenoRegularThreadCount1D;
			threadGroupSizeY = 1;
			threadGroupSizeZ = 1;
			break;
		}
		case 2:
		case 3: {
			threadGroupSizeX = 
				( device.Type == VDT_MaliBifrost ) ? VulkanMaliBifrostThreadCount2D_X : VulkanAdrenoRegularThreadCount2D_X;
			threadGroupSizeY = 
				( device.Type == VDT_MaliBifrost ) ? VulkanMaliBifrostThreadCount2D_Y : VulkanAdrenoRegularThreadCount2D_Y;
			threadGroupSizeZ = 1;
			break;
		}
	}
}


} // namespace NeoML

#endif // NEOML_USE_VULKAN
