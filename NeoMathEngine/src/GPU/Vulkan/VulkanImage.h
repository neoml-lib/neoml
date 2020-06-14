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
#include <vulkan/vulkan.h>
#include <shaders/common/CommonStruct.h>

namespace NeoML {

struct CVulkanDevice;

// Temporary processing images (FRGBA32)
enum TTmpVulkanImage {
	TVI_0 = 0,
	TVI_1,
	TVI_2,
	TVI_3,
	TVI_4,

	TVI_Count,

	// Alternative names for the image types
	TVI_ConvSource = TVI_0,
	TVI_ConvFilter = TVI_1,
	TVI_FreeTerm = TVI_2,
	TVI_MatrixLeft = TVI_3,
	TVI_MatrixRight = TVI_4,
	TVI_DiagMatrix = TVI_2
};

// The class implementing a Vulkan image (FRGBA32)
class CVulkanImage : public CCrtAllocatedObject {
public:
	CVulkanImage( CVulkanDevice& device, int _width, int _height );
	~CVulkanImage();

	// Checks if an image of newWidth x newHeight size will fit in this one
	bool IsImageFit( int& newWidth, int& newHeight ) const;

	// Sets the image size for the next call
	void SetWorkingArea( int workingWidth, int workingHeight );

	// Sets the descriptor set parameters before use
	void UpdateDescriptorSet( VkCommandBuffer commandBuffer, VkDescriptorSet desc, VkPipelineLayout pipelineLayout,
		int index, bool isSampled ) const;

	const VkImage& GetVkImage() const; 
private:
	CVulkanDevice& device;
	const int width;
	const int height;

	VkImage image;
	VkImageView imageView;
	VkDeviceMemory imageMemory;
	VkSampler sampler;
};

} // namespace NeoML

#endif // NEOML_USE_VULKAN
