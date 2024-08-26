/* Copyright © 2017-2024 ABBYY

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

#include <algorithm>
#include <DllLoader.h>
#include <VulkanMathEngine.h>
#include <MathEngineCommon.h>
#include <MemoryHandleInternal.h>
#include <VulkanDll.h>
#include <VulkanCommandQueue.h>
#include <VulkanShader.h>

namespace NeoML {

// Include the shaders code
#include <shaders/generated/VectorToImage.h>

// The maximum number of bytes that may be asynchronously copied into GPU
constexpr size_t VulkanMaxExchangeAsyncSize = 65536;

// The size of the buffer for data exchange
constexpr size_t VulkanExchangeBufferSize = 4 * 1024 * 1024; // 4 MB

// The maximum number of groups over the X dimension when working with a 1D (vector) shader
// With larger sizes, the shader data will be represented in two dimensions
constexpr int VulkanMaxVectorXGroupCount = 8192;

//------------------------------------------------------------------------------------------------------------

static inline void getDeviceInfo( const CVulkanDeviceInfo& deviceInfo, CMathEngineInfo& info )
{
	info.Type = MET_Vulkan;
	::memset( info.Name, 0, sizeof( info.Name ) );
	::strcpy( info.Name, deviceInfo.Properties.deviceName );
	info.Id = deviceInfo.DeviceID;
	info.AvailableMemory = deviceInfo.AvailableMemory;
}

bool LoadVulkanEngineInfo( const CVulkanDll& dll, std::vector< CMathEngineInfo, CrtAllocator<CMathEngineInfo> >& result )
{
	for( const auto& deviceInfo : dll.GetDevices() ) {
		result.emplace_back();
		getDeviceInfo( deviceInfo, result.back() );
	}
	return !result.empty();
}

//------------------------------------------------------------------------------------------------------------

constexpr int VulkanMemoryAlignment = 16;

CVulkanMathEngine::CVulkanMathEngine( std::unique_ptr<const CVulkanDevice>& _device, size_t memoryLimit ) :
	dllLoader( CDllLoader::VULKAN_DLL ),
	device( std::move( _device ) ),
	tmpImages( TVI_Count, nullptr )
{
	ASSERT_EXPR( device != 0 ); // failed to create the device
	shaderLoader = std::unique_ptr<CVulkanShaderLoader>( new CVulkanShaderLoader( *device ) );
	commandQueue = std::unique_ptr<CVulkanCommandQueue>( new CVulkanCommandQueue( *device ) );
	memoryLimit = std::min<size_t>( memoryLimit == 0 ? SIZE_MAX : memoryLimit, device->AvailableMemory );

	InitializeMemory( this, memoryLimit, VulkanMemoryAlignment, /*reuse*/false, /*hostStack*/true );
}

CVulkanMathEngine::~CVulkanMathEngine()
{
	CleanUp();
}

void CVulkanMathEngine::CleanUpSpecial()
{
	commandQueue->CleanUp();
	for( auto& cur : tmpImages ) {
		delete cur;
		cur = 0;
	}
}

void CVulkanMathEngine::DataExchangeRaw( const CMemoryHandle& to, const void* from, size_t size )
{
	ASSERT_EXPR( to.GetMathEngine() == this );

	CTypedMemoryHandle<char> toPtr( to );
	const char* fromPtr = reinterpret_cast<const char*>( from );

	std::lock_guard<std::mutex> lock( Mutex );
	while( size != 0 ) {
		CVulkanMemory* vulkanMemory = GetRawAllocation( toPtr );
		ptrdiff_t vulkanOffset = GetRawOffset( toPtr );

		if( size <= VulkanMaxExchangeAsyncSize ) {
			// We can write the data asynchronously
			commandQueue->RunUpdateBuffer( vulkanMemory->Buffer(), vulkanOffset, fromPtr, size );
			break;
		}

		commandQueue->Wait(); // make sure that the exchange buffer is not used by anyone else

		size_t toCopy = size;
		if( toCopy > VulkanExchangeBufferSize ) {
			toCopy = VulkanExchangeBufferSize;
		}

		if( vulkanMemory->HostVisible() ) { 			
			void* mappedData = nullptr;
			vkSucceded( device->vkMapMemory( vulkanMemory->Memory(), vulkanOffset, toCopy, 0, &mappedData ) );
			memcpy( mappedData, fromPtr, toCopy );
			device->vkUnmapMemory( vulkanMemory->Memory() );
		} else {
			CVulkanMemory stagingMemory( *device, toCopy, VK_BUFFER_USAGE_TRANSFER_SRC_BIT );
			
			void* mappedData = nullptr;
			vkSucceded( device->vkMapMemory( stagingMemory.Memory(), 0, toCopy, 0, &mappedData ) );
			memcpy( mappedData, fromPtr, toCopy );
			device->vkUnmapMemory( stagingMemory.Memory() );

			VkBufferCopy copyRegion{};
			copyRegion.srcOffset = 0;
			copyRegion.dstOffset = vulkanOffset;
			copyRegion.size = toCopy;

			commandQueue->RunCopyBuffer( stagingMemory.Buffer(), vulkanMemory->Buffer(), copyRegion );
			commandQueue->Wait();
		}
		size -= toCopy;
		toPtr += toCopy;
		fromPtr += toCopy;
	}
}

void CVulkanMathEngine::DataExchangeRaw( void* to, const CMemoryHandle& from, size_t size )
{
	ASSERT_EXPR( from.GetMathEngine() == this );

	CTypedMemoryHandle<char> fromPtr( from );
	char* toPtr = reinterpret_cast<char*>( to );

	std::lock_guard<std::mutex> lock( Mutex );
	while( size != 0 ) {
		CVulkanMemory* vulkanMemory = GetRawAllocation( fromPtr );
		ptrdiff_t vulkanOffset = GetRawOffset( fromPtr );

		size_t toCopy = size;
		if( toCopy > VulkanExchangeBufferSize ) {
			toCopy = VulkanExchangeBufferSize;
		}

		commandQueue->Wait(); // wait for the data to be written into the exchange buffer

		if( vulkanMemory->HostVisible() ) {
			void* mappedData = nullptr;
			vkSucceded( device->vkMapMemory( vulkanMemory->Memory(), vulkanOffset, toCopy, 0, &mappedData ) );
			memcpy( toPtr, mappedData, toCopy );
			device->vkUnmapMemory( vulkanMemory->Memory() );
		} else {
			CVulkanMemory stagingMemory( *device, toCopy, VK_BUFFER_USAGE_TRANSFER_DST_BIT );

			VkBufferCopy copyRegion{};
			copyRegion.srcOffset = vulkanOffset;
			copyRegion.dstOffset = 0;
			copyRegion.size = toCopy;

			commandQueue->RunCopyBuffer( vulkanMemory->Buffer(), stagingMemory.Buffer(), copyRegion );
			commandQueue->Wait();

			void* mappedData = nullptr;
			vkSucceded( device->vkMapMemory( stagingMemory.Memory(), 0, toCopy, 0, &mappedData ) );
			memcpy( toPtr, mappedData, toCopy );
			device->vkUnmapMemory( stagingMemory.Memory() );
		}
		size -= toCopy;
		fromPtr += toCopy;
		toPtr += toCopy;
	}
}

CMemoryHandle CVulkanMathEngine::Alloc( size_t size )
{
	VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | 
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT | 
		VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags properties = device->Type != VDT_Nvidia ?
		VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT : 
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	
	auto vulkanMemory = new CVulkanMemory( *device, size, usage, properties );

	return CMemoryHandleInternal::CreateMemoryHandle( &mathEngine(), vulkanMemory );
}

void CVulkanMathEngine::Free( const CMemoryHandle& handle )
{
	ASSERT_EXPR( handle.GetMathEngine() == this );

	// Make sure that the memory we are about to clean is not used
	commandQueue->Wait();

	CVulkanMemory* vulkanMemory = GetRawAllocation( handle );
	delete vulkanMemory;
}

void CVulkanMathEngine::GetMathEngineInfo( CMathEngineInfo& info ) const
{
	getDeviceInfo( device->Info(), info );
}

//------------------------------------------------------------------------------------------------------------
// private methods

// Calculate the channels group size (intended for images with large height)
int CVulkanMathEngine::getChannelGroupSize( int height, int channels ) const
{
	ASSERT_EXPR(height > 0);
	ASSERT_EXPR(channels > 0);

	if( !device->IsImageBased ) {
		return channels; // no images used, so no limitations on geometric size
	}

	if( height * channels <= static_cast<int>( device->Properties.limits.maxImageDimension2D ) ) {
		return channels; // the image fits into the limitations
	}

	return device->Properties.limits.maxImageDimension2D / height;
}

// Gets a temporary object with the given id and size
const CVulkanImage* CVulkanMathEngine::getTmpImage( TTmpVulkanImage imageId, int width, int height )
{
	ASSERT_EXPR( device->IsImageBased );

	int newWidth = width;
	int newHeight = height;
	if( tmpImages[imageId] != 0 && tmpImages[imageId]->IsImageFit(newWidth, newHeight) ) {
		tmpImages[imageId]->SetWorkingArea(width, height);
		return tmpImages[imageId];
	}

	if( tmpImages[imageId] != 0 ) {
		commandQueue->Wait(); // make sure the previous temporary image is not used
		delete tmpImages[imageId];
		tmpImages[imageId] = 0;
	}

	CVulkanImage *image = new CVulkanImage( *device, newWidth, newHeight );
	commandQueue->RunChangeLayoutForImage( image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL );
	tmpImages[imageId] = image;

	tmpImages[imageId]->SetWorkingArea(width, height);
	return tmpImages[imageId];
}

const CVulkanImage* CVulkanMathEngine::getTmpImage( TTmpVulkanImage imageId )
{
	ASSERT_EXPR( device->IsImageBased );
	ASSERT_EXPR( tmpImages[imageId] != 0 );
	return tmpImages[imageId];
}

void CVulkanMathEngine::runShader( const CVulkanShaderData& shader, const void* param, int paramSize,
	const CVulkanImage** images, int imageCount, const CVulkanImage** samplers, int samplerCount,
	const CMemoryHandle* dataBuffers, const size_t* dataSizes, int dataBufferCount,
	int countX, int countY, int countZ )
{
	std::lock_guard<std::mutex> lock( Mutex );
	commandQueue->RunComputeShader( shader, Ceil(countX, shader.GroupSizeX), Ceil(countY, shader.GroupSizeY), Ceil(countZ, shader.GroupSizeZ),
		param, paramSize, images, imageCount, samplers, samplerCount,
		dataBuffers, dataSizes, dataBufferCount );
}

void CVulkanMathEngine::runVectorShader( const CVulkanShaderData& shader, const void* param, int paramSize,
	const CVulkanImage** images, int imageCount, const CVulkanImage** samplers, int samplerCount,
	const CMemoryHandle* dataBuffers, const size_t* dataSizes, int dataBufferCount, int count )
{
	int groupCountX = Ceil(count, shader.GroupSizeX);
	int groupCountY = Ceil(groupCountX, VulkanMaxVectorXGroupCount);
	groupCountX = std::min<int>(groupCountX, VulkanMaxVectorXGroupCount);

	ASSERT_EXPR(shader.GroupSizeY == 1 && shader.GroupSizeZ == 1);

	std::lock_guard<std::mutex> lock( Mutex );
	commandQueue->RunComputeShader( shader, groupCountX, groupCountY, 1,  param, paramSize, images, imageCount, samplers, samplerCount,
		dataBuffers, dataSizes, dataBufferCount );
}

const CVulkanImage& CVulkanMathEngine::batchVectorToImage( int batchSize, const CConstFloatHandle& vector, int size, int imageId )
{
	int size4 = Ceil(size, 4);
	const CVulkanImage* images[1] = { getTmpImage((TTmpVulkanImage)imageId, size4, batchSize) };

	CMemoryHandle bufs[1] = { vector };
	size_t sizes[1] = { static_cast<size_t>( batchSize ) * size * sizeof( float ) };

	PARAM_STRUCT(VectorToImage) param = { batchSize, size };

	runVectorShader( shaderLoader->GET_SHADER_DATA( VectorToImage, true, 1, 0, 1 ),
		&param, sizeof(param), images, 1, 0, 0, bufs, sizes, 1, size4 * batchSize );

	return *images[0];
}

} // namespace NeoML

#endif // NEOML_USE_VULKAN
