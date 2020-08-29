#pragma once

#include <vulkan/vulkan_core.h>

#include <string>

// The macro checks if a vulkanAPI function call was successful
#define vkSucceded( functionCall ) { VkResult temp = functionCall; temp; assert( temp == VK_SUCCESS ); }

namespace NeoML {

class VulkanError: public std::runtime_error {
public:
	explicit VulkanError( VkResult error ):
		std::runtime_error( "VkResult = " + std::to_string( error ) )
	{}
}; 

inline void VkCheck( VkResult result )
{
	if( result != VK_SUCCESS ) {
		throw VulkanError( result );
	}
}

template <typename T>
class CResourceBase 
{
public:
	explicit CResourceBase( T handle_ ) noexcept :
		handle( handle_ ) {}

	CResourceBase( const CResourceBase& ) = delete;
	CResourceBase& operator=( const CResourceBase& ) = delete;

	operator T() const { return handle; }

protected: // data
	T handle;
};

template<typename T>
using Deleter = void(*)( T, const VkAllocationCallbacks* );

template <typename T, Deleter<T> D>
class CResource: public CResourceBase<T>
{
protected:
	using CResourceType = CResource;
public:
	explicit CResource( T handle ) noexcept:
		CResourceBase<T>( handle ) 
	{}
	
	~CResource() noexcept {
		if( handle != nullptr ) {
			D( handle, nullptr );
		}
	}
};

template<typename T>
using DeviceDeleter = void(*)( VkDevice, T, const VkAllocationCallbacks* );

template <typename T, DeviceDeleter<T> D>
class CDeviceResource: public CResourceBase<T>
{
public:
	explicit CDeviceResource( T handle, VkDevice device_ ) noexcept:
		CResourceBase<T>( handle ),
		device( device_ )
	{}

	~CDeviceResource() noexcept {
		if( handle != nullptr ) {
			D( device, handle, nullptr );
		}
	}
private:
	VkDevice device;
};

} // NeoML