#pragma once

#include <vulkan/vulkan_core.h>

// The macro checks if a vulkanAPI function call was successful
#define vkSucceded( functionCall ) { VkResult temp = functionCall; temp; assert( temp == VK_SUCCESS ); }

namespace NeoML {

template<typename T> 
using Deleter = void(*)(T, const VkAllocationCallbacks*);

template<typename T> 
using DeviceDeleter = void(*)(VkDevice, T, const VkAllocationCallbacks*);

// RAII for vulkan resources
template <typename T, Deleter<T> D>
class CResource {
public:
	explicit CResource( T handle_ ) noexcept: 
		handle( handle_ ) {}

	CResource( const CResource& ) = delete;
	CResource& operator=(const CResource& ) = delete;

	~CResource() noexcept {
		if( handle != nullptr ) {
			D( handle, nullptr );
		}
	}

	operator T() const { return handle; }

protected: // data
	T handle;
	using Base = CResource;
};

} // NeoML