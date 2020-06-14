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

#ifdef NEOML_USE_METAL

#include <NeoMathEngine/CrtAllocatedObject.h>

@import Foundation;
@import MetalKit;
@import std.__mutex_base;

namespace NeoML {

// The wrapper for the command buffer
class CMetalCommandBuffer : public CCrtAllocatedObject {
public:
	explicit CMetalCommandBuffer( id<MTLCommandQueue> commandQueue );
	~CMetalCommandBuffer();

	// Gets the buffer descriptor (nil if failed to create buffer)
	id<MTLCommandBuffer> GetHandle() const { return handle; }

private:
	NSAutoreleasePool* pool; // autorelease pool for the objects created between CreateCommandBuffer and CommitCommandBuffer
	id<MTLCommandBuffer> handle; // the buffer descriptor
};

//------------------------------------------------------------------------------------------------------------

// The default command queue for a metal device
class CMetalCommandQueue : public CCrtAllocatedObject {
public:
	CMetalCommandQueue();
	~CMetalCommandQueue();

	// Creates the mechanism
	// Returns false if failed to create
	bool Create();

	// Gets the device descriptor
	id<MTLDevice> GetDevice() const { return device; }

	// Gets the compiled compute program descriptor
	id<MTLComputePipelineState> GetComputePipelineState( const char* name );

	// Creates a command buffer
	CMetalCommandBuffer* CreateCommandBuffer();

	// Adds a command buffer to the queue
	void CommitCommandBuffer( CMetalCommandBuffer* commandBuffer );

	// Waits for all commands to be completed
	void WaitCommandBuffer();

private:
	std::mutex mutex; // used for thread-safe operation
	id<MTLDevice> device; // metal device descriptor
	id<MTLCommandQueue> commandQueue; // the processing device command queue
	id<MTLCommandBuffer> commandBuffer; // the current command buffer
	id<MTLLibrary> metalLibrary; // the program library
	NSMutableDictionary* pipelines; // the compiled programs
};

} // namespace NeoML

#endif // NEOML_USE_METAL
