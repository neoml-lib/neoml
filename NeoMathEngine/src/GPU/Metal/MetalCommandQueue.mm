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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_METAL

#include <MetalCommandQueue.h>

@import Foundation;
@import MetalKit;

@interface MathEngineHelper: NSObject
{
}

+(NSBundle*) GetBundle;

@end

@implementation MathEngineHelper

+(NSBundle*) GetBundle
{
	return [NSBundle bundleForClass:[self class]];
}

@end

namespace NeoML {

CMetalCommandBuffer::CMetalCommandBuffer( id<MTLCommandQueue> commandQueue ) :
	pool( [[NSAutoreleasePool alloc] init] ),
	handle( [[commandQueue commandBuffer] retain] )
{
	if( handle == nil ) {
		NSLog( @"NeoMathEngine: create command buffer error." );
	}
}

CMetalCommandBuffer::~CMetalCommandBuffer()
{
	[pool release];
	[handle release];
}

//------------------------------------------------------------------------------------------------------------

CMetalCommandQueue::CMetalCommandQueue() :
	device( nil ),
	commandQueue( nil ),
	commandBuffer( nil ),
	metalLibrary( nil ),
	pipelines( 0 )
{
}

bool CMetalCommandQueue::Create()
{
	device = MTLCreateSystemDefaultDevice();
	if( device == nil ) {
		NSLog( @"NeoMathEngine: MTLCreateSystemDefaultDevice error." );
		return false;
	}

	commandQueue = [device newCommandQueue];
	if( commandQueue == nil ) {
		NSLog( @"NeoMathEngine: create command queue error." );
		return false;
	}

	NSBundle *frameworkBundle = [MathEngineHelper GetBundle];
	NSString* metalLibraryPath = [frameworkBundle pathForResource:@"NeoMetalLib" ofType:@"metallib"];
	metalLibrary = [device newLibraryWithFile:metalLibraryPath error:nil];
	if( metalLibrary == nil ) {
		return false;
	}
	pipelines = [NSMutableDictionary new];
	return true;
}

CMetalCommandQueue::~CMetalCommandQueue()
{
	[commandBuffer release];
	[commandQueue release];
	[pipelines release];
	[metalLibrary release];
	[device release];
}

id<MTLComputePipelineState> CMetalCommandQueue::GetComputePipelineState( const char* name )
{
	NSString* kernelName = [[NSString alloc] initWithUTF8String: name];
	id<MTLComputePipelineState> computePipelineState;
	
	std::lock_guard<std::mutex> lock( mutex );
	if( pipelines[kernelName] ) {
		// The compute program has been compiled, use it
		computePipelineState = pipelines[kernelName];
	} else {
		// Load the program from the library
		id<MTLFunction> kernelFunction = [metalLibrary newFunctionWithName: kernelName];
		if( kernelFunction == nil ) {
			NSLog( @"NeoMathEngine: load MTLFunction error." );
			[kernelName release];
			return nil;
		}
		
		computePipelineState = [device newComputePipelineStateWithFunction: kernelFunction error:nil];
		[kernelFunction release];

		if( computePipelineState == nil ) {
			NSLog( @"NeoMathEngine: create MTLComputePipelineState error." );
			[kernelName release];
			return nil;
		}
		[pipelines setObject:computePipelineState forKey:kernelName];
	}
	[kernelName release];
	
	return computePipelineState;
}

CMetalCommandBuffer* CMetalCommandQueue::CreateCommandBuffer()
{
	return new CMetalCommandBuffer( commandQueue );
}

void CMetalCommandQueue::CommitCommandBuffer( CMetalCommandBuffer* newCommandBuffer )
{
	[newCommandBuffer->GetHandle() commit];
	
	std::lock_guard<std::mutex> lock( mutex );
	if( commandBuffer != nil ) {
		[commandBuffer release];
	}
	commandBuffer = newCommandBuffer->GetHandle();
	[commandBuffer retain];
}

void CMetalCommandQueue::WaitCommandBuffer()
{
	id<MTLCommandBuffer> bufferForWait = nil;
	{
		std::lock_guard<std::mutex> lock( mutex );

		if( commandBuffer != nil ) {
			bufferForWait = commandBuffer;
			[bufferForWait retain];
		}
	}

	if( bufferForWait != nil ) {
		[bufferForWait waitUntilCompleted];
		[bufferForWait release];
	}
}

} // namespace NeoML

#endif // NEOML_USE_METAL
