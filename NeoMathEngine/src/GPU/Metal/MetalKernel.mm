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

#include <MetalKernel.h>
#include <MemoryHandleInternal.h>

@import Foundation;
@import MetalKit;

namespace NeoML {

// Blob descriptor that is passed to a kernel
struct CMetalBlobDesc {
    int BatchLength; // the maximum sequence length for a recurrent network
    int BatchWidth; // the number of sequences in the blob
    int ListSize; // the number of elements in the list
    int Height;        // the blob height
    int Width;        // the blob width
    int Depth;        // the blob depth
    int Channels;    // the number of channels
};

//----------------------------------------------------------------------------------------------------------------------------

// We did not find that using more than defaultThreadCount(state.threadExecutionWidth) threads gives any performance increase
//          (tested on iPhone 7 and iPhone 8)
// The algorithm that determines thread group size seems simple and reasonably close to optimal for existing kernels
static void getMetalThreadgroupSize( int defaultThreadCount, int gridDepth, int gridHeight, int,
    int& threadgroupDepth, int& threadgroupHeight, int& threadgroupWidth )
{
    // Here we assume that defaultThreadCount(state.threadExecutionWidth) is a power of 2
    // That has been so up until iPhone X (the last as of when this code was written) and is not expected to change
    
    threadgroupDepth = 2;
    while( threadgroupDepth > gridDepth ) {
        threadgroupDepth /= 2;
    }
    
    threadgroupHeight = 4;
    while( threadgroupHeight > gridHeight ) {
        threadgroupHeight /= 2;
    }
   
    threadgroupWidth = defaultThreadCount >= ( threadgroupHeight * threadgroupDepth ) ?
        defaultThreadCount / ( threadgroupHeight * threadgroupDepth ) : 1;
    
    while( threadgroupDepth * threadgroupHeight * threadgroupWidth * 2 <= defaultThreadCount ) {
        threadgroupWidth *= 2;
    }
    
    // Finally, we get defaultThreadCount threads (or more if defaultThreadCount is not a power of 2 for some reason)
}

// Gets the number of thread groups for processing data of given size
static void getMetalThreadgroupCount( int gridDepth, int gridHeight, int gridWidth, const MTLSize& threadgroupSize,
    MTLSize& threadgroupCount )
{
    threadgroupCount.width = ( gridWidth + threadgroupSize.width - 1 ) / threadgroupSize.width;
    threadgroupCount.height = ( gridHeight + threadgroupSize.height - 1 ) / threadgroupSize.height;
    threadgroupCount.depth = ( gridDepth + threadgroupSize.depth - 1 ) / threadgroupSize.depth;
}

//----------------------------------------------------------------------------------------------------------------------------

CMetalKernel::CMetalKernel( CMetalCommandQueue& _queue, const char* name, int depthCombine,
        int heightCombine, int widthCombine, int depth, int height, int width ) :
	queue( _queue ),
    gridDepth( 0 ),
    gridHeight( 0 ),
    gridWidth( 0 ),
    threadgroupDepth( 0 ),
    threadgroupHeight( 0 ),
    threadgroupWidth( 0 ),
    computeEncoder( nil )
{
    ASSERT_EXPR( depth > 0 );
    ASSERT_EXPR( height > 0 );
    ASSERT_EXPR( width > 0 );
    ASSERT_EXPR( depthCombine > 0 );
    ASSERT_EXPR( heightCombine > 0 );
    ASSERT_EXPR( widthCombine > 0 );

    // Create a buffer to load the new kernel for processing
    commandBuffer = std::unique_ptr<CMetalCommandBuffer>( queue.CreateCommandBuffer() );
    if( commandBuffer->GetHandle() == nil ) {
        NSLog( @"NeoMathEngine: create command buffer error." );
        return;
    }

    computeEncoder = [[commandBuffer->GetHandle() computeCommandEncoder] retain];
    if( computeEncoder == nil ) {
        NSLog( @"NeoMathEngine: create compute encoder error." );
        return;
    }

    id<MTLComputePipelineState> state = queue.GetComputePipelineState(name);
    if( state == nil ) {
        [computeEncoder release];
        computeEncoder = nil;
        return;
    }
    [computeEncoder setComputePipelineState:state];

    // Calculate the size needed to execute the kernel
    gridDepth = ( depth + depthCombine - 1 ) / depthCombine;
    gridHeight = ( height + heightCombine - 1 ) / heightCombine;
    gridWidth = ( width + widthCombine - 1 ) / widthCombine;

    getMetalThreadgroupSize( (int)state.threadExecutionWidth, gridDepth, gridHeight, gridWidth,
        threadgroupDepth, threadgroupHeight, threadgroupWidth );
}

CMetalKernel::~CMetalKernel()
{
    [computeEncoder release];
}

void CMetalKernel::SetParam( const CConstFloatHandle& handle, int index )
{
    if( computeEncoder == nil ) {
        return;
    }

    [computeEncoder setBuffer: (id<MTLBuffer>)GetRawAllocation( handle ) offset: GetRawOffset( handle ) atIndex:index];
}

void CMetalKernel::SetParam( const CConstIntHandle& handle, int index )
{
    if( computeEncoder == nil ) {
        return;
    }

    [computeEncoder setBuffer: (id<MTLBuffer>)GetRawAllocation( handle ) offset: GetRawOffset( handle ) atIndex:index];
}

void CMetalKernel::SetParam( const CBlobDesc& desc, int index )
{
    if( computeEncoder == nil ) {
        return;
    }

    CMetalBlobDesc metalDesc;
    metalDesc.BatchLength = desc.BatchLength();
    metalDesc.BatchWidth = desc.BatchWidth();
    metalDesc.ListSize = desc.ListSize();
    metalDesc.Height = desc.Height();
    metalDesc.Width = desc.Width();
    metalDesc.Depth = desc.Depth();
    metalDesc.Channels = desc.Channels();
    
    [computeEncoder setBytes: &metalDesc length: sizeof(CMetalBlobDesc) atIndex:index];
}
    
void CMetalKernel::SetParam( float value, int index )
{
    if( computeEncoder == nil ) {
        return;
    }

    [computeEncoder setBytes: &value length: sizeof(float) atIndex:index];
}

void CMetalKernel::SetParam( int value, int index )
{
    if( computeEncoder == nil ) {
        return;
    }

    [computeEncoder setBytes: &value length: sizeof(int) atIndex:index];
}

void CMetalKernel::SetSharedParam( int size, int index )
{
    if( computeEncoder == nil ) {
        return;
    }

    [computeEncoder setThreadgroupMemoryLength:size atIndex:index];
}

bool CMetalKernel::Run( int threadgroupCountDepth, int threadgroupCountHeight, int threadgroupCountWidth )
{
    if( computeEncoder == nil ) {
        return false;
    }

    const MTLSize threadgroupSize = { (NSUInteger)threadgroupWidth, (NSUInteger)threadgroupHeight, (NSUInteger)threadgroupDepth };
    MTLSize threadgroupCount;
    getMetalThreadgroupCount( gridDepth, gridHeight, gridWidth, threadgroupSize, threadgroupCount );

    if( threadgroupCountDepth == 0 && threadgroupCountHeight == 0 && threadgroupCountWidth == 0 ) {
        [computeEncoder dispatchThreadgroups: threadgroupCount threadsPerThreadgroup: threadgroupSize];
    } else {
        MTLSize fixedThreadgroupCount;
        fixedThreadgroupCount.depth = ( threadgroupCountDepth != 0 ? threadgroupCountDepth : threadgroupCount.depth );
        fixedThreadgroupCount.height = ( threadgroupCountHeight != 0 ? threadgroupCountHeight : threadgroupCount.height );
        fixedThreadgroupCount.width = ( threadgroupCountWidth != 0 ? threadgroupCountWidth : threadgroupCount.width );

        [computeEncoder dispatchThreadgroups: fixedThreadgroupCount threadsPerThreadgroup: threadgroupSize];
    }
    [computeEncoder endEncoding];
    queue.CommitCommandBuffer( commandBuffer.get() );
    commandBuffer.reset();
    return true;
}

//----------------------------------------------------------------------------------------------------------------------------

C1DKernel::C1DKernel( CMetalCommandQueue& queue, const char* name, int widthCombine, int width ) :
    CMetalKernel( queue, name, 1, 1, widthCombine, 1, 1, width )
{
}

C2DKernel::C2DKernel( CMetalCommandQueue& queue, const char* name, int heightCombine, int widthCombine,
		int height, int width ) :
    CMetalKernel( queue, name, 1, heightCombine, widthCombine, 1, height, width )
{
}

C3DKernel::C3DKernel( CMetalCommandQueue& queue, const char* name, int depthCombine, int heightCombine, int widthCombine,
		int depth, int height, int width ) :
    CMetalKernel( queue, name, depthCombine, heightCombine, widthCombine, depth, height, width )
{
}

} // namespace NeoML

#endif // NEOML_USE_METAL
