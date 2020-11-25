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

#include <NeoMathEngine/CrtAllocatedObject.h>
#include <NeoMathEngine/BlobDesc.h>
#include <NeoMathEngine/MemoryHandle.h>
#include <MathEngineCommon.h>
#include <MetalCommandQueue.h>
#include <memory>

@import Foundation;
@import MetalKit;

namespace NeoML {

// The base class for a kernel
class CMetalKernel : public CCrtAllocatedObject {
public:
    // Gets the number of threads in the execution pool
    int GetThreadCount() const { return threadgroupDepth * threadgroupHeight * threadgroupWidth; }
    
    // Gets the width of the grid
    int GetGridWidth() const { return gridWidth; }
    
    // Gets the height of the gird (1 for 1D kernels)
    int GetGridHeight() const { return gridHeight; }
    
    // Gets the depth of the grid (1 for 1D and 2D kernels)
    int GetGridDepth() const { return gridDepth; }
    
    // Sets the kernel parameters for different types
    // index is the parameter number (the numbering starts with 0)
    void SetParam( const CConstIntHandle& handle, int index );
    void SetParam( const CConstFloatHandle& handle, int index );
    void SetParam( const CBlobDesc& desc, int index );
    void SetParam( float value, int index );
    void SetParam( int value, int index );

    // Sets the shared memory size
    void SetSharedParam( int size, int index );
    
    // Starts asynchronous operation with the specified number of thread groups
    // If the blocks value over the given dimension == 0, the precalculated value is used
    // Returns false if couldn't start the kernel
    bool Run( int depth = 0, int height = 0, int width = 0 );
    
protected:
    CMetalKernel( CMetalCommandQueue& queue, const char* name,
        int depthCombine, int heightCombine, int wdthCombine, int depth, int height, int width );
    ~CMetalKernel();

	CMetalCommandQueue& queue; // the command queue
    int gridDepth; // grid depth
    int gridHeight; // grid height
    int gridWidth; // grid width
    int threadgroupDepth; // thread group depth
    int threadgroupHeight; // thread group height
    int threadgroupWidth; // thread group width
    std::unique_ptr<CMetalCommandBuffer> commandBuffer; // the buffer to execute the current kernel
    id<MTLComputeCommandEncoder> computeEncoder; // the data encoder for the current kernel
};

//-----------------------------------------------------------------------------------------------------------------
  
// Starts the 1D kernel (the executing thread group is 1D)
class C1DKernel : public CMetalKernel {
public:
    C1DKernel( CMetalCommandQueue& queue, const char* name, int widthCombine, int width );
};

//-----------------------------------------------------------------------------------------------------------------
  
// Starts the 2D kernel (the executing thread group is 2D)
class C2DKernel : public CMetalKernel {
public:
    C2DKernel( CMetalCommandQueue& queue, const char* name, int heightCombine, int widthCombine,
		int height, int width );
};

//-----------------------------------------------------------------------------------------------------------------

// Starts the 3D kernel (the executing thread group is 3D)
class C3DKernel : public CMetalKernel {
public:
    C3DKernel( CMetalCommandQueue& queue, const char* name,
		int depthCombine, int heightCombine, int widthCombine, int depth, int height, int width );
};

} // namespace NeoML

#endif // NEOML_USE_METAL
