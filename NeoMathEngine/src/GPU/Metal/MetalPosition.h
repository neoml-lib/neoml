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

#include <metal_stdlib>

using namespace metal;

// The position in a 3D kernel
class C3DPosition {
public:
    explicit C3DPosition( uint3 _gridIndex ) :
        gridIndex( _gridIndex )
    {
    }
    
    inline bool GetMetalTaskIndex3D( int depth, int height, int width, thread int& dIndex, thread int& hIndex, thread int& wIndex )
    {
        wIndex = gridIndex.x;
        hIndex = gridIndex.y;
        dIndex = gridIndex.z;
        return ( dIndex < depth && wIndex < width && hIndex < height );
    }
    
private:
    uint3 gridIndex;
};

//---------------------------------------------------------------------------------------------------------------------

// The position in a 3D kernel with combine
class C3DCombinePosition {
public:
    C3DCombinePosition( uint3 _threadIndex, uint3 _blockSize, uint3 _blockIndex ) :
        threadIndex( _threadIndex ),
        blockSize( _blockSize ),
        blockIndex( _blockIndex )
    {
    }
    
    inline int GetMetalHeightTaskCountAndIndex( int taskCount, int combineCount, thread int& index, thread int& step )
    {
        index = blockIndex.y * combineCount * blockSize.y + threadIndex.y;
        step = blockSize.y;
        
        return min( combineCount, (taskCount - index + step - 1) / step );
    }
    
    inline int GetMetalWidthTaskCountAndIndex( int taskCount, int combineCount, thread int& index, thread int& step )
    {
        index = blockIndex.x * combineCount * blockSize.x + threadIndex.x;
        step = blockSize.x;
        
        return min( combineCount, (taskCount - index + step - 1) / step );
    }
    
    inline bool GetMetalTaskIndex3D( int depth, int height, int width, int dCombine, int hCombine, int wCombine,
        thread int& dIndex, thread int& hIndex, thread int& wIndex )
    {
        wIndex = blockIndex.x * blockSize.x * wCombine + threadIndex.x;
        hIndex = blockIndex.y * blockSize.y * hCombine + threadIndex.y;
        dIndex = blockIndex.z * blockSize.z * dCombine + threadIndex.z;
        return ( wIndex < width && hIndex < height && dIndex < depth );
    }
    
private:
    uint3 threadIndex;
    uint3 blockSize;
    uint3 blockIndex;
};

//---------------------------------------------------------------------------------------------------------------------

// The position in a 2D kernel
class C2DPosition {
public:
    explicit C2DPosition( uint2 _gridIndex ) :
        gridIndex( _gridIndex )
    {
    }
    
    inline bool GetMetalTaskIndex2D( int height, int width, thread int& hIndex, thread int& wIndex )
    {
        wIndex = gridIndex.x;
        hIndex = gridIndex.y;
        return ( wIndex < width && hIndex < height );
    }
    
private:
    uint2 gridIndex;
};

//---------------------------------------------------------------------------------------------------------------------

// The position in a 2D kernel with combine
class C2DCombinePosition {
public:
    C2DCombinePosition( uint2 _threadIndex, uint2 _blockSize, uint2 _blockIndex ) :
        threadIndex( _threadIndex ),
        blockSize( _blockSize ),
        blockIndex( _blockIndex )
    {
    }
    
    inline int GetMetalHeightTaskCountAndIndex( int taskCount, int combineCount, thread int& index, thread int& step )
    {
        index = blockIndex.y * combineCount * blockSize.y + threadIndex.y;
        step = blockSize.y;
        
        return min( combineCount, (taskCount - index + step - 1) / step );
    }
    
    inline int GetMetalWidthTaskCountAndIndex( int taskCount, int combineCount, thread int& index, thread int& step )
    {
        index = blockIndex.x * combineCount * blockSize.x + threadIndex.x;
        step = blockSize.x;
        
        return min( combineCount, (taskCount - index + step - 1) / step );
    }
    
    inline bool GetMetalTaskIndex2D( int height, int width, int hCombine, int wCombine, thread int& hIndex, thread int& wIndex )
    {
        wIndex = blockIndex.x * blockSize.x * wCombine + threadIndex.x;
        hIndex = blockIndex.y * blockSize.y * hCombine + threadIndex.y;
        return ( wIndex < width && hIndex < height );
    }
    
private:
    uint2 threadIndex;
    uint2 blockSize;
    uint2 blockIndex;
};

//---------------------------------------------------------------------------------------------------------------------

// The position in a 1D kernel
class C1DPosition {
public:
    explicit C1DPosition( uint _gridIndex ) :
        gridIndex( _gridIndex )
    {
    }
    
    inline bool GetMetalTaskIndex( int width, thread int& wIndex )
    {
        if( (int)gridIndex < width ) {
            wIndex = gridIndex;
            return true;
        }
        return false;
    }
    
private:
    uint gridIndex;
};

//---------------------------------------------------------------------------------------------------------------------

// The position in a 1D kernel with combine
class C1DCombinePosition {
public:
    C1DCombinePosition( uint _threadIndex, uint _blockSize, uint _blockIndex ) :
        threadIndex( _threadIndex ),
        blockSize( _blockSize ),
        blockIndex( _blockIndex )
    {
    }
    
    inline int GetMetalTaskCountAndIndex( int taskCount, int combineCount, thread int& index, thread int& step )
    {
        index = blockIndex * combineCount * blockSize + threadIndex;
        step = blockSize;
        
        return min( combineCount, (taskCount - index + step - 1) / step );
    }
    
private:
    uint threadIndex;
    uint blockSize;
    uint blockIndex;
};
