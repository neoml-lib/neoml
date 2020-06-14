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

inline void Reduce1DSum( thread uint thread_position_in_threadgroup, thread uint threads_per_threadgroup, threadgroup float* buffer )
{
    thread uint s = 1;
    while( s * 2 < threads_per_threadgroup ) {
        s = s * 2;
    }
    
    int bufferIndex = thread_position_in_threadgroup;
    
    for( uint i = s; i >= 1; i = i >> 1 ) {
        if( thread_position_in_threadgroup < i && thread_position_in_threadgroup + i < threads_per_threadgroup ) {
            buffer[bufferIndex] += buffer[bufferIndex + i];
        }
    }
}

inline void Reduce2DMax( thread uint2 thread_position_in_threadgroup, thread uint2 threads_per_threadgroup, threadgroup float* buffer )
{
    uint s = 1;
    while( s * 2 < threads_per_threadgroup.x ) {
        s = s * 2;
    }
    
    int bufferIndex = thread_position_in_threadgroup.y * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
    
    for( uint i = s; i >= 1; i = i >> 1 ) {
        if( thread_position_in_threadgroup.x < i && thread_position_in_threadgroup.x + i < threads_per_threadgroup.x ) {
            if( buffer[bufferIndex] <= buffer[bufferIndex + i] ) {
                buffer[bufferIndex] = buffer[bufferIndex + i];
            }
        }
    }
}

inline void Reduce2DMax( thread uint2 thread_position_in_threadgroup, thread uint2 threads_per_threadgroup, threadgroup float* buffer, threadgroup int* indexes )
{
    thread uint s = 1;
    while( s * 2 < threads_per_threadgroup.x ) {
        s = s * 2;
    }
    
    int bufferIndex = thread_position_in_threadgroup.y * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
    
    for( uint i = s; i >= 1; i = i >> 1 ) {
        if( thread_position_in_threadgroup.x < i && thread_position_in_threadgroup.x + i < threads_per_threadgroup.x ) {
            if( buffer[bufferIndex] <= buffer[bufferIndex + i] ) {
                buffer[bufferIndex] = buffer[bufferIndex + i];
                indexes[bufferIndex] = indexes[bufferIndex + i];
            }
        }
    }
}

inline void Reduce2DSum( thread uint2 thread_position_in_threadgroup, thread uint2 threads_per_threadgroup, threadgroup float* buffer )
{
    thread uint s = 1;
    while( s * 2 < threads_per_threadgroup.x ) {
        s = s * 2;
    }
    
    int bufferIndex = thread_position_in_threadgroup.y * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
    
    for( uint i = s; i >= 1; i = i >> 1 ) {
        if( thread_position_in_threadgroup.x < i && thread_position_in_threadgroup.x + i < threads_per_threadgroup.x ) {
            buffer[bufferIndex] += buffer[bufferIndex + i];
        }
    }
}

inline void Reduce2DSum( thread uint3 thread_position_in_threadgroup, thread uint3 threads_per_threadgroup, threadgroup float* buffer )
{
    thread uint s = 1;
    while( s * 2 < threads_per_threadgroup.x ) {
        s = s * 2;
    }
    
    int bufferIndex = (thread_position_in_threadgroup.z * threads_per_threadgroup.y + thread_position_in_threadgroup.y) * threads_per_threadgroup.x
        + thread_position_in_threadgroup.x;
    
    for( uint i = s; i >= 1; i = i >> 1 ) {
        if( thread_position_in_threadgroup.x < i && thread_position_in_threadgroup.x + i < threads_per_threadgroup.x ) {
            buffer[bufferIndex] += buffer[bufferIndex + i];
        }
    }
}

inline void Reduce2DSumTrans( thread uint2 thread_position_in_threadgroup, thread uint2 threads_per_threadgroup, threadgroup float* buffer )
{
    thread uint s = 1;
    while( s * 2 < threads_per_threadgroup.y ) {
        s = s * 2;
    }
    
    int bufferIndex = thread_position_in_threadgroup.y * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
    
    for( uint i = s; i >= 1; i = i >> 1 ) {
        if( thread_position_in_threadgroup.y < i && thread_position_in_threadgroup.y + i < threads_per_threadgroup.y ) {
            buffer[bufferIndex] += buffer[bufferIndex + i * threads_per_threadgroup.x];
        }
    }
}

inline void Reduce2DMaxTrans( thread uint2 thread_position_in_threadgroup, thread uint2 threads_per_threadgroup, threadgroup float* buffer )
{
    uint s = 1;
    while( s * 2 < threads_per_threadgroup.y ) {
        s = s * 2;
    }
    
    int bufferIndex = thread_position_in_threadgroup.y * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
    
    for( uint i = s; i >= 1; i = i >> 1 ) {
        if( thread_position_in_threadgroup.y < i && thread_position_in_threadgroup.y + i < threads_per_threadgroup.y ) {
            if( buffer[bufferIndex] <= buffer[bufferIndex + i * threads_per_threadgroup.x] ) {
                buffer[bufferIndex] = buffer[bufferIndex + i * threads_per_threadgroup.x];
            }
        }
    }
}
