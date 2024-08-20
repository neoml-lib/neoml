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

#include "MetalCommon.h"
#include "MetalPosition.h"
#include "MetalRandomGenerator.h"
#include "MetalReduce.h"

static constant int VectorCombineCount = 8;
static constant int MatrixCombineCount = 4;

kernel void vectorKernelFillFloat( device float* first [[buffer(0)]],
                                   constant float* value [[buffer(1)]],
                                   constant int* vectorSize [[buffer(2)]],
                                   uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                   uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                   uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *vectorSize, VectorCombineCount, index, step );
    
    first += index;
    
    for( int i = 0; i < actionCount; ++i ) {
        *first = *value;
        first += step;
    }
}
    
kernel void vectorKernelFillInt( device int* first [[buffer(0)]],
                                 constant int* value [[buffer(1)]],
                                 constant int* vectorSize [[buffer(2)]],
                                 uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                 uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                 uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *vectorSize, VectorCombineCount, index, step );
    
    first += index;
    
    for( int i = 0; i < actionCount; ++i ) {
        *first = *value;
        first += step;
    }
}

kernel void vectorKernelConvertFloatToInt( constant float* from [[buffer(0)]],
                                           device int* to [[buffer(1)]],
                                           constant int* vectorSize [[buffer(2)]],
                                           uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                           uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                           uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *vectorSize, VectorCombineCount, index, step );

    from += index;
    to += index;

    for( int i = 0; i < actionCount; ++i ) {
        *to = static_cast<int>( *from );
        from += step;
        to += step;
    }
}

kernel void vectorKernelConvertIntToFloat( constant int* from [[buffer(0)]],
                                           device float* to [[buffer(1)]],
                                           constant int* vectorSize [[buffer(2)]],
                                           uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                           uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                           uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *vectorSize, VectorCombineCount, index, step );

    from += index;
    to += index;

    for( int i = 0; i < actionCount; ++i ) {
        *to = static_cast<float>( *from );
        from += step;
        to += step;
    }
}
    
kernel void vectorKernelVectorFillBernoulli( device float* result [[buffer(0)]],
                                             constant float& p [[buffer(1)]],
                                             constant int& vectorSize [[buffer(2)]],
                                             constant float& value [[buffer(3)]],
                                             constant int& randomInit [[buffer(4)]],
                                             uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                             uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                             uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( ( vectorSize + 3 ) / 4, VectorCombineCount, index, step );
    
    if( actionCount > 0 ) {
        CMathEngineRandom random( randomInit );
        random.Skip( index );

        index *= 4;
        result += index;

        const unsigned int threshold = p * 0xFFFFFFFF;

        for( int i = 0; i < actionCount; ++i ) {
            CIntArray<4> generated = random.Next();
            for( int j = 0; j < 4 && index + j < vectorSize; ++j ) {
                result[j] = generated[j] <= threshold ? value : 0;
            }
            result += step * 4;
            index += step * 4;
            random.Skip( step - 1 );
        }
    }
}

kernel void vectorKernelCopy( device float* first [[buffer(0)]],
                              constant float* second [[buffer(1)]],
                              constant int* size [[buffer(2)]],
                              uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                              uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                              uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *size, VectorCombineCount, index, step );
    
    for( int i = 0; i < actionCount; ++i ) {
        first[index] = second[index];
        index += step;
    }
}
    
kernel void vectorFilterSmallValues( device float* first [[buffer(0)]],
                                     constant int* firstSize [[buffer(1)]],
                                     constant float* threshold [[buffer(2)]],
                                     uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                     uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                     uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *firstSize, VectorCombineCount, index, step );
    
    first += index;
    for( int i = 0; i < actionCount; ++i ) {
        if( *first < *threshold && *first > -*threshold ) {
            *first = 0;
        }
        first += step;
    }
}
  
kernel void vectorKernelAddVectorToMatrixElements( device float* matrix [[buffer(0)]],
                                                   constant int* height [[buffer(1)]],
                                                   constant int* width [[buffer(2)]],
                                                   constant int* indices [[buffer(3)]],
                                                   constant float* vector [[buffer(4)]],
                                                   uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                                   uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                                   uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int jPos;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *height, VectorCombineCount, jPos, step );
    
    for( int i = 0; i < actionCount; ++i ) {
        int index = indices[jPos];
        if( index >= 0 && index < *width ) {
            matrix[jPos * *width + index] += vector[jPos];
        }
        jPos += step;
    }
}
    
kernel void matrixKernelAddVectorToMatrixElementsEx( device float* matrix [[buffer(0)]],
                                                     constant int* height [[buffer(1)]],
                                                     constant int* width [[buffer(2)]],
                                                     constant int* rowIndices [[buffer(3)]],
                                                     constant int* columnIndices [[buffer(4)]],
                                                     constant float* vector [[buffer(5)]],
                                                     constant int* vectorSize [[buffer(6)]],
                                                     uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    int hIndex;
    int wIndex;
    if( pos.GetMetalTaskIndex2D( ( *height + 7 ) / 8, ( *width + 7 ) / 8, hIndex, wIndex ) ) {
        for( int i = 0; i < *vectorSize; i++ ) {
            int rIndex = rowIndices[i];
            int cIndex = columnIndices[i];
            if( hIndex * 8 <= rIndex && rIndex <= hIndex * 8 + 7 &&
                wIndex * 8 <= cIndex && cIndex <= wIndex * 8 + 7 )
            {
                matrix[rIndex * *width + cIndex] += vector[i];
            }
        }
    }
}
    
kernel void vectorKernelAddMatrixElementsToMatrix( constant float* matrix [[buffer(0)]],
                                                   constant int* height [[buffer(1)]],
                                                   constant int* width [[buffer(2)]],
                                                   device float* result [[buffer(3)]],
                                                   constant int* indices [[buffer(4)]],
                                                   uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                                   uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                                   uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
     C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
     int jPos;
     int step;
     int actionCount = pos.GetMetalTaskCountAndIndex( *height, VectorCombineCount, jPos, step );
    
     for( int i = 0; i < actionCount; ++i ) {
         int index = indices[jPos];
         if( index >= 0 && index < *width ) {
             result[jPos * *width + index] += matrix[jPos * *width + index];
         }
         jPos += step;
     }
}
    
kernel void vectorKernelAddMatrixElementsToVector( constant float* matrix [[buffer(0)]],
                                                   constant int* height [[buffer(1)]],
                                                   constant int* width [[buffer(2)]],
                                                   constant int* indices [[buffer(3)]],
                                                   device float* result [[buffer(4)]],
                                                   uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                                   uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                                   uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int jPos;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *height, VectorCombineCount, jPos, step );
    
    for( int i = 0; i < actionCount; ++i ) {
        int index = indices[jPos];
        if( index >= 0 && index < *width ) {
            result[jPos] += matrix[jPos * *width + index];
        }
        jPos += step;
    }
}
    
kernel void vectorKernelAddMatrixElementsToVectorEx( constant float* matrix [[buffer(0)]],
                                                     constant int* height [[buffer(1)]],
                                                     constant int* width [[buffer(2)]],
                                                     constant int* rowIndices [[buffer(3)]],
                                                     constant int* columnIndices [[buffer(4)]],
                                                     device float* result [[buffer(5)]],
                                                     constant int* vectorSize [[buffer(6)]],
                                                     uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                                     uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                                     uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *vectorSize, VectorCombineCount, index, step );
    
    for( int i = 0; i < actionCount; ++i ) {
        result[index] += matrix[rowIndices[index] * *width + columnIndices[index]];
        index += step;
    }
}
    
kernel void vectorKernelSum( constant const float* first [[buffer(0)]],
                             constant const int* firstSize [[buffer(1)]],
                             constant const int* isNegative [[buffer(2)]],
                             device float* result [[buffer(3)]],
                             threadgroup float* buffer [[threadgroup(4)]],
                             uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                             uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                             uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    const int combineCount = (*firstSize + threads_per_threadgroup - 1) / threads_per_threadgroup;
    int bufferIndex = thread_position_in_threadgroup;
    buffer[bufferIndex] = 0;
    
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *firstSize, combineCount, index, step );
    
    first += index;
    for( int i = 0; i < actionCount; ++i ) {
        buffer[bufferIndex] += *first;
        first += step;
    }
    
    Reduce1DSum( thread_position_in_threadgroup, threads_per_threadgroup, buffer );
    
    if( thread_position_in_threadgroup == 0 ) {
        *result += ( *isNegative != 0 ? -buffer[bufferIndex] : buffer[bufferIndex] );
    }
}

kernel void vectorKernelEqual( constant const int* first [[buffer(0)]],
                               constant const int* second [[buffer(1)]],
                               constant const int* vectorSize [[buffer(2)]],
                               device float* result [[buffer(3)]],
                               uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                               uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                               uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *vectorSize, VectorCombineCount, index, step );
    
    first += index;
    second += index;
    result += index;
    for( int action = 0; action < actionCount; ++action ) {
        *result = *first == *second ? 1.0 : 0.0;
        first += step;
        second += step;
        result += step;
    }
}

kernel void vectorKernelEqualValue( constant const int* first [[buffer(0)]],
                                    constant const int& value [[buffer(1)]],
                                    constant const int* vectorSize [[buffer(2)]],
                                    device float* result [[buffer(3)]],
                                    uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                    uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                    uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *vectorSize, VectorCombineCount, index, step );
    
    first += index;
    result += index;
    for( int action = 0; action < actionCount; ++action ) {
        *result = *first == value ? 1.0 : 0.0;
        first += step;
        result += step;
    }
}
    
kernel void vectorKernelELU( constant const float* first [[buffer(0)]],
                             device float* result [[buffer(1)]],
                             constant const int* vectorSize [[buffer(2)]],
                             constant const float* alpha [[buffer(3)]],
                             uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                             uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                             uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *vectorSize, VectorCombineCount, index, step );
    
    first += index;
    result += index;
    for( int action = 0; action < actionCount; ++action ) {
        *result = *first >= 0 ? *first : *alpha * ( ExponentFunc( *first ) - 1. );
        first += step;
        result += step;
    }
}

kernel void vectorKernelELUDiff( constant float* first [[buffer(0)]],
                                 constant float* second [[buffer(1)]],
                                 device float* result [[buffer(2)]],
                                 constant int* count [[buffer(3)]],
                                 constant float* alpha [[buffer(4)]],
                                 uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                 uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                 uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    second += index;
    result += index;
    
    for( int i = 0; i < actionCount; ++i ) {
        *result = *first >= 0 ? *second : *second * ExponentFunc( *first ) * *alpha;
        first += step;
        second += step;
        result += step;
    }
}
    
kernel void vectorKernelELUDiffOp( constant float* first [[buffer(0)]],
                                   constant float*  second [[buffer(1)]],
                                   device float* result [[buffer(2)]],
                                   constant int* count [[buffer(3)]],
                                   constant float* alpha [[buffer(4)]],
                                   uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                   uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                   uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    second += index;
    result += index;
    
    for( int i = 0; i < actionCount; ++i ) {
        *result = *first >= 0 ? *second : *second * ( *first + *alpha );
        first += step;
        second += step;
        result += step;
    }
}

kernel void vectorKernelReLUFloat4( constant float4* first [[buffer(0)]],
                                    device float4* result [[buffer(1)]],
                                    constant int& count [[buffer(2)]],
                                    constant float& threshold [[buffer(3)]],
                                    uint thread_position_in_grid [[ thread_position_in_grid ]] )
{
    if( (int)thread_position_in_grid >= count ) {
        return;
    }
    const int index = thread_position_in_grid * 4;
    first += index;

    float4 val1 = first[0];
    float4 val2 = first[1];
    float4 val3 = first[2];
    float4 val4 = first[3];

    if( threshold > 0 ) {
        val1 = min(val1, float4(threshold));
        val2 = min(val2, float4(threshold));
        val3 = min(val3, float4(threshold));
        val4 = min(val4, float4(threshold));
    }

    float4 res1 = max(val1, float4(0));
    float4 res2 = max(val2, float4(0));
    float4 res3 = max(val3, float4(0));
    float4 res4 = max(val4, float4(0));

    result += index;
    result[0] = res1;
    result[1] = res2;
    result[2] = res3;
    result[3] = res4;
}

kernel void vectorKernelReLUFloat( constant float* first [[buffer(0)]],
                              device float* result [[buffer(1)]],
                              constant int& vectorSize [[buffer(2)]],
                              constant float& threshold [[buffer(3)]],
                              uint thread_position_in_grid [[ thread_position_in_grid ]] )
{
    int index = thread_position_in_grid;
    
    if( index >= vectorSize )
        return;

    float value = first[index];
    if( threshold > 0 )
    {
        value = min(value, threshold);
    }

    result[index] = max(value, 0.0);
}

kernel void vectorKernelReLUDiff( constant float* first [[buffer(0)]],
                                  constant float* second [[buffer(1)]],
                                  device float* result [[buffer(2)]],
                                  constant int* count [[buffer(3)]],
                                  constant float* threshold [[buffer(4)]],
                                  uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                  uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                  uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );

    first += index;
    second += index;
    result += index;
    
    if( *threshold > 0 ) {
        for(int i = 0; i < actionCount; ++i) {
            *result = (*first > 0 && *first < *threshold) ? *second : 0;
            first += step;
            second += step;
            result += step;
        }
    } else {
        for(int i = 0; i < actionCount; ++i) {
            *result = *first > 0 ? *second : 0;
            first += step;
            second += step;
            result += step;
        }
    }
}

kernel void vectorKernelLeakyReLU( constant float* first [[buffer(0)]],
                                   device float* result [[buffer(1)]],
                                   constant int* count [[buffer(2)]],
                                   constant float*  alpha [[buffer(3)]],
                                   uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                   uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                   uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    result += index;
    for( int i = 0; i < actionCount; ++i ) {
        float value = *first;
        *result = value > 0 ? value : *alpha * value;
        first += step;
        result += step;
    }
}

kernel void vectorKernelLeakyReLUDiff( constant float* first [[buffer(0)]],
                                       constant float* second [[buffer(1)]],
                                       device float* result [[buffer(2)]],
                                       constant int* count [[buffer(3)]],
                                       constant float* alpha [[buffer(4)]],
                                       uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                       uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                       uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    second += index;
    result += index;
    
    for( int i = 0; i < actionCount; ++i ) {
        *result = *first > 0 ? *second : *second * *alpha;
        first += step;
        second += step;
        result += step;
    }
}

kernel void vectorKernelHSwish( constant float* first [[buffer(0)]],
                                device float* result [[buffer(1)]],
                                constant int* count [[buffer(2)]],
                                uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    result += index;
    
    for( int i = 0; i < actionCount; ++i ) {
		float val = *first;
		if( val <= -3.f ) {
			*result = 0;
		} else if( val >= 3.f ) {
			*result = val;
		} else {
			*result = val * ( val + 3.f ) / 6.f; 
		}
		first += step;
		result += step;
    }
}

kernel void vectorKernelHSwishDiff( constant float* first [[buffer(0)]],
                                    constant float* second [[buffer(1)]],
                                    device float* result [[buffer(2)]],
                                    constant int* count [[buffer(3)]],
                                    uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                    uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                    uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    second += index;
    result += index;
    
    for( int i = 0; i < actionCount; ++i ) {
		float val = *first;
		if( val <= -3.f ) {
			*result = 0;
		} else if( val >= 3.f ) {
			*result = *second;
		} else {
			*result = ( val / 3.f + 0.5f ) * *second; 
		}
		first += step;
		second += step;
		result += step;
    }
}
                                                                   
kernel void vectorKernelEltwiseMax( constant float* first [[buffer(0)]],
                                    constant float* second [[buffer(1)]],
                                    device float* result [[buffer(2)]],
                                    constant int* count [[buffer(3)]],
                                    uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                    uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                    uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );

    first += index;
    second += index;
    result += index;

    for(int i = 0; i < actionCount; ++i) {
        float value1 = *first;
        float value2 = *second;
        *result = value1 > value2 ? value1 : value2;
        first += step;
        second += step;
        result += step;
    }
}

kernel void vectorKernelEltwiseMin( constant float* first [[buffer(0)]],
                                    constant float* second [[buffer(1)]],
                                    device float* result [[buffer(2)]],
                                    constant int* count [[buffer(3)]],
                                    uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                    uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                    uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );

    first += index;
    second += index;
    result += index;

    for(int i = 0; i < actionCount; ++i) {
        float value1 = *first;
        float value2 = *second;
        *result = value1 < value2 ? value1 : value2;
        first += step;
        second += step;
        result += step;
    }
}
                                                                   
kernel void vectorKernelAbs( constant float* first [[buffer(0)]],
                             device float* result [[buffer(1)]],
                             constant int* count [[buffer(2)]],
                             uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                             uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                             uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );

    first += index;
    result += index;

    for(int i = 0; i < actionCount; ++i) {
        float value = *first;
        *result = value > 0 ? value : -value;
        first += step;
        result += step;
    }
}

kernel void vectorKernelAbsDiff( constant float* first [[buffer(0)]],
                                 constant float* second [[buffer(1)]],
                                 device float* result [[buffer(2)]],
                                 constant int* count [[buffer(3)]],
                                 uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                 uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                 uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );

    first += index;
    second += index;
    result += index;

    for(int i = 0; i < actionCount; ++i) {
        *result = *first > 0 ? *second : -*second;
        first += step;
        second += step;
        result += step;
    }
}
    
kernel void vectorKernelHinge( constant float* first [[buffer(0)]],
                               device float* result [[buffer(1)]],
                               constant int* count [[buffer(2)]],
                               uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                               uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                               uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    result += index;
    
    for(int i = 0; i < actionCount; ++i) {
        float value = 1 - *first;
        *result = value > 0 ? value : 0;
        first += step;
        result += step;
    }
}

kernel void vectorKernelHingeDiff( constant float* first [[buffer(0)]],
                                   constant float* second [[buffer(1)]],
                                   device float* result [[buffer(2)]],
                                   constant int* count [[buffer(3)]],
                                   uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                   uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                   uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    second += index;
    result += index;
    
    for( int i = 0; i < actionCount; ++i ) {
        *result = *first < 1 ? -*second : 0;
        first += step;
        second += step;
        result += step;
    }
}

kernel void vectorKernelSquaredHinge( constant float* first [[buffer(0)]],
                                      device float* result [[buffer(1)]],
                                      constant int* count [[buffer(2)]],
                                      uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                      uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                      uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    result += index;
    
    for( int i = 0; i < actionCount; ++i ) {
        float value = *first;
        if(value < -1) {
            *result = -4 * value;
        } else {
            value = 1 - value;
            *result = value < 0 ? 0 : value * value;
        }
        first += step;
        result += step;
    }
}

kernel void vectorKernelSquaredHingeDiff( constant float* first [[buffer(0)]],
                                          constant float* second [[buffer(1)]],
                                          device float* result [[buffer(2)]],
                                          constant int* count [[buffer(3)]],
                                          uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                          uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                          uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    second += index;
    result += index;
    
    for(int i = 0; i < actionCount; ++i) {
        float value = *first;
        if(value < -1) {
            *result = -4 * (*second);
        } else {
            value = 1 - value;
            *result = value < 0 ? 0 : -2 * value * (*second);
        }
        first += step;
        second += step;
        result += step;
    }
}
    
kernel void vectorKernelHuber( constant float* first [[buffer(0)]],
                               device float* result [[buffer(1)]],
                               constant int* count [[buffer(2)]],
                               uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                               uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                               uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    result += index;
    
    for(int i = 0; i < actionCount; ++i) {
        if(*first < -1) {
            *result = -(*first) - 0.5f;
        } else if(*first > 1) {
            *result = *first - 0.5f;
        } else {
            *result = *first * (*first) * 0.5f;
        }
        first += step;
        result += step;
    }
}

kernel void vectorKernelHuberDiff( constant float* first [[buffer(0)]],
                                   device float* result [[buffer(1)]],
                                   constant int* count [[buffer(2)]],
                                   uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                   uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                   uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    result += index;
    
    for( int i = 0; i < actionCount; ++i ) {
        if(*first < -1) {
            *result = -1;
        } else if(*first > 1) {
            *result = 1;
        } else {
            *result = *first;
        }
        first += step;
        result += step;
    }
}
    
kernel void vectorKernelHardTanh( constant float* first [[buffer(0)]],
                                  device float* result [[buffer(1)]],
                                  constant int* count [[buffer(2)]],
                                  uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                  uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                  uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    result += index;
    
    for(int i = 0; i < actionCount; ++i) {
        float value = *first;
        if(value < -1) {
            *result = -1;
        } else if(value > 1) {
            *result = 1;
        } else {
            *result = value;
        }
        first += step;
        result += step;
    }
}

kernel void vectorKernelHardTanhDiff( constant float* first [[buffer(0)]],
                                      constant float* second [[buffer(1)]],
                                      device float* result [[buffer(2)]],
                                      constant int* count [[buffer(3)]],
                                      uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                      uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                      uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    second += index;
    result += index;
    
    for(int i = 0; i < actionCount; ++i) {
        float value = *first;
        if(value <= -1 || value >= 1) {
            *result = 0;
        } else {
            *result = *second;
        }
        first += step;
        second += step;
        result += step;
    }
}

kernel void vectorKernelHardSigmoid( constant float* first [[buffer(0)]],
                                     device float* result [[buffer(1)]],
                                     constant int* count [[buffer(2)]],
                                     constant float* slope [[buffer(3)]],
                                     constant float* bias [[buffer(4)]],
                                     uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                     uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                     uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    result += index;
    
    for( int i = 0; i < actionCount; ++i ) {
        float value = *first * *slope + *bias;
        if( value < 0 ) {
            *result = 0;
        } else if( value > 1 ) {
            *result = 1;
        } else {
            *result = value;
        }
        first += step;
        result += step;
    }
}

kernel void vectorKernelHardSigmoidDiff( constant float* first [[buffer(0)]],
                                         constant float* second [[buffer(1)]],
                                         device float* result [[buffer(2)]],
                                         constant int* count [[buffer(3)]],
                                         constant float* slope [[buffer(4)]],
                                         constant float* bias [[buffer(5)]],
                                         uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                         uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                         uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    second += index;
    result += index;
    
	float minX = -*bias / *slope;
	float maxX = ( 1.f - *bias ) / *slope;

    for( int i = 0; i < actionCount; ++i ) {
        float value = *first;
        if( value <= minX || value >= maxX ) {
            *result = 0;
        } else {
            *result = *second * *slope;
        }
        first += step;
        second += step;
        result += step;
    }
}

kernel void vectorKernelHardSigmoidDiffOp( constant float* first [[buffer(0)]],
                                           constant float* second [[buffer(1)]],
                                           device float* result [[buffer(2)]],
                                           constant int* count [[buffer(3)]],
                                           constant float* slope [[buffer(4)]],
                                           uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                           uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                           uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    second += index;
    result += index;
    
    for( int i = 0; i < actionCount; ++i ) {
        float value = *first;
        if( value <= 0 || value >= 1 ) {
            *result = 0;
        } else {
            *result = *second * *slope;
        }
        first += step;
        second += step;
        result += step;
    }
}

kernel void vectorKernelExp( constant float* first [[buffer(0)]],
                             device float* result [[buffer(1)]],
                             constant int* count [[buffer(2)]],
                             uint thread_position_in_grid   [[ thread_position_in_grid ]])
{
    C1DPosition pos( thread_position_in_grid );
    int index;
    if( pos.GetMetalTaskIndex( *count, index ) ) {
        result[index] = ExponentFunc(first[index]);
    }
}

kernel void vectorKernelLog( constant float* first [[buffer(0)]],
                             device float* result [[buffer(1)]],
                             constant int* count [[buffer(2)]],
                             uint thread_position_in_grid   [[ thread_position_in_grid ]] )
{
    C1DPosition pos( thread_position_in_grid );
    int index;
    if( pos.GetMetalTaskIndex( *count, index ) ) {
        result[index] = LogFunc( first[index] );
    }
}

kernel void vectorKernelNegLog( constant float* first [[buffer(0)]],
                                device float* result [[buffer(1)]],
                                constant int* count [[buffer(2)]],
                                uint thread_position_in_grid   [[ thread_position_in_grid ]] )
{
    C1DPosition pos( thread_position_in_grid );
    int index;
    if( pos.GetMetalTaskIndex( *count, index ) ) {
        result[index] = -LogFunc( first[index] );
    }
}

kernel void vectorKernelBernulliKLDerivative( constant float* first [[buffer(0)]],
                                              device float* result [[buffer(1)]],
                                              constant int* count [[buffer(2)]],
                                              constant float* target [[buffer(3)]],
                                              uint thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C1DPosition pos( thread_position_in_grid );
    int index;
    if( pos.GetMetalTaskIndex( *count, index ) ) {
        float value = first[index];
        float klDer = -*target / value + (1 - *target) / (1 - value);
        if(klDer < -10) {
            klDer = -10;
        } else if(klDer > 10) {
            klDer = 10;
        }
        result[index] = klDer;
    }
}

kernel void vectorKernelAddFloat( constant float* first [[buffer(0)]],
                                  constant float* second [[buffer(1)]],
                                  device float* result [[buffer(2)]],
                                  constant int* count [[buffer(3)]],
                                  uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                  uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                  uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    second += index;
    result += index;
    
    for(int i = 0; i < actionCount; ++i) {
        *result = *first + *second;
        first += step;
        second += step;
        result += step;
    }
}

kernel void vectorKernelAddInt( constant int* first [[buffer(0)]],
                                constant int* second [[buffer(1)]],
                                device int* result [[buffer(2)]],
                                constant int* count [[buffer(3)]],
                                uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    second += index;
    result += index;
    
    for(int i = 0; i < actionCount; ++i) {
        *result = *first + *second;
        first += step;
        second += step;
        result += step;
    }
}

kernel void vectorKernelAddValueFloat( constant float* first [[buffer(0)]],
                                       device float* result [[buffer(1)]],
                                       constant int* count [[buffer(2)]],
                                       constant float* addition [[buffer(3)]],
                                       uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                       uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                       uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    result += index;
    
    for(int i = 0; i < actionCount; ++i) {
        *result = *first + *addition;
        first += step;
        result += step;
    }
}

kernel void vectorKernelAddValueInt( constant int* first [[buffer(0)]],
                                     device int* result [[buffer(1)]],
                                     constant int* count [[buffer(2)]],
                                     constant int* addition [[buffer(3)]],
                                     uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                     uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                     uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    result += index;
    
    for(int i = 0; i < actionCount; ++i) {
        *result = *first + *addition;
        first += step;
        result += step;
    }
}

kernel void vectorKernelSubInt( constant int* first [[buffer(0)]],
                                constant int* second [[buffer(1)]],
                                device int* result [[buffer(2)]],
                                constant int* count [[buffer(3)]],
                                uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    second += index;
    result += index;
    
    for(int i = 0; i < actionCount; ++i) {
        *result = *first - *second;
        first += step;
        second += step;
        result += step;
    }
}

kernel void vectorKernelSubFloat( constant float* first [[buffer(0)]],
                                  constant float* second [[buffer(1)]],
                                  device float* result [[buffer(2)]],
                                  constant int* count [[buffer(3)]],
                                  uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                  uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                  uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    second += index;
    result += index;
    
    for(int i = 0; i < actionCount; ++i) {
        *result = *first - *second;
        first += step;
        second += step;
        result += step;
    }
}

kernel void vectorKernelMultiply( constant float* first [[buffer(0)]],
                                  device float* result [[buffer(1)]],
                                  constant int* count [[buffer(2)]],
                                  constant float* multiplier [[buffer(3)]],
                                  uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                  uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                  uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    result += index;
    
    for(int i = 0; i < actionCount; ++i) {
        *result = *first * (*multiplier);
        first += step;
        result += step;
    }
}

kernel void vectorKernelNegMultiply( constant float* first [[buffer(0)]],
                                     device float* result [[buffer(1)]],
                                     constant int* count [[buffer(2)]],
                                     constant float* multiplier [[buffer(3)]],
                                     uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                     uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                     uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    result += index;
    
    const float mul = -(*multiplier);
    
    for( int i = 0; i < actionCount; ++i ) {
        *result = *first * mul;
        first += step;
        result += step;
    }
}

kernel void vectorKernelEltwiseMultiplyInt( constant int* first [[buffer(0)]],
                                            constant int* second [[buffer(1)]],
                                            device int* result [[buffer(2)]],
                                            constant int* count [[buffer(3)]],
                                            uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                            uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                            uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    second += index;
    result += index;
    
    for( int i = 0; i < actionCount; ++i ) {
        *result = *first * *second;
        first += step;
        second += step;
        result += step;
    }
}

kernel void vectorKernelEltwiseMultiplyFloat( constant float* first [[buffer(0)]],
                                              constant float* second [[buffer(1)]],
                                              device float* result [[buffer(2)]],
                                              constant int* count [[buffer(3)]],
                                              uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                              uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                              uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    second += index;
    result += index;
    
    for( int i = 0; i < actionCount; ++i ) {
        *result = *first * *second;
        first += step;
        second += step;
        result += step;
    }
}

kernel void vectorKernelEltwiseMultiplyAdd( constant float* first [[buffer(0)]],
                                            constant float* second [[buffer(1)]],
                                            device float* result [[buffer(2)]],
                                            constant int* count [[buffer(3)]],
                                            uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                            uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                            uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    second += index;
    result += index;
    
    for(int i = 0; i < actionCount; ++i) {
        *result += *first * (*second);
        first += step;
        second += step;
        result += step;
    }
}

kernel void vectorKernelEltwiseNegMultiply( constant float* first [[buffer(0)]],
                                            constant float* second [[buffer(1)]],
                                            device float* result [[buffer(2)]],
                                            constant int* count [[buffer(3)]],
                                            uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                            uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                            uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    second += index;
    result += index;
    
    for( int i = 0; i < actionCount; ++i ) {
        *result = - *first * *second;
        first += step;
        second += step;
        result += step;
    }
}

kernel void vectorKernelEltwiseDivideInt( constant int* first [[buffer(0)]],
                                          constant int* second [[buffer(1)]],
                                          device int* result [[buffer(2)]],
                                          constant int* count [[buffer(3)]],
                                          uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                          uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                          uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );

    first += index;
    second += index;
    result += index;

    for( int i = 0; i < actionCount; ++i ) {
        *result = *first / (*second);
        first += step;
        second += step;
        result += step;
    }
}

kernel void vectorKernelEltwiseDivideFloat( constant float* first [[buffer(0)]],
                                            constant float* second [[buffer(1)]],
                                            device float* result [[buffer(2)]],
                                            constant int* count [[buffer(3)]],
                                            uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                            uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                            uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    second += index;
    result += index;
    
    for( int i = 0; i < actionCount; ++i ) {
        *result = *first / (*second);
        first += step;
        second += step;
        result += step;
    }
}

kernel void vectorKernelEltwisePower( constant float* first [[buffer(0)]],
                                      constant float* second [[buffer(1)]],
                                      device float* result [[buffer(2)]],
                                      constant int* count [[buffer(3)]],
                                      uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                      uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                      uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    second += index;
    result += index;

    for( int i = 0; i < actionCount; ++i ) {
        if( *second == 1 ) {
            *result = *first;
        } else if( *second == 0 ) {
            *result = 1;
        } else {
            *result = pow(*first, *second);
        }
        first += step;
        second += step;
        result += step;
    }
}

kernel void vectorKernelSqrt( constant float* first [[buffer(0)]],
                              device float* result [[buffer(1)]],
                              constant int* count [[buffer(2)]],
                              uint thread_position_in_grid [[ thread_position_in_grid ]])
{
    C1DPosition pos( thread_position_in_grid );
    int index;
    if( pos.GetMetalTaskIndex( *count, index ) ) {
        result[index] = sqrt(first[index]);
    }
}
    
kernel void vectorKernelInv( constant float* first [[buffer(0)]],
                             device float* result [[buffer(1)]],
                             constant int* count [[buffer(2)]],
                             uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                             uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                             uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    result += index;
    
    for( int i = 0; i < actionCount; ++i ) {
        if(-FLT_MIN <= *first && *first < 0) {
            *result = -FLT_MAX;
        } else if(0 <= *first && *first <= FLT_MIN) {
            *result = FLT_MAX;
        } else {
            *result = 1.f / (*first);
        }
        first += step;
        result += step;
    }
}

kernel void vectorKernelMinMax( constant float* first [[buffer(0)]],
                                device float* result [[buffer(1)]],
                                constant int* count [[buffer(2)]],
                                constant float* minValue [[buffer(3)]],
                                constant float* maxValue [[buffer(4)]],
                                uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *count, VectorCombineCount, index, step );
    
    first += index;
    result += index;
    
    for(int i = 0; i < actionCount; ++i) {
        *result = min(max(*first, *minValue), *maxValue);
        first += step;
        result += step;
    }
}

kernel void vectorKernelSigmoid( constant float* first [[buffer(0)]],
                                 device float* result [[buffer(1)]],
                                 constant int* count [[buffer(2)]],
                                 uint thread_position_in_grid [[ thread_position_in_grid ]])
{
    C1DPosition pos( thread_position_in_grid );
    int index;
    if( pos.GetMetalTaskIndex( *count, index ) ) {
        result[index] = 1.f / (1.f + ExponentFunc(-first[index]));
    }
}

kernel void vectorKernelSigmoidDiff( constant float* first [[buffer(0)]],
                                     constant float* second [[buffer(1)]],
                                     device float* result [[buffer(2)]],
                                     constant int* count [[buffer(3)]],
                                     uint thread_position_in_grid [[ thread_position_in_grid ]])
{
    C1DPosition pos( thread_position_in_grid );
    int index;
    if( pos.GetMetalTaskIndex( *count, index ) ) {
        float expVal = ExponentFunc(-first[index]);
        float expVal1 = expVal + 1.f;
        result[index] = expVal / expVal1 / expVal1;
        result[index] *= second[index];
    }
}

kernel void vectorKernelSigmoidDiffOp( constant float* first [[buffer(0)]],
                                       constant float* second [[buffer(1)]],
                                       device float* result [[buffer(2)]],
                                       constant int* vectorSize [[buffer(3)]],
                                       uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                       uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                       uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *vectorSize, VectorCombineCount, index, step );
    
    first += index;
    second += index;
    result += index;
    
    for( int i = 0; i < actionCount; ++i ) {
        *result = *first * (1.f - *first) * *second;
        first += step;
        second += step;
        result += step;
    }
}
    
kernel void vectorKernelTanh( constant float* first [[buffer(0)]],
                              device float* result [[buffer(1)]],
                              constant int* count [[buffer(2)]],
                              uint thread_position_in_grid [[ thread_position_in_grid ]])
{
    C1DPosition pos( thread_position_in_grid );
    int index;
    if( pos.GetMetalTaskIndex( *count, index ) ) {
        result[index] = -1.f  + 2 / (1.f + ExponentFunc(-2 * first[index]));
    }
}

kernel void vectorKernelTanhDiff( constant float* first [[buffer(0)]],
                                  constant float* second [[buffer(1)]],
                                  device float* result [[buffer(2)]],
                                  constant int* count [[buffer(3)]],
                                  uint thread_position_in_grid [[ thread_position_in_grid ]])
{
    C1DPosition pos( thread_position_in_grid );
    int index;
    if( pos.GetMetalTaskIndex( *count, index ) ) {
        float tanh = -1.f  + 2 / (1.f + ExponentFunc(-2 * first[index]));
        result[index] = second[index] * (1.f - tanh * tanh);
    }
}

kernel void vectorKernelTanhDiffOp( constant float* first [[buffer(0)]],
                                    constant float* second [[buffer(1)]],
                                    device float* result [[buffer(2)]],
                                    constant int* vectorSize [[buffer(3)]],
                                    uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                    uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                    uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *vectorSize, VectorCombineCount, index, step );
    
    first += index;
    second += index;
    result += index;
    
    for( int i = 0; i < actionCount; ++i ) {
        *result = (1.f - *first * *first) * *second;
        first += step;
        second += step;
        result += step;
    }
}
    
kernel void vectorKernelPower( constant float* exponent [[buffer(0)]],
                               constant float* first [[buffer(1)]],
                               device float* result [[buffer(2)]],
                               constant int* count [[buffer(3)]],
                               uint thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C1DPosition pos( thread_position_in_grid );
    int index;
    if( pos.GetMetalTaskIndex( *count, index ) ) {
        result[index] = pow(first[index], *exponent);
    }
}

kernel void vectorKernelPowerDiff( constant float* exponent [[buffer(0)]],
                                   constant float* first [[buffer(1)]],
                                   constant float*  second [[buffer(2)]],
                                   device float* result [[buffer(3)]],
                                   constant int* count [[buffer(4)]],
                                   uint thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C1DPosition pos( thread_position_in_grid );
    int index;
    if( pos.GetMetalTaskIndex( *count, index ) ) {
        result[index] = second[index] * *exponent * pow(first[index], *exponent - 1);
    }
}


kernel void vectorKernelPowerDiffOp( constant float* exponent [[buffer(0)]],
                                     constant float* first [[buffer(1)]],
                                     constant float* second [[buffer(2)]],
                                     device float* result [[buffer(3)]],
                                     constant int* count [[buffer(4)]],
                                     uint thread_position_in_grid [[ thread_position_in_grid ]])
{
    C1DPosition pos( thread_position_in_grid );
    int index;
    if( pos.GetMetalTaskIndex( *count, index ) ) {
        result[index] = second[index] * *exponent * pow(first[index], (*exponent - 1.f) / *exponent);
    }
}
    
kernel void vectorKernelL1DiffAdd( constant float* first [[buffer(0)]],
                                   constant float* second [[buffer(1)]],
                                   device float* result [[buffer(2)]],
                                   constant int* vectorSize [[buffer(3)]],
                                   constant float* threshold [[buffer(4)]],
                                   constant float* mult [[buffer(5)]],
                                   uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                   uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                   uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]])
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int actionCount = pos.GetMetalTaskCountAndIndex( *vectorSize, VectorCombineCount, index, step );
    
    first += index;
    second += index;
    result += index;
    
    float negThres = -*threshold;
    float thres = *threshold;
    float mulVal = *mult;
    
    for( int i = 0; i < actionCount; ++i ) {
        float x = *second;
        if(x < negThres) {
            x = negThres;
        } else if(x > thres) {
            x = thres;
        }
        
        *result = *first + mulVal * x;
        
        first += step;
        second += step;
        result += step;
    }
}

kernel void vectorKernelAddWidthIndexFloat( constant float* input [[buffer(0)]],
                                            constant int& width [[buffer(1)]],
                                            constant int& height [[buffer(2)]],
                                            constant int& channels [[buffer(3)]],
                                            constant int& objectCount [[buffer(4)]],
                                            constant bool& isForward [[buffer(5)]],
                                            device float* output [[buffer(6)]],
                                            uint thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C1DPosition pos( thread_position_in_grid );
    int index;
    if( pos.GetMetalTaskIndex( objectCount * channels * height * width, index ) ) {
        const int inputColumn = index % width;
        output[index] = input[index] + ( isForward ? inputColumn : -inputColumn );
    }
}

kernel void vectorKernelAddWidthIndexInt( constant int* input [[buffer(0)]],
                                          constant int& width [[buffer(1)]],
                                          constant int& height [[buffer(2)]],
                                          constant int& channels [[buffer(3)]],
                                          constant int& objectCount [[buffer(4)]],
                                          constant bool& isForward [[buffer(5)]],
                                          device int* output [[buffer(6)]],
                                          uint thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C1DPosition pos( thread_position_in_grid );
    int index;
    if( pos.GetMetalTaskIndex( objectCount * channels * height * width, index ) ) {
        const int inputColumn = index % width;
        output[index] = input[index] + ( isForward ? inputColumn : -inputColumn );
    }
}

kernel void vectorKernelAddHeightIndexFloat( constant float* input [[buffer(0)]],
                                             constant int& width [[buffer(1)]],
                                             constant int& height [[buffer(2)]],
                                             constant int& channels [[buffer(3)]],
                                             constant int& objectCount [[buffer(4)]],
                                             constant bool& isForward [[buffer(5)]],
                                             device float* output [[buffer(6)]],
                                             uint thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C1DPosition pos( thread_position_in_grid );
    int index;
    if( pos.GetMetalTaskIndex( objectCount * channels * height * width, index ) ) {
        const int inputRow = ( index / width) % height;
        output[index] = input[index] + ( isForward ? inputRow : -inputRow );
    }
}

kernel void vectorKernelAddHeightIndexInt( constant int* input [[buffer(0)]],
                                           constant int& width [[buffer(1)]],
                                           constant int& height [[buffer(2)]],
                                           constant int& channels [[buffer(3)]],
                                           constant int& objectCount [[buffer(4)]],
                                           constant bool& isForward [[buffer(5)]],
                                           device int* output [[buffer(6)]],
                                           uint thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C1DPosition pos( thread_position_in_grid );
    int index;
    if( pos.GetMetalTaskIndex( objectCount * channels * height * width, index ) ) {
        const int inputRow = ( index / width) % height;
        output[index] = input[index] + ( isForward ? inputRow : -inputRow );
    }
}
  
kernel void matrixKernelBatchVectorChannelLookupAndCopyFloatIndicesFloatData( constant int* batchSize [[buffer(0)]],
                                                                              constant const float* input [[buffer(1)]],
                                                                              constant int* inputChannels [[buffer(2)]],
                                                                              constant float* lookup [[buffer(3)]],
                                                                              constant int* vectorSize [[buffer(4)]],
                                                                              device float* output [[buffer(5)]],
                                                                              constant int* outputChannels [[buffer(6)]],
                                                                              constant int* batchNorm [[buffer(7)]],
                                                                              uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    int b;
    int index;
    if( pos.GetMetalTaskIndex2D( *batchNorm, *vectorSize, b, index ) ) {
        b *= MatrixCombineCount;
        int bLast = b + MatrixCombineCount;
        if( bLast > *batchSize ) {
            bLast = *batchSize;
        }
    
        int count = bLast - b;
    
        input += b * *inputChannels;
        output += b * *outputChannels + index;
        lookup += index;
        for( int k = 0; k < count; ++k ) {
            int tableIndex = (int)(*input);
            input += *inputChannels;
            *output = lookup[tableIndex * *vectorSize];
            output += *outputChannels;
        }
    }
}
    
kernel void matrixKernelBatchVectorChannelLookupAndCopyIntIndicesFloatData( constant int* batchSize [[buffer(0)]],
                                                                            constant const int* input [[buffer(1)]],
                                                                            constant int* inputChannels [[buffer(2)]],
                                                                            constant float* lookup [[buffer(3)]],
                                                                            constant int* vectorSize [[buffer(4)]],
                                                                            device float* output [[buffer(5)]],
                                                                            constant int* outputChannels [[buffer(6)]],
                                                                            constant int* batchNorm [[buffer(7)]],
                                                                            uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    int b;
    int index;
    if( pos.GetMetalTaskIndex2D( *batchNorm, *vectorSize, b, index ) ) {
        b *= MatrixCombineCount;
        int bLast = b + MatrixCombineCount;
        if( bLast > *batchSize ) {
            bLast = *batchSize;
        }
        
        int count = bLast - b;
        
        input += b * *inputChannels;
        output += b * *outputChannels + index;
        lookup += index;
        for( int k = 0; k < count; ++k ) {
            int tableIndex = (int)(*input);
            input += *inputChannels;
            *output = lookup[tableIndex * *vectorSize];
            output += *outputChannels;
        }
    }
}

kernel void matrixKernelBatchVectorChannelLookupAndCopyIntIndicesIntData( constant int* batchSize [[buffer(0)]],
                                                                          constant const int* input [[buffer(1)]],
                                                                          constant int* inputChannels [[buffer(2)]],
                                                                          constant int* lookup [[buffer(3)]],
                                                                          constant int* vectorSize [[buffer(4)]],
                                                                          device int* output [[buffer(5)]],
                                                                          constant int* outputChannels [[buffer(6)]],
                                                                          constant int* batchNorm [[buffer(7)]],
                                                                          uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    int b;
    int index;
    if( pos.GetMetalTaskIndex2D( *batchNorm, *vectorSize, b, index ) ) {
        b *= MatrixCombineCount;
        int bLast = b + MatrixCombineCount;
        if( bLast > *batchSize ) {
            bLast = *batchSize;
        }
        
        int count = bLast - b;
        
        input += b * *inputChannels;
        output += b * *outputChannels + index;
        lookup += index;
        for( int k = 0; k < count; ++k ) {
            int tableIndex = (int)(*input);
            input += *inputChannels;
            *output = lookup[tableIndex * *vectorSize];
            output += *outputChannels;
        }
    }
}

kernel void matrixKernelBatchVectorChannelCopyFloatIndicesFloatData( constant int* batchSize [[buffer(0)]],
                                                                     constant float* input [[buffer(1)]],
                                                                     constant int* inputChannels [[buffer(2)]],
                                                                     constant int* vectorSize [[buffer(3)]],
                                                                     device float* output [[buffer(4)]],
                                                                     constant int* outputChannels [[buffer(5)]],
                                                                     constant int* batchNorm [[buffer(6)]],
                                                                     uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    int b;
    int index;
    if( !pos.GetMetalTaskIndex2D( *batchNorm, *vectorSize, b, index ) ) {
        return;
    }
    
    b *= MatrixCombineCount;
    int bLast = b + MatrixCombineCount;
    if( bLast > *batchSize ) {
        bLast = *batchSize;
    }
    
    int count = bLast - b;
    
    input += b * *inputChannels + index;
    output += b * *outputChannels + index;
    for( int k = 0; k < count; ++k ) {
        *output = *input;
        input += *inputChannels;
        output += *outputChannels;
    }
}

kernel void matrixKernelBatchVectorChannelCopyIntIndicesFloatData( constant int* batchSize [[buffer(0)]],
                                                                   constant int* input [[buffer(1)]],
                                                                   constant int* inputChannels [[buffer(2)]],
                                                                   constant int* vectorSize [[buffer(3)]],
                                                                   device float* output [[buffer(4)]],
                                                                   constant int* outputChannels [[buffer(5)]],
                                                                   constant int* batchNorm [[buffer(6)]],
                                                                   uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    int b;
    int index;
    if( !pos.GetMetalTaskIndex2D( *batchNorm, *vectorSize, b, index ) ) {
        return;
    }
    
    b *= MatrixCombineCount;
    int bLast = b + MatrixCombineCount;
    if( bLast > *batchSize ) {
        bLast = *batchSize;
    }
    
    int count = bLast - b;
    
    input += b * *inputChannels;
    output += b * *outputChannels + index;
    for( int k = 0; k < count; ++k ) {
        *output = *input;
        input += *inputChannels;
        output += *outputChannels;
    }
}

kernel void matrixKernelBatchVectorChannelCopyIntIndicesIntData( constant int* batchSize [[buffer(0)]],
                                                                 constant int* input [[buffer(1)]],
                                                                 constant int* inputChannels [[buffer(2)]],
                                                                 constant int* vectorSize [[buffer(3)]],
                                                                 device int* output [[buffer(4)]],
                                                                 constant int* outputChannels [[buffer(5)]],
                                                                 constant int* batchNorm [[buffer(6)]],
                                                                 uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    int b;
    int index;
    if( !pos.GetMetalTaskIndex2D( *batchNorm, *vectorSize, b, index ) ) {
        return;
    }
    
    b *= MatrixCombineCount;
    int bLast = b + MatrixCombineCount;
    if( bLast > *batchSize ) {
        bLast = *batchSize;
    }
    
    int count = bLast - b;
    
    input += b * *inputChannels;
    output += b * *outputChannels + index;
    for( int k = 0; k < count; ++k ) {
        *output = *input;
        input += *inputChannels;
        output += *outputChannels;
    }
}

kernel void matrixKernelBatchVectorChannelLookupAndAddToTableFloat( constant int* batchSize [[buffer(0)]],
                                                                    constant float* input [[buffer(1)]],
                                                                    constant int* inputIndex [[buffer(2)]],
                                                                    constant int* inputChannelCount [[buffer(3)]],
                                                                    device float* lookup [[buffer(4)]],
                                                                    constant int* lookupVectorCount [[buffer(5)]],
                                                                    constant int* lookupVectorSize [[buffer(6)]],
                                                                    constant float* mult [[buffer(7)]],
                                                                    constant float* matrix [[buffer(8)]],
                                                                    constant int* matrixIndex [[buffer(9)]],
                                                                    constant int* outputChannelCount [[buffer(10)]],
                                                                    uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    int hIndex;
    int wIndex;
    if( !pos.GetMetalTaskIndex2D( *lookupVectorCount, *lookupVectorSize, hIndex, wIndex ) ) {
        return;
    }
    
    input += *inputIndex;
    matrix += *matrixIndex;
    
    for( int i = 0; i < *batchSize; i++ ) {
        int tableIndex = (int)(*input);
        if( hIndex == tableIndex ) {
            lookup[tableIndex * *lookupVectorSize + wIndex] += matrix[wIndex] * *mult;
        }
        input += *inputChannelCount;
        matrix += *outputChannelCount;
    }
}
    
kernel void matrixKernelBatchVectorChannelLookupAndAddToTableInt( constant int* batchSize [[buffer(0)]],
                                                                  constant int* input [[buffer(1)]],
                                                                  constant int* inputIndex [[buffer(2)]],
                                                                  constant int* inputChannelCount [[buffer(3)]],
                                                                  device float* lookup [[buffer(4)]],
                                                                  constant int* lookupVectorCount [[buffer(5)]],
                                                                  constant int* lookupVectorSize [[buffer(6)]],
                                                                  constant float* mult [[buffer(7)]],
                                                                  constant float* matrix [[buffer(8)]],
                                                                  constant int* matrixIndex [[buffer(9)]],
                                                                  constant int* outputChannelCount [[buffer(10)]],
                                                                  uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    int hIndex;
    int wIndex;
    if( !pos.GetMetalTaskIndex2D( *lookupVectorCount, *lookupVectorSize, hIndex, wIndex ) ) {
        return;
    }
    
    input += *inputIndex;
    matrix += *matrixIndex;
    
    for( int i = 0; i < *batchSize; i++ ) {
        int tableIndex = *input;
        if( hIndex == tableIndex ) {
            lookup[tableIndex * *lookupVectorSize + wIndex] += matrix[wIndex] * *mult;
        }
        input += *inputChannelCount;
        matrix += *outputChannelCount;
    }
}
    
kernel void matrixKernelLookupAndSum( constant int* indices [[buffer(0)]],
                                      constant int* batchSize [[buffer(1)]],
                                      constant int* indexCount [[buffer(2)]],
                                      constant float* table [[buffer(3)]],
                                      constant int* vectorSize [[buffer(4)]],
                                      device float* result [[buffer(5)]],
                                      uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    int batch;
    int elem;
    if( pos.GetMetalTaskIndex2D( *batchSize, *vectorSize, batch, elem ) ) {
        result += batch * *vectorSize + elem;
        indices += batch * *indexCount;
        table += elem;
        if( *indices >= 0 ) {
            *result = table[*indices * *vectorSize];
        } else {
            *result = 0.f;
        }
        for( int i = 1; i < *indexCount; ++i ) {
            ++indices;
            if( *indices >= 0 ) {
                *result += table[*indices * *vectorSize];
            }
        }
    }
}

kernel void vectorKernelEnumBinarizationFloat( constant int* batchSize [[buffer(0)]],
                                               constant float* input [[buffer(1)]],
                                               constant int* enumSize [[buffer(2)]],
                                               device float* result [[buffer(3)]],
                                               uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                               uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                               uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int count = pos.GetMetalTaskCountAndIndex( *batchSize * *enumSize, VectorCombineCount, index, step );
    
    for(int i = 0; i < count; ++i) {
        int batch = index / *enumSize;
        int pos = index % *enumSize;
        if( batch >= *batchSize ) {
            break;
        }
        result[index] = ((int)input[batch] == pos) ? 1 : 0;
        index += step;
    }
}

kernel void vectorKernelEnumBinarizationInt( constant int* batchSize [[buffer(0)]],
                                             constant int* input [[buffer(1)]],
                                             constant int* enumSize [[buffer(2)]],
                                             device float* result [[buffer(3)]],
                                             uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                             uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                             uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int count = pos.GetMetalTaskCountAndIndex( *batchSize * *enumSize, VectorCombineCount, index, step );
    
    for(int i = 0; i < count; ++i) {
        int batch = index / *enumSize;
        int pos = index % *enumSize;
        if( batch >= *batchSize ) {
            break;
        }
        result[index] = ((int)input[batch] == pos) ? 1 : 0;
        index += step;
    }
}
    
kernel void matrixKernelLookupAndAddToTable( constant int* indices [[buffer(0)]],
                                             constant int* batchSize [[buffer(1)]],
                                             constant int* indexCount [[buffer(2)]],
                                             constant float* additions [[buffer(3)]],
                                             constant int* vectorSize [[buffer(4)]],
                                             device float* table [[buffer(5)]],
                                             constant int* vectorCount [[buffer(6)]],
                                             uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    
    int wIndex;
    int hIndex;
    if( pos.GetMetalTaskIndex2D( *vectorCount, *vectorSize, hIndex, wIndex ) ) {
        for( int i = 0; i < *batchSize * *indexCount; i++ ) {
            if( indices[i] == hIndex ) {
                table[hIndex * *vectorSize + wIndex] += additions[( i / *indexCount ) * *vectorSize + wIndex];
            }
        }
    }
}

kernel void vectorKernelBitSetBinarization( constant int* batchSize [[buffer(0)]],
                                            constant int* bitSetElementCount [[buffer(1)]],
                                            constant int* input [[buffer(2)]],
                                            constant int* outputVectorSize [[buffer(3)]],
                                            device float* result [[buffer(4)]],
                                            uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                            uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                            uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    const int BitsPerElement = sizeof(int) * CHAR_BIT;
    
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    
    int index;
    int step;
    int count = pos.GetMetalTaskCountAndIndex( *batchSize * *outputVectorSize, 1, index, step );
    
    for( int i = 0; i < count; ++i ) {
        int batchIndex = index / *outputVectorSize;
        int inputBatchBegin = batchIndex * *bitSetElementCount;
        int globalBitIndex = index % *outputVectorSize;
        int elementIndex = globalBitIndex / BitsPerElement;
        
        int inputElement = input[inputBatchBegin + elementIndex];
        int bitIndex = globalBitIndex % 32;
        
        result[index] = inputElement & ( 1 << bitIndex ) ? 1.0f : 0.0f;
        
        index += step;
    }
}

kernel void matrixKernelUpsampling2DForwardInt( constant int* heightCopyCount [[buffer(0)]],
                                                constant int* widthCopyCount [[buffer(1)]],
                                                constant int* pixelSize [[buffer(2)]],
                                                constant int* batchSize [[buffer(3)]],
                                                constant int* inputHeight [[buffer(4)]],
                                                constant int* inputRowSize [[buffer(5)]],
                                                constant int* input [[buffer(6)]],
                                                constant int* resultHeight [[buffer(7)]],
                                                constant int* resultRowSize [[buffer(8)]],
                                                device int* result [[buffer(9)]],
                                                uint2 thread_position_in_grid [[ thread_position_in_grid ]]  )
{
    C2DPosition pos( thread_position_in_grid );
    
    int resultI;
    int resultJ;
    if( !pos.GetMetalTaskIndex2D( *resultHeight, *resultRowSize, resultI, resultJ ) ) {
        return;
    }
    const int inputI = resultI / *heightCopyCount;
    const int inputJ = ( resultJ / *pixelSize / *widthCopyCount ) * *pixelSize + resultJ % *pixelSize;
    
    for( int batchIndex = 0; batchIndex < *batchSize; ++batchIndex ) {
        *( result + resultI * *resultRowSize + resultJ ) = *( input + inputI * *inputRowSize + inputJ );
        input += *inputHeight * *inputRowSize;
        result += *resultHeight * *resultRowSize;
    }
}

kernel void matrixKernelUpsampling2DForwardFloat( constant int* heightCopyCount [[buffer(0)]],
                                                  constant int* widthCopyCount [[buffer(1)]],
                                                  constant int* pixelSize [[buffer(2)]],
                                                  constant int* batchSize [[buffer(3)]],
                                                  constant int* inputHeight [[buffer(4)]],
                                                  constant int* inputRowSize [[buffer(5)]],
                                                  constant float* input [[buffer(6)]],
                                                  constant int* resultHeight [[buffer(7)]],
                                                  constant int* resultRowSize [[buffer(8)]],
                                                  device float* result [[buffer(9)]],
                                                  uint2 thread_position_in_grid [[ thread_position_in_grid ]]  )
{
    C2DPosition pos( thread_position_in_grid );
    
    int resultI;
    int resultJ;
    if( !pos.GetMetalTaskIndex2D( *resultHeight, *resultRowSize, resultI, resultJ ) ) {
        return;
    }
    const int inputI = resultI / *heightCopyCount;
    const int inputJ = ( resultJ / *pixelSize / *widthCopyCount ) * *pixelSize + resultJ % *pixelSize;
    
    for( int batchIndex = 0; batchIndex < *batchSize; ++batchIndex ) {
        *( result + resultI * *resultRowSize + resultJ ) = *( input + inputI * *inputRowSize + inputJ );
        input += *inputHeight * *inputRowSize;
        result += *resultHeight * *resultRowSize;
    }
}

kernel void matrixKernelUpsampling2DBackward( constant int* heightCopyCount [[buffer(0)]],
                                              constant int* widthCopyCount [[buffer(1)]],
                                              constant int* pixelSize [[buffer(2)]],
                                              constant int* batchSize [[buffer(3)]],
                                              constant int* inputHeight [[buffer(4)]],
                                              constant int* inputRowSize [[buffer(5)]],
                                              constant float* input [[buffer(6)]],
                                              constant int* resultHeight [[buffer(7)]],
                                              constant int* resultRowSize [[buffer(8)]],
                                              device float* result [[buffer(9)]],
                                              uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    
    int resultI;
    int resultJ;
    if( !pos.GetMetalTaskIndex2D( *resultHeight, *resultRowSize, resultI, resultJ ) ) {
        return;
    }
    
    const int pixel = resultJ % *pixelSize;
    resultJ /= *pixelSize;
    
    const int inputIFirst = *heightCopyCount * resultI;
    const int inputILast = min( *inputHeight, *heightCopyCount * ( resultI + 1 ) );
    
    const int inputJFirst = *widthCopyCount * resultJ;
    const int inputJLast = min( *inputRowSize / *pixelSize, *widthCopyCount * ( resultJ + 1 ) );
    
    for( int batchIndex = 0; batchIndex < *batchSize; ++batchIndex ) {
        float res = 0;
        for( int inputI = inputIFirst; inputI < inputILast; inputI++ ) {
            for( int inputJ = inputJFirst; inputJ < inputJLast; inputJ++ ) {
                res += *( input + inputI * *inputRowSize + inputJ * *pixelSize + pixel );
            }
        }
        *( result + resultI * *resultRowSize + resultJ * *pixelSize + pixel ) = res;
        result += *resultHeight * *resultRowSize;
        input += *inputHeight * *inputRowSize;
    }
}

kernel void vectorKernelBuildIntegerHist( constant int* numbers [[buffer(0)]],
                                          constant int& numbersCount [[buffer(1)]], 
                                          device atomic_int* resultHandle [[buffer(2)]],
                                          uint thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C1DPosition pos( thread_position_in_grid );
    int index;
    if( pos.GetMetalTaskIndex( numbersCount, index ) ) {
        const int currNumber = numbers[index];
        if( currNumber >= 0 ) {
            atomic_fetch_add_explicit( resultHandle + currNumber, 1, memory_order_relaxed );
        }
    }
}

kernel void matrixKernelMatrixRowsToVectorSquaredL2Distance( constant float* matrix [[buffer(0)]],
                                                             constant int& matrixHeight [[buffer(1)]],
                                                             constant int& matrixWidth [[buffer(2)]],
                                                             constant float* vector [[buffer(3)]],
                                                             device float* result [[buffer(4)]],
                                                             threadgroup float* intermediate [[threadgroup(5)]],
                                                             uint2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                                             uint2 threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                                             uint2 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{


    int widthCombineCount = ( matrixWidth + threads_per_threadgroup.x - 1) / threads_per_threadgroup.x;

    C2DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int column;
    int row;
    if( pos.GetMetalTaskIndex2D( matrixHeight, matrixWidth, 1, 1, row, column ) ) {
        int step;
        int count = pos.GetMetalWidthTaskCountAndIndex( matrixWidth, widthCombineCount, column, step );

        float squareSum = 0;
        matrix += row * matrixWidth + column;
        vector += column;
        for( int i = 0; i < count; i++ ) {
            squareSum += ( *matrix - *vector ) * ( *matrix - *vector );
            matrix += step;
            vector += step;
        }    

        intermediate[thread_position_in_threadgroup.y * threads_per_threadgroup.x + thread_position_in_threadgroup.x ] = squareSum;

        if( thread_position_in_threadgroup.x == 0) {
            float rowResult = 0;
            intermediate += thread_position_in_threadgroup.y * threads_per_threadgroup.x;
            for( uint i = 0; i < threads_per_threadgroup.x; i++ ) {
                rowResult += intermediate[i];
            }

            result[row] = rowResult;
        }
    }
}

kernel void vectorQrnnFPooling( constant bool& reverse [[buffer(0)]],
                                constant int& sequenceLength [[buffer(1)]],
                                constant int& objectSize [[buffer(2)]],
                                constant float* z [[buffer(3)]],
                                constant float* f [[buffer(4)]],
                                constant float* h0 [[buffer(5)]],
                                device float* res [[buffer(6)]],
                                uint thread_position_in_grid [[thread_position_in_grid]] )
{
    C1DPosition pos( thread_position_in_grid );
    int index;
    if( pos.GetMetalTaskIndex( objectSize, index ) ) {
        int currOffset = reverse != 0 ? index + ( sequenceLength - 1 ) * objectSize : index;
        int nextObjectOffset = reverse != 0 ? -objectSize : objectSize;

        float prevRes = f[currOffset] * h0[index] + ( 1 - f[currOffset] ) * z[currOffset];
        res[currOffset] = prevRes;
        
        for( int step = 0; step < sequenceLength - 1; ++step ) {
            currOffset += nextObjectOffset;
            float currRes = f[currOffset] * ( prevRes - z[currOffset] ) + z[currOffset];
            res[currOffset] = currRes;
            prevRes = currRes;
        }
    }
}

kernel void vectorQrnnIfPooling( constant bool& reverse [[buffer(0)]],
                                 constant int& sequenceLength [[buffer(1)]],
                                 constant int& objectSize [[buffer(2)]],
                                 constant float* z [[buffer(3)]],
                                 constant float* f [[buffer(4)]],
                                 constant float* i [[buffer(5)]],
                                 constant float* h0 [[buffer(6)]],
                                 device float* res [[buffer(7)]],
                                 uint thread_position_in_grid [[thread_position_in_grid]] )
{
    C1DPosition pos( thread_position_in_grid );
    int index;
    if( pos.GetMetalTaskIndex( objectSize, index ) ) {
        int currOffset = reverse != 0 ? index + ( sequenceLength - 1 ) * objectSize : index;
        int nextObjectOffset = reverse != 0 ? -objectSize : objectSize;

        float prevRes = f[currOffset] * h0[index] + i[currOffset] * z[currOffset];
        res[currOffset] = prevRes;
        
        for( int step = 0; step < sequenceLength - 1; ++step ) {
            currOffset += nextObjectOffset;
            float currRes = f[currOffset] * prevRes + i[currOffset] * z[currOffset];
            res[currOffset] = currRes;
            prevRes = currRes;
        }
    }
}

kernel void matrixIndRnnRecurrentSigmoid( constant bool& reverse [[buffer(0)]],
                                          constant int& sequenceLength [[buffer(1)]],
                                          constant int& batchSize [[buffer(2)]],
                                          constant int& objectSize [[buffer(3)]],
                                          constant float* wx [[buffer(4)]],
                                          constant float* u [[buffer(5)]],
                                          device float* h [[buffer(6)]],
                                          uint2 thread_position_in_grid [[thread_position_in_grid]] )
{
    C2DPosition pos( thread_position_in_grid );
    int batch;
    int elem;
    if( pos.GetMetalTaskIndex2D( batchSize, objectSize, batch, elem ) ) {
        const float weight = u[elem];
		const int stepOffset = reverse ? -batchSize * objectSize : batchSize * objectSize;

		int currOffset = batch * objectSize + elem;
		if( reverse ) {
			currOffset += ( sequenceLength - 1 ) * batchSize * objectSize;
		}

		float currRes = 1.f / (1.f + ExponentFunc( -wx[currOffset] ) );
		h[currOffset] = currRes;

		for( int step = 0; step < sequenceLength - 1; ++step ) {
			currOffset += stepOffset;
			currRes = wx[currOffset] + weight * currRes;
			currRes = 1.f / (1.f + ExponentFunc( -currRes ) );
			h[currOffset] = currRes;
		}
    }
}

kernel void matrixIndRnnRecurrentReLU( constant bool& reverse [[buffer(0)]],
                                       constant int& sequenceLength [[buffer(1)]],
                                       constant int& batchSize [[buffer(2)]],
                                       constant int& objectSize [[buffer(3)]],
                                       constant float* wx [[buffer(4)]],
                                       constant float* u [[buffer(5)]],
                                       device float* h [[buffer(6)]],
                                       uint2 thread_position_in_grid [[thread_position_in_grid]] )
{
    C2DPosition pos( thread_position_in_grid );
    int batch;
    int elem;
    if( pos.GetMetalTaskIndex2D( batchSize, objectSize, batch, elem ) ) {
        const float weight = u[elem];
		const int stepOffset = reverse ? -batchSize * objectSize : batchSize * objectSize;

		int currOffset = batch * objectSize + elem;
		if( reverse ) {
			currOffset += ( sequenceLength - 1 ) * batchSize * objectSize;
		}

		float currRes = wx[currOffset];
        currRes = currRes > 0.f ? currRes : 0.f;
		h[currOffset] = currRes;

		for( int step = 0; step < sequenceLength - 1; ++step ) {
			currOffset += stepOffset;
			currRes = wx[currOffset] + weight * currRes;
            currRes = currRes > 0.f ? currRes : 0.f;
			h[currOffset] = currRes;
		}
    }
}

kernel void vectorBertConv( constant float* data [[buffer(0)]],
                            constant float* kernelData [[buffer(1)]],
                            constant int& seqLen [[buffer(2)]],
                            constant int& batchSize [[buffer(3)]],
                            constant int& numHeads [[buffer(4)]],
                            constant int& headSize [[buffer(5)]],
                            constant int& kernelSize [[buffer(6)]],
                            device float* output [[buffer(7)]],
                            uint thread_position_in_grid [[thread_position_in_grid]] )
{
    C1DPosition pos( thread_position_in_grid );
    const int taskCount = seqLen * batchSize * numHeads * headSize;
    int index;
    if( pos.GetMetalTaskIndex( taskCount, index ) ) {
        const int pad = ( kernelSize - 1 ) / 2;
        const int dataSeqStep = batchSize * numHeads * headSize;

        const int outputOffset = index;
        const int h = index % headSize;
        index /= headSize;
        const int b = index % ( batchSize * numHeads );
        const int seq = index / ( batchSize * numHeads );

        const int kernelOffset = index * kernelSize;

        float res = 0.f;
        const int kernelStart = max( 0, pad - seq );
        const int kernelEnd = min( kernelSize, seqLen + pad - seq );
        int dataOffset = h + b * headSize + ( seq - pad + kernelStart ) * dataSeqStep;

        for( int k = kernelStart; k < kernelEnd; ++k ) {
            res += data[dataOffset] * kernelData[kernelOffset + k];
            dataOffset += dataSeqStep;
        }

        output[outputOffset] = res;
    }
}
