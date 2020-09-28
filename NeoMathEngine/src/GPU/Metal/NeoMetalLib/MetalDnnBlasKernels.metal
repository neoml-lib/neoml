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
#include "MetalReduce.h"

kernel void vectorKernelVectorDotProduct( constant float* first [[buffer(0)]],
                                          constant float* second [[buffer(1)]],
                                          constant int* vectorSize [[buffer(2)]],
                                          device float* result [[buffer(3)]],
                                          threadgroup float* buffer [[threadgroup(4)]],
                                          uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                          uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                          uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    buffer[thread_position_in_threadgroup] = 0;
    
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int combine = ( *vectorSize + threads_per_threadgroup - 1 ) / threads_per_threadgroup;
    int count = pos.GetMetalTaskCountAndIndex( *vectorSize, combine, index, step );
    first += index;
    second += index;
    for( int i = 0; i < count; ++i ) {
        buffer[thread_position_in_threadgroup] += (*first) * (*second);
        first += step;
        second += step;
    }
    
    Reduce1DSum( thread_position_in_threadgroup, threads_per_threadgroup, buffer );
    
    if( thread_position_in_threadgroup == 0 ) {
        *result = buffer[0];
    }
}

kernel void matrixKernelRowMultiplyMatrixByMatrix( constant float* first [[buffer(0)]],
                                                   constant float* second [[buffer(1)]],
                                                   constant int* height [[buffer(2)]],
                                                   constant int* width [[buffer(3)]],
                                                   device float* result [[buffer(4)]],
                                                   threadgroup float* buffer [[threadgroup(5)]],
                                                   uint2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                                   uint2 threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                                   uint2 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
	int bufferIndex = thread_position_in_threadgroup.y * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
	buffer[bufferIndex] = 0;

    C2DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
	int row;
	int column;
    if( pos.GetMetalTaskIndex2D( *height, *width, 1, 1, row, column ) ) {
        first += row * *width + column;
        second += row * *width + column;

        int step;
        int combine = ( *width + threads_per_threadgroup.x - 1 ) / threads_per_threadgroup.x;
        int count = pos.GetMetalWidthTaskCountAndIndex( *width, combine, column, step );
        for( int i = 0; i < count; ++i ) {
            buffer[bufferIndex] += (*first) * (*second);
            first += step;
            second += step;
        }
    }

    Reduce2DSum( thread_position_in_threadgroup, threads_per_threadgroup, buffer );

    if( row < *height && thread_position_in_threadgroup.x == 0 ) {
        result[row] += buffer[bufferIndex];
    }
}

kernel void vectorKernelVectorMultiplyAndAdd( constant float* first [[buffer(0)]],
                                             constant float* second [[buffer(1)]],
                                             device float* result [[buffer(2)]],
                                             constant int* count [[buffer(3)]],
                                             constant float* mult [[buffer(4)]],
                                             uint thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C1DPosition pos( thread_position_in_grid );
    int index;
    if( pos.GetMetalTaskIndex( *count, index ) ) {
        result[index] = first[index] + *mult * second[index];
    }
}

kernel void vectorKernelVectorMultiplyAndSub( constant float* first [[buffer(0)]],
                                              constant float* second [[buffer(1)]],
                                              device float* result [[buffer(2)]],
                                              constant int* count [[buffer(3)]],
                                              constant float* mult [[buffer(4)]],
                                              uint thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C1DPosition pos( thread_position_in_grid );
	int index;
	if( pos.GetMetalTaskIndex( *count, index ) ) {
		result[index] = first[index] - *mult * second[index];
	}
}

kernel void vectorKernelEltwiseNotNegative( constant float* first [[buffer(0)]],
                                            device float* result [[buffer(1)]],
                                            constant int& vectorSize [[buffer(2)]],
                                            uint thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C1DPosition pos( thread_position_in_grid );
    int index;
    if( pos.GetMetalTaskIndex( vectorSize, index ) ) {
        result[index] = first[index] >= 0 ? 1.0 : 0.0;
    }
}

kernel void matrixKernelSubVectorFromMatrixRows( constant float* matrix [[buffer(0)]],
                                                 device float* result [[buffer(1)]],
                                                 constant int* matrixHeight [[buffer(2)]],
                                                 constant int* matrixWidth [[buffer(3)]],
                                                 constant float* vector [[buffer(4)]],
                                                 uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
	int i;
	int j;
	if( pos.GetMetalTaskIndex2D( *matrixHeight, *matrixWidth, j, i) ) {
		int index = *matrixWidth * j + i;
		result[index] = matrix[index] - vector[i];
	}
}

kernel void matrixKernelSetVectorToMatrixRows( device float* result [[buffer(0)]],
                                               constant int* matrixHeight [[buffer(1)]],
                                               constant int* matrixWidth [[buffer(2)]],
                                               constant float* vector [[buffer(3)]],
                                               uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
	int i;
	int j;
	if( pos.GetMetalTaskIndex2D( *matrixHeight, *matrixWidth, j, i) ) {
		int index = *matrixWidth * j + i;
		result[index] = vector[i];
	}
}

kernel void vectorKernelSetVectorToMatrixElements( device float* matrix [[buffer(0)]],
                                             constant int* height [[buffer(1)]],
                                             constant int* width [[buffer(2)]],
                                             constant int* rowIndices [[buffer(3)]],
                                             constant int* columnIndices [[buffer(4)]],
                                             constant float* vector [[buffer(5)]],
                                             constant int* vectorSize [[buffer(6)]],
                                             uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                             uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                             uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]]  )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    
    int index;
    int step;
    int count = pos.GetMetalTaskCountAndIndex( *vectorSize, 4, index, step );
    
    for( int i = 0; i < count; ++i ) {
        matrix[rowIndices[index] * *width + columnIndices[index]] = vector[index];
        index += step;
    }
}

kernel void matrixKernelAddVectorToMatrixColumnsFloat( constant float* matrix [[buffer(0)]],
                                                       device float* result [[buffer(1)]],
                                                       constant int* matrixHeight [[buffer(2)]],
                                                       constant int* matrixWidth [[buffer(3)]],
                                                       constant float* vector [[buffer(4)]],
                                                       uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    int i;
    int j;
    if( pos.GetMetalTaskIndex2D( *matrixHeight, *matrixWidth, j, i ) ) {
        int index = *matrixWidth * j + i;
        result[index] = matrix[index] + vector[j];
    }
}

kernel void vectorKernelAddVectorToMatrixColumns( device float* result [[buffer(0)]],
                                                  constant int* matrixHeight [[buffer(1)]],
                                                  constant int* matrixWidth [[buffer(2)]],
                                                  constant float* vector [[buffer(3)]],
                                                  uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                                  uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                                  uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int index;
    int step;
    int count = pos.GetMetalTaskCountAndIndex( *matrixHeight * *matrixWidth, 8, index, step );
    
    for( int i = 0; i < count; i++ ) {
        result[index] += vector[index / *matrixWidth];
        index += step;
    }
}

kernel void matrixKernelSetVectorToMatrixColumns( device float* result [[buffer(0)]],
                                                  constant int* matrixHeight [[buffer(1)]],
                                                  constant int* matrixWidth [[buffer(2)]],
                                                  constant float* vector [[buffer(3)]],
                                                  uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    int i;
    int j;
    if( pos.GetMetalTaskIndex2D( *matrixHeight, *matrixWidth, j, i ) ) {
        int index = *matrixWidth * j + i;
        result[index] = vector[j];
    }
}

kernel void matrixKernelAddVectorToMatrixColumnsInt( constant int* matrix [[buffer(0)]],
                                                     device int* result [[buffer(1)]],
                                                     constant int* matrixHeight [[buffer(2)]],
                                                     constant int* matrixWidth [[buffer(3)]],
                                                     constant int* vector [[buffer(4)]],
                                                     uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
	int i;
	int j;
	if( pos.GetMetalTaskIndex2D( *matrixHeight, *matrixWidth, j, i ) ) {
		int index = *matrixWidth * j + i;
		result[index] = matrix[index] + vector[j];
	}
}

kernel void matrixKernelSubVectorFromMatrixColumns( constant float* matrix [[buffer(0)]],
                                                    device float* result [[buffer(1)]],
                                                    constant int* matrixHeight [[buffer(2)]],
                                                    constant int* matrixWidth [[buffer(3)]],
                                                    constant float* vector [[buffer(4)]],
                                                    uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
	int i;
	int j;
	if( pos.GetMetalTaskIndex2D( *matrixHeight, *matrixWidth, j, i) ) {
		int index = *matrixWidth * j + i;
		result[index] = matrix[index] - vector[j];
	}
}

kernel void matrixKernelSumMatrixColumns( device float* result [[buffer(0)]],
                                          constant float* matrix [[buffer(1)]],
                                          constant int* matrixHeight [[buffer(2)]],
                                          constant int* matrixWidth [[buffer(3)]],
                                          constant int* isNeg [[buffer(4)]],
                                          threadgroup float* buffer [[threadgroup(5)]],
                                          uint2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                          uint2 threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                          uint2 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    int bufferIndex = thread_position_in_threadgroup.y * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
    buffer[bufferIndex] = 0;
    
    C2DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );

    int row;
    int column;
    if( pos.GetMetalTaskIndex2D( *matrixHeight, *matrixWidth, 1, 1, row, column ) ) {
        matrix += row * *matrixWidth + column;
        
        int step;
        int combine = ( *matrixWidth + threads_per_threadgroup.x - 1 ) / threads_per_threadgroup.x;
        int count = pos.GetMetalWidthTaskCountAndIndex( *matrixWidth, combine, column, step );
        for( int i = 0; i < count; ++i ) {
            buffer[bufferIndex] += (*matrix);
            matrix += step;
        }
    }
    
    Reduce2DSum( thread_position_in_threadgroup, threads_per_threadgroup, buffer );
    
    if( row < *matrixHeight && thread_position_in_threadgroup.x == 0 ) {
        result[row] += ( *isNeg != 0 ) ? -buffer[bufferIndex] : buffer[bufferIndex];
    }
}

kernel void matrixKernelFindMaxValueWithIndicesInRows( constant float* matrix [[buffer(0)]],
                                                       constant int* matrixHeight [[buffer(1)]],
                                                       constant int* matrixWidth [[buffer(2)]],
                                                       device float* result [[buffer(3)]],
                                                       device int* indices [[buffer(4)]],
                                                       threadgroup float* bufferValues [[threadgroup(5)]],
                                                       threadgroup int* bufferIndexes [[threadgroup(6)]],
                                                       uint2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                                       uint2 threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                                       uint2 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    int bufferIndex = thread_position_in_threadgroup.y * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
    bufferValues[bufferIndex] = -FLT_MAX;
    bufferIndexes[bufferIndex] = 0;
    
    C2DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int row;
    int column;
    if( pos.GetMetalTaskIndex2D( *matrixHeight, *matrixWidth, 1, 1, row, column ) ) {
        matrix += row * *matrixWidth + column;
        
        int combine = ( *matrixWidth + threads_per_threadgroup.x - 1 ) / threads_per_threadgroup.x;
        int step;
        int count = pos.GetMetalWidthTaskCountAndIndex( *matrixWidth, combine, column, step );
        for( int i = 0; i < count; ++i ) {
            if( bufferValues[bufferIndex] < (*matrix) ) {
                bufferValues[bufferIndex] = (*matrix);
                bufferIndexes[bufferIndex] = column + i * step;
            }
            matrix += step;
        }
    }
    
    Reduce2DMax( thread_position_in_threadgroup, threads_per_threadgroup, bufferValues, bufferIndexes );
    
    if( row < *matrixHeight && thread_position_in_threadgroup.x == 0 ) {
        if( result[row] <= bufferValues[bufferIndex] ) {
            result[row] = bufferValues[bufferIndex];
            indices[row] = bufferIndexes[bufferIndex];
        }
    }
}

kernel void matrixKernelFindMaxValueInRows( constant float* matrix [[buffer(0)]],
                                            constant int* matrixHeight [[buffer(1)]],
                                            constant int* matrixWidth [[buffer(2)]],
                                            device float* result [[buffer(3)]],
                                            threadgroup float* bufferValues [[threadgroup(4)]],
                                            uint2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                            uint2 threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                            uint2 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    int bufferIndex = thread_position_in_threadgroup.y * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
    bufferValues[bufferIndex] = -FLT_MAX;
    
    C2DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    int row;
    int column;
    if( pos.GetMetalTaskIndex2D( *matrixHeight, *matrixWidth, 1, 1, row, column ) ) {
        matrix += row * *matrixWidth + column;
        
        int step;
        int combine = ( *matrixWidth + threads_per_threadgroup.x - 1 ) / threads_per_threadgroup.x;
        int count = pos.GetMetalWidthTaskCountAndIndex( *matrixWidth, combine, column, step );
        for( int i = 0; i < count; ++i ) {
            if( bufferValues[bufferIndex] <= (*matrix) ) {
                bufferValues[bufferIndex] = (*matrix);
            }
            matrix += step;
        }
    }
    
    Reduce2DMax( thread_position_in_threadgroup, threads_per_threadgroup, bufferValues );
    
    if( row < *matrixHeight && thread_position_in_threadgroup.x == 0 ) {
        if( result[row] <= bufferValues[bufferIndex] ) {
            result[row] = bufferValues[bufferIndex];
        }
    }
}

kernel void matrixKernelFindMaxValueInColumns( constant int& batchSize [[buffer(0)]],
                                              constant float* matrix [[buffer(1)]],
                                              constant int& matrixHeight [[buffer(2)]],
                                              constant int& matrixWidth [[buffer(3)]],
                                              device float* result [[buffer(4)]],
                                              device int* resultIndices [[buffer(5)]],
                                              uint2 thread_position_in_grid [[ thread_position_in_grid ]],
                                              uint2 threads_per_threadgroup [[ threads_per_threadgroup ]],
                                              uint2 threadgroup_position_in_grid [[ threadgroup_position_in_grid ]] )
{
  C2DPosition pos( thread_position_in_grid );
  int batchIndex;
  int colIndex;
  if( pos.GetMetalTaskIndex2D( batchSize, matrixWidth, batchIndex, colIndex ) ) {
    matrix += batchIndex * matrixHeight * matrixWidth + colIndex;
    float maxVal = *matrix;
    int maxInd = 0;
    matrix += matrixWidth;
    for( int i = 1; i < matrixHeight; i++ ) {
      if( *matrix > maxVal ) {
        maxVal = *matrix;
        maxInd = i;
      }
      matrix += matrixWidth;
    }

    result[batchIndex * matrixWidth + colIndex] = maxVal;
    resultIndices[batchIndex * matrixWidth + colIndex] = maxInd;
  }
}

kernel void vectorKernelFindMaxValueInSet( device float* result [[buffer(0)]],
                                                 constant float* vector [[buffer(1)]],
                                                 constant int* vectorSize [[buffer(2)]],
                                                 uint thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C1DPosition pos( thread_position_in_grid );
    
    int index;
	if( pos.GetMetalTaskIndex( *vectorSize, index ) ) {
		float value = vector[index];
        if( result[index] < value ) {
            result[index] = value;
        }
	}
}

kernel void vectorKernelFindMaxValueInSetWithIndices( device float* result [[buffer(0)]],
                                                           device int* indices [[buffer(1)]],
                                                           constant float* vector [[buffer(2)]],
                                                           constant int* number [[buffer(3)]],
                                                           constant int* vectorSize [[buffer(4)]],
                                                           uint thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C1DPosition pos( thread_position_in_grid );
    
    int index;
    if( pos.GetMetalTaskIndex( *vectorSize, index ) ) {
        float value = vector[index];
        if( result[index] < value ) {
            result[index] = value;
            indices[index] = *number;
        }
    }
}

kernel void vectorKernelVectorSpreadValues( constant float* source [[buffer(0)]],
                                            constant int* indices [[buffer(1)]],
                                            device float* vector [[buffer(2)]],
                                            constant int* number [[buffer(3)]],
                                            constant int* vectorSize [[buffer(4)]],
                                            uint thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C1DPosition pos( thread_position_in_grid );
    
	int index;
	if( pos.GetMetalTaskIndex( *vectorSize, index ) ) {
        if( indices[index] == *number ) {
            vector[index] = source[index];
        }
	}
}

// The kernel for matrix product. Each thread calculates one element of the result
kernel void matrixKernelMultiplyMatrixByMatrixThread1x1( constant float* first [[buffer(0)]],
                                                        constant int* firstHeight [[buffer(1)]],
                                                        constant int* firstWidth [[buffer(2)]],
                                                        constant float* second [[buffer(3)]],
                                                        constant int* secondWidth[[buffer(4)]],
                                                        device float* result [[buffer(5)]],
                                                        uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    int heightIndex;
    int widthIndex;
    if( pos.GetMetalTaskIndex2D( *firstHeight, *secondWidth, heightIndex, widthIndex ) ) {
        int resultIndex = heightIndex * *secondWidth + widthIndex;
        float res = 0;
        first += heightIndex * *firstWidth;
        second += widthIndex;
        for( int i = 0; i < *firstWidth; i++ ) {
            res += *first * *second;
            first += 1;
            second += *secondWidth;
        }
        result[resultIndex] = res;
    }
}

// Calculates a 4*4 block of the matrix product
inline void computeMultiplyMatrixByMatrixFor4x4Block( constant float* a, constant float* b, int firstWidth, int secondWidth,
                                                    thread float4& acc0, thread float4& acc1, thread float4& acc2, thread float4& acc3 )
{
    int i = 0;
    while( i <= firstWidth - 4 ) {
        float4 a0 = float4(a[0], a[1], a[2], a[3]);
        float4 a1 = float4(a[firstWidth], a[firstWidth + 1], a[firstWidth + 2], a[firstWidth + 3]);
        float4 a2 = float4(a[2 * firstWidth], a[2 * firstWidth + 1], a[2 * firstWidth + 2], a[2 * firstWidth + 3]);
        float4 a3 = float4(a[3 * firstWidth], a[3 * firstWidth + 1], a[3 * firstWidth + 2], a[3 * firstWidth + 3]);

        float4 b0 = float4(b[0], b[1], b[2], b[3]);
        b += secondWidth;

        acc0.x = fma(a0.x, b0.x, acc0.x);
        acc0.y = fma(a0.x, b0.y, acc0.y);
        acc0.z = fma(a0.x, b0.z, acc0.z);
        acc0.w = fma(a0.x, b0.w, acc0.w);

        acc1.x = fma(a1.x, b0.x, acc1.x);
        acc1.y = fma(a1.x, b0.y, acc1.y);
        acc1.z = fma(a1.x, b0.z, acc1.z);
        acc1.w = fma(a1.x, b0.w, acc1.w);

        acc2.x = fma(a2.x, b0.x, acc2.x);
        acc2.y = fma(a2.x, b0.y, acc2.y);
        acc2.z = fma(a2.x, b0.z, acc2.z);
        acc2.w = fma(a2.x, b0.w, acc2.w);

        acc3.x = fma(a3.x, b0.x, acc3.x);
        acc3.y = fma(a3.x, b0.y, acc3.y);
        acc3.z = fma(a3.x, b0.z, acc3.z);
        acc3.w = fma(a3.x, b0.w, acc3.w);

        b0 = float4(b[0], b[1], b[2], b[3]);
        b += secondWidth;

        acc0.x = fma(a0.y, b0.x, acc0.x);
        acc0.y = fma(a0.y, b0.y, acc0.y);
        acc0.z = fma(a0.y, b0.z, acc0.z);
        acc0.w = fma(a0.y, b0.w, acc0.w);

        acc1.x = fma(a1.y, b0.x, acc1.x);
        acc1.y = fma(a1.y, b0.y, acc1.y);
        acc1.z = fma(a1.y, b0.z, acc1.z);
        acc1.w = fma(a1.y, b0.w, acc1.w);

        acc2.x = fma(a2.y, b0.x, acc2.x);
        acc2.y = fma(a2.y, b0.y, acc2.y);
        acc2.z = fma(a2.y, b0.z, acc2.z);
        acc2.w = fma(a2.y, b0.w, acc2.w);

        acc3.x = fma(a3.y, b0.x, acc3.x);
        acc3.y = fma(a3.y, b0.y, acc3.y);
        acc3.z = fma(a3.y, b0.z, acc3.z);
        acc3.w = fma(a3.y, b0.w, acc3.w);

        b0 = float4(b[0], b[1], b[2], b[3]);
        b += secondWidth;

        acc0.x = fma(a0.z, b0.x, acc0.x);
        acc0.y = fma(a0.z, b0.y, acc0.y);
        acc0.z = fma(a0.z, b0.z, acc0.z);
        acc0.w = fma(a0.z, b0.w, acc0.w);

        acc1.x = fma(a1.z, b0.x, acc1.x);
        acc1.y = fma(a1.z, b0.y, acc1.y);
        acc1.z = fma(a1.z, b0.z, acc1.z);
        acc1.w = fma(a1.z, b0.w, acc1.w);

        acc2.x = fma(a2.z, b0.x, acc2.x);
        acc2.y = fma(a2.z, b0.y, acc2.y);
        acc2.z = fma(a2.z, b0.z, acc2.z);
        acc2.w = fma(a2.z, b0.w, acc2.w);

        acc3.x = fma(a3.z, b0.x, acc3.x);
        acc3.y = fma(a3.z, b0.y, acc3.y);
        acc3.z = fma(a3.z, b0.z, acc3.z);
        acc3.w = fma(a3.z, b0.w, acc3.w);

        b0 = float4(b[0], b[1], b[2], b[3]);
        b += secondWidth;

        acc0.x = fma(a0.w, b0.x, acc0.x);
        acc0.y = fma(a0.w, b0.y, acc0.y);
        acc0.z = fma(a0.w, b0.z, acc0.z);
        acc0.w = fma(a0.w, b0.w, acc0.w);

        acc1.x = fma(a1.w, b0.x, acc1.x);
        acc1.y = fma(a1.w, b0.y, acc1.y);
        acc1.z = fma(a1.w, b0.z, acc1.z);
        acc1.w = fma(a1.w, b0.w, acc1.w);

        acc2.x = fma(a2.w, b0.x, acc2.x);
        acc2.y = fma(a2.w, b0.y, acc2.y);
        acc2.z = fma(a2.w, b0.z, acc2.z);
        acc2.w = fma(a2.w, b0.w, acc2.w);

        acc3.x = fma(a3.w, b0.x, acc3.x);
        acc3.y = fma(a3.w, b0.y, acc3.y);
        acc3.z = fma(a3.w, b0.z, acc3.z);
        acc3.w = fma(a3.w, b0.w, acc3.w);

        a += 4;
        i += 4;
    }

    while( i < firstWidth ) {
        float a0 = a[0];
        float a1 = a[firstWidth];
        float a2 = a[2*firstWidth];
        float a3 = a[3*firstWidth];

        float4 b0 = float4(b[0], b[1], b[2], b[3]);
        b += secondWidth;

        acc0.x = fma(a0, b0.x, acc0.x);
        acc0.y = fma(a0, b0.y, acc0.y);
        acc0.z = fma(a0, b0.z, acc0.z);
        acc0.w = fma(a0, b0.w, acc0.w);

        acc1.x = fma(a1, b0.x, acc1.x);
        acc1.y = fma(a1, b0.y, acc1.y);
        acc1.z = fma(a1, b0.z, acc1.z);
        acc1.w = fma(a1, b0.w, acc1.w);

        acc2.x = fma(a2, b0.x, acc2.x);
        acc2.y = fma(a2, b0.y, acc2.y);
        acc2.z = fma(a2, b0.z, acc2.z);
        acc2.w = fma(a2, b0.w, acc2.w);

        acc3.x = fma(a3, b0.x, acc3.x);
        acc3.y = fma(a3, b0.y, acc3.y);
        acc3.z = fma(a3, b0.z, acc3.z);
        acc3.w = fma(a3, b0.w, acc3.w);

        a++;
        i++;
    }
}

// Calculates a 4*4 block when multiplying a matrix by a transposed matrix
inline void computeMultiplyMatrixByTransposedMatrixFor4x4Block( constant float* a, constant float* bi, int firstWidth,
                                                                thread float4& acc0, thread float4& acc1, thread float4& acc2, thread float4& acc3 )
{
    int i = 0;
    while( i <= firstWidth - 4 ) {
        float4 a0 = float4(a[0], a[1], a[2], a[3]);
        float4 a1 = float4(a[firstWidth], a[firstWidth + 1], a[firstWidth + 2], a[firstWidth + 3]);
        float4 a2 = float4(a[2 * firstWidth], a[2 * firstWidth + 1], a[2 * firstWidth + 2], a[2 * firstWidth + 3]);
        float4 a3 = float4(a[3 * firstWidth], a[3 * firstWidth + 1], a[3 * firstWidth + 2], a[3 * firstWidth + 3]);

        constant float* b = bi;
        float4 b0 = float4(b[0], b[1], b[2], b[3]);
        b += firstWidth;

        acc0.x = fma(a0.x, b0.x, acc0.x);
        acc0.x = fma(a0.y, b0.y, acc0.x);
        acc0.x = fma(a0.z, b0.z, acc0.x);
        acc0.x = fma(a0.w, b0.w, acc0.x);
        acc1.x = fma(a1.x, b0.x, acc1.x);
        acc1.x = fma(a1.y, b0.y, acc1.x);
        acc1.x = fma(a1.z, b0.z, acc1.x);
        acc1.x = fma(a1.w, b0.w, acc1.x);
        acc2.x = fma(a2.x, b0.x, acc2.x);
        acc2.x = fma(a2.y, b0.y, acc2.x);
        acc2.x = fma(a2.z, b0.z, acc2.x);
        acc2.x = fma(a2.w, b0.w, acc2.x);
        acc3.x = fma(a3.x, b0.x, acc3.x);
        acc3.x = fma(a3.y, b0.y, acc3.x);
        acc3.x = fma(a3.z, b0.z, acc3.x);
        acc3.x = fma(a3.w, b0.w, acc3.x);

        b0 = float4(b[0], b[1], b[2], b[3]);
        b += firstWidth;

        acc0.y = fma(a0.x, b0.x, acc0.y);
        acc0.y = fma(a0.y, b0.y, acc0.y);
        acc0.y = fma(a0.z, b0.z, acc0.y);
        acc0.y = fma(a0.w, b0.w, acc0.y);
        acc1.y = fma(a1.x, b0.x, acc1.y);
        acc1.y = fma(a1.y, b0.y, acc1.y);
        acc1.y = fma(a1.z, b0.z, acc1.y);
        acc1.y = fma(a1.w, b0.w, acc1.y);
        acc2.y = fma(a2.x, b0.x, acc2.y);
        acc2.y = fma(a2.y, b0.y, acc2.y);
        acc2.y = fma(a2.z, b0.z, acc2.y);
        acc2.y = fma(a2.w, b0.w, acc2.y);
        acc3.y = fma(a3.x, b0.x, acc3.y);
        acc3.y = fma(a3.y, b0.y, acc3.y);
        acc3.y = fma(a3.z, b0.z, acc3.y);
        acc3.y = fma(a3.w, b0.w, acc3.y);

        b0 = float4(b[0], b[1], b[2], b[3]);
        b += firstWidth;
    
        acc0.z = fma(a0.x, b0.x, acc0.z);
        acc0.z = fma(a0.y, b0.y, acc0.z);
        acc0.z = fma(a0.z, b0.z, acc0.z);
        acc0.z = fma(a0.w, b0.w, acc0.z);
        acc1.z = fma(a1.x, b0.x, acc1.z);
        acc1.z = fma(a1.y, b0.y, acc1.z);
        acc1.z = fma(a1.z, b0.z, acc1.z);
        acc1.z = fma(a1.w, b0.w, acc1.z);
        acc2.z = fma(a2.x, b0.x, acc2.z);
        acc2.z = fma(a2.y, b0.y, acc2.z);
        acc2.z = fma(a2.z, b0.z, acc2.z);
        acc2.z = fma(a2.w, b0.w, acc2.z);
        acc3.z = fma(a3.x, b0.x, acc3.z);
        acc3.z = fma(a3.y, b0.y, acc3.z);
        acc3.z = fma(a3.z, b0.z, acc3.z);
        acc3.z = fma(a3.w, b0.w, acc3.z);

        b0 = float4(b[0], b[1], b[2], b[3]);
        b += firstWidth;
    
        acc0.w = fma(a0.x, b0.x, acc0.w);
        acc0.w = fma(a0.y, b0.y, acc0.w);
        acc0.w = fma(a0.z, b0.z, acc0.w);
        acc0.w = fma(a0.w, b0.w, acc0.w);
        acc1.w = fma(a1.x, b0.x, acc1.w);
        acc1.w = fma(a1.y, b0.y, acc1.w);
        acc1.w = fma(a1.z, b0.z, acc1.w);
        acc1.w = fma(a1.w, b0.w, acc1.w);
        acc2.w = fma(a2.x, b0.x, acc2.w);
        acc2.w = fma(a2.y, b0.y, acc2.w);
        acc2.w = fma(a2.z, b0.z, acc2.w);
        acc2.w = fma(a2.w, b0.w, acc2.w);
        acc3.w = fma(a3.x, b0.x, acc3.w);
        acc3.w = fma(a3.y, b0.y, acc3.w);
        acc3.w = fma(a3.z, b0.z, acc3.w);
        acc3.w = fma(a3.w, b0.w, acc3.w);

        a += 4;
        i += 4;
        bi += 4;
    }

    while( i < firstWidth ) {
        float a0 = a[0];
        float a1 = a[firstWidth];
        float a2 = a[2 * firstWidth];
        float a3 = a[3 * firstWidth];

        float4 b0 = float4(bi[0], bi[firstWidth], bi[2 * firstWidth], bi[3 * firstWidth]);
        bi += 1;

        acc0.x = fma(a0, b0.x, acc0.x);
        acc0.y = fma(a0, b0.y, acc0.y);
        acc0.z = fma(a0, b0.z, acc0.z);
        acc0.w = fma(a0, b0.w, acc0.w);

        acc1.x = fma(a1, b0.x, acc1.x);
        acc1.y = fma(a1, b0.y, acc1.y);
        acc1.z = fma(a1, b0.z, acc1.z);
        acc1.w = fma(a1, b0.w, acc1.w);

        acc2.x = fma(a2, b0.x, acc2.x);
        acc2.y = fma(a2, b0.y, acc2.y);
        acc2.z = fma(a2, b0.z, acc2.z);
        acc2.w = fma(a2, b0.w, acc2.w);

        acc3.x = fma(a3, b0.x, acc3.x);
        acc3.y = fma(a3, b0.y, acc3.y);
        acc3.z = fma(a3, b0.z, acc3.z);
        acc3.w = fma(a3, b0.w, acc3.w);

        a++;
        i++;
    }
}

// Calculates a 4*4 block when multiplying a transposed matrix by a matrix
inline void computeMultiplyTransposedMatrixByMatrixFor4x4Block( constant float* a, constant float* b, int firstHeight, int firstWidth, int secondWidth,
                                                    thread float4& acc0, thread float4& acc1, thread float4& acc2, thread float4& acc3 )
{
    int i = 0;
    while( i <= firstHeight - 4 ) {
        float4 a0 = float4(a[0], a[firstWidth], a[2 * firstWidth], a[3 * firstWidth]);
        float4 a1 = float4(a[1], a[firstWidth + 1], a[2 * firstWidth + 1], a[3 * firstWidth + 1]);
        float4 a2 = float4(a[2], a[firstWidth + 2], a[2 * firstWidth + 2], a[3 * firstWidth + 2]);
        float4 a3 = float4(a[3], a[firstWidth + 3], a[2 * firstWidth + 3], a[3 * firstWidth + 3]);

        float4 b0 = float4(b[0], b[1], b[2], b[3]);
        b += secondWidth;

        acc0.x = fma(a0.x, b0.x, acc0.x);
        acc0.y = fma(a0.x, b0.y, acc0.y);
        acc0.z = fma(a0.x, b0.z, acc0.z);
        acc0.w = fma(a0.x, b0.w, acc0.w);

        acc1.x = fma(a1.x, b0.x, acc1.x);
        acc1.y = fma(a1.x, b0.y, acc1.y);
        acc1.z = fma(a1.x, b0.z, acc1.z);
        acc1.w = fma(a1.x, b0.w, acc1.w);

        acc2.x = fma(a2.x, b0.x, acc2.x);
        acc2.y = fma(a2.x, b0.y, acc2.y);
        acc2.z = fma(a2.x, b0.z, acc2.z);
        acc2.w = fma(a2.x, b0.w, acc2.w);

        acc3.x = fma(a3.x, b0.x, acc3.x);
        acc3.y = fma(a3.x, b0.y, acc3.y);
        acc3.z = fma(a3.x, b0.z, acc3.z);
        acc3.w = fma(a3.x, b0.w, acc3.w);

        b0 = float4(b[0], b[1], b[2], b[3]);
        b += secondWidth;

        acc0.x = fma(a0.y, b0.x, acc0.x);
        acc0.y = fma(a0.y, b0.y, acc0.y);
        acc0.z = fma(a0.y, b0.z, acc0.z);
        acc0.w = fma(a0.y, b0.w, acc0.w);

        acc1.x = fma(a1.y, b0.x, acc1.x);
        acc1.y = fma(a1.y, b0.y, acc1.y);
        acc1.z = fma(a1.y, b0.z, acc1.z);
        acc1.w = fma(a1.y, b0.w, acc1.w);

        acc2.x = fma(a2.y, b0.x, acc2.x);
        acc2.y = fma(a2.y, b0.y, acc2.y);
        acc2.z = fma(a2.y, b0.z, acc2.z);
        acc2.w = fma(a2.y, b0.w, acc2.w);

        acc3.x = fma(a3.y, b0.x, acc3.x);
        acc3.y = fma(a3.y, b0.y, acc3.y);
        acc3.z = fma(a3.y, b0.z, acc3.z);
        acc3.w = fma(a3.y, b0.w, acc3.w);

        b0 = float4(b[0], b[1], b[2], b[3]);
        b += secondWidth;

        acc0.x = fma(a0.z, b0.x, acc0.x);
        acc0.y = fma(a0.z, b0.y, acc0.y);
        acc0.z = fma(a0.z, b0.z, acc0.z);
        acc0.w = fma(a0.z, b0.w, acc0.w);

        acc1.x = fma(a1.z, b0.x, acc1.x);
        acc1.y = fma(a1.z, b0.y, acc1.y);
        acc1.z = fma(a1.z, b0.z, acc1.z);
        acc1.w = fma(a1.z, b0.w, acc1.w);

        acc2.x = fma(a2.z, b0.x, acc2.x);
        acc2.y = fma(a2.z, b0.y, acc2.y);
        acc2.z = fma(a2.z, b0.z, acc2.z);
        acc2.w = fma(a2.z, b0.w, acc2.w);

        acc3.x = fma(a3.z, b0.x, acc3.x);
        acc3.y = fma(a3.z, b0.y, acc3.y);
        acc3.z = fma(a3.z, b0.z, acc3.z);
        acc3.w = fma(a3.z, b0.w, acc3.w);

        b0 = float4(b[0], b[1], b[2], b[3]);
        b += secondWidth;

        acc0.x = fma(a0.w, b0.x, acc0.x);
        acc0.y = fma(a0.w, b0.y, acc0.y);
        acc0.z = fma(a0.w, b0.z, acc0.z);
        acc0.w = fma(a0.w, b0.w, acc0.w);

        acc1.x = fma(a1.w, b0.x, acc1.x);
        acc1.y = fma(a1.w, b0.y, acc1.y);
        acc1.z = fma(a1.w, b0.z, acc1.z);
        acc1.w = fma(a1.w, b0.w, acc1.w);

        acc2.x = fma(a2.w, b0.x, acc2.x);
        acc2.y = fma(a2.w, b0.y, acc2.y);
        acc2.z = fma(a2.w, b0.z, acc2.z);
        acc2.w = fma(a2.w, b0.w, acc2.w);

        acc3.x = fma(a3.w, b0.x, acc3.x);
        acc3.y = fma(a3.w, b0.y, acc3.y);
        acc3.z = fma(a3.w, b0.z, acc3.z);
        acc3.w = fma(a3.w, b0.w, acc3.w);

        a += 4 * firstWidth;
        i += 4;
    }

    while( i < firstHeight ) {
        float a0 = a[0];
        float a1 = a[1];
        float a2 = a[2];
        float a3 = a[3];

        float4 b0 = float4(b[0], b[1], b[2], b[3]);
        b += secondWidth;

        acc0.x = fma(a0, b0.x, acc0.x);
        acc0.y = fma(a0, b0.y, acc0.y);
        acc0.z = fma(a0, b0.z, acc0.z);
        acc0.w = fma(a0, b0.w, acc0.w);

        acc1.x = fma(a1, b0.x, acc1.x);
        acc1.y = fma(a1, b0.y, acc1.y);
        acc1.z = fma(a1, b0.z, acc1.z);
        acc1.w = fma(a1, b0.w, acc1.w);

        acc2.x = fma(a2, b0.x, acc2.x);
        acc2.y = fma(a2, b0.y, acc2.y);
        acc2.z = fma(a2, b0.z, acc2.z);
        acc2.w = fma(a2, b0.w, acc2.w);

        acc3.x = fma(a3, b0.x, acc3.x);
        acc3.y = fma(a3, b0.y, acc3.y);
        acc3.z = fma(a3, b0.z, acc3.z);
        acc3.w = fma(a3, b0.w, acc3.w);

        a += firstWidth;
        i++;
    }
}

// Multiplies a matrix by a transposed matrix using a kernel and adding to the result
// Each thread calculates one element of the result
kernel void matrixKernelMultiplyMatrixByTransposedMatrixAndAddThread1x1( constant float* first [[buffer(0)]],
                                                                        constant int* firstHeight [[buffer(1)]],
                                                                        constant int* firstWidth [[buffer(2)]],
                                                                        constant float* second [[buffer(3)]],
                                                                        constant int* secondHeight[[buffer(4)]],
                                                                        device float* result [[buffer(5)]],
                                                                        uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    int heightIndex;
    int widthIndex;
    if( pos.GetMetalTaskIndex2D( *firstHeight, *secondHeight, heightIndex, widthIndex ) ) {
        int resultIndex = heightIndex * *secondHeight + widthIndex;
        first += heightIndex * *firstWidth;
        second += widthIndex * *firstWidth;
        for( int i = 0; i < *firstWidth; i++ ) {
            result[resultIndex] += *first * *second;
            first += 1;
            second += 1;
        }
    }
}

// Multiplies a matrix by a transposed matrix using a kernel and adding to the result
// Each thread calculates 16 elements (a 4*4 block) of the result
kernel void matrixKernelMultiplyMatrixByTransposedMatrixAndAddThread4x4( constant float* first [[buffer(0)]],
                                                                      constant int& firstHeight [[buffer(1)]],
                                                                      constant int& firstWidth [[buffer(2)]],
                                                                      constant float* second [[buffer(3)]],
                                                                      constant int& secondHeight[[buffer(4)]],
                                                                      device float* result [[buffer(5)]],
                                                                      uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    int x, y;

    if( pos.GetMetalTaskIndex2D( firstHeight / 4, secondHeight / 4, y, x ) ) {
        constant float *a = first + firstWidth * y * 4;
        constant float *b = second + firstWidth * x * 4;

        float4 acc0 = 0;
        float4 acc1 = 0;
        float4 acc2 = 0;
        float4 acc3 = 0;

        computeMultiplyMatrixByTransposedMatrixFor4x4Block( a, b, firstWidth, acc0, acc1, acc2, acc3 );

        result += y * 4 * secondHeight + x * 4;
        result[0] += acc0.x;
        result[1] += acc0.y;
        result[2] += acc0.z;
        result[3] += acc0.w;
        result += secondHeight;
        result[0] += acc1.x;
        result[1] += acc1.y;
        result[2] += acc1.z;
        result[3] += acc1.w;
        result += secondHeight;
        result[0] += acc2.x;
        result[1] += acc2.y;
        result[2] += acc2.z;
        result[3] += acc2.w;
        result += secondHeight;
        result[0] += acc3.x;
        result[1] += acc3.y;
        result[2] += acc3.z;
        result[3] += acc3.w;
    }
}

// Multiplies a matrix by a transposed matrix using a kernel and adding to the result
// Used to calculate the elements left after matrixKernelMultiplyMatrixByTransposedMatrixThread4x4
kernel void matrixKernelMultiplyMatrixByTransposedMatrixAndAddThread4x4Borders( constant float* first [[buffer(0)]],
                                                                    constant int& firstHeight [[buffer(1)]],
                                                                    constant int& firstWidth [[buffer(2)]],
                                                                    constant float* second [[buffer(3)]],
                                                                    constant int& secondHeight[[buffer(4)]],
                                                                    constant int& leftOffset [[buffer(5)]],
                                                                    constant int& topOffset [[buffer(6)]],
                                                                    device float* result [[buffer(7)]],
                                                                    uint thread_position_in_grid [[ thread_position_in_grid ]] )
{
    const int rightBorderWidth = secondHeight - leftOffset;
    const int bottomBorderWidth = secondHeight - rightBorderWidth;
    int x, y;
    if( (int)thread_position_in_grid < rightBorderWidth * firstHeight ) {
        y = thread_position_in_grid / rightBorderWidth;
        x = leftOffset + thread_position_in_grid % rightBorderWidth;
    } else if( (int)thread_position_in_grid < rightBorderWidth * firstHeight + bottomBorderWidth * ( firstHeight - topOffset ) ) {
        const int index = thread_position_in_grid - rightBorderWidth * firstHeight;
        y = topOffset + index / bottomBorderWidth;
        x = index % bottomBorderWidth;
    } else {
        return;
    }

    int resultIndex = y * secondHeight + x;
    float res = 0;
    first += y * firstWidth;
    second += x * firstWidth;
    for( int i = 0; i < firstWidth; i++ ) {
        res += *first * *second;
        first += 1;
        second += 1;
    }
    result[resultIndex] += res;
}

kernel void cubeKernelBatchMultiplyMatrixByTransposedMatrix( constant int* batchSize [[buffer(0)]],
                                                             constant float* first [[buffer(1)]],
                                                             constant int* firstHeight [[buffer(2)]],
                                                             constant int* firstWidth [[buffer(3)]],
                                                             constant float* second [[buffer(4)]],
                                                             constant int* secondHeight [[buffer(5)]],
                                                             device float* result [[buffer(6)]],
                                                             uint3 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C3DPosition pos( thread_position_in_grid );
    int batchIndex;
    int heightIndex;
    int widthIndex;
    if( pos.GetMetalTaskIndex3D( *batchSize, *firstHeight, *secondHeight, batchIndex, heightIndex, widthIndex ) ) {
        int resultIndex = batchIndex * *firstHeight * *secondHeight + heightIndex * *secondHeight + widthIndex;
        float res = 0;
        first += batchIndex * *firstHeight * *firstWidth + heightIndex * *firstWidth;
        second += batchIndex * *firstWidth * *secondHeight + widthIndex * *firstWidth;
        for( int i = 0; i < *firstWidth; i++ ) {
            res += *first * *second;
            first += 1;
            second += 1;
        }
        result[resultIndex] = res;
    }
}

// Matrix product with a kernel
// Each thread calculates 16 elements (a 4*4 block) of the result
kernel void matrixKernelMultiplyMatrixByMatrixThread4x4( constant float* first [[buffer(0)]],
                                                        constant int& firstHeight [[buffer(1)]],
                                                        constant int& firstWidth [[buffer(2)]],
                                                        constant float* second [[buffer(3)]],
                                                        constant int& secondWidth[[buffer(4)]],
                                                        device float* result [[buffer(5)]],
                                                        uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    int x, y;

    if( pos.GetMetalTaskIndex2D( firstHeight / 4, secondWidth / 4, y, x ) ) {
        constant float *a = first + firstWidth * y * 4;
        constant float *b = second + x * 4;

        float4 acc0 = 0;
        float4 acc1 = 0;
        float4 acc2 = 0;
        float4 acc3 = 0;

        computeMultiplyMatrixByMatrixFor4x4Block( a, b, firstWidth, secondWidth, acc0, acc1, acc2, acc3 );

        result += y * 4 * secondWidth + x * 4;
        result[0] = acc0.x;
        result[1] = acc0.y;
        result[2] = acc0.z;
        result[3] = acc0.w;
        result += secondWidth;
        result[0] = acc1.x;
        result[1] = acc1.y;
        result[2] = acc1.z;
        result[3] = acc1.w;
        result += secondWidth;
        result[0] = acc2.x;
        result[1] = acc2.y;
        result[2] = acc2.z;
        result[3] = acc2.w;
        result += secondWidth;
        result[0] = acc3.x;
        result[1] = acc3.y;
        result[2] = acc3.z;
        result[3] = acc3.w;
    }
}

// Matrix product with a kernel, calculating the result with left and top offset
// Used to calculate the rest of the elements left after matrixKernelMultiplyMatrixByMatrixThread4x4.
kernel void matrixKernelMultiplyMatrixByMatrixThread4x4Borders( constant float* first [[buffer(0)]],
                                                                constant int& firstHeight [[buffer(1)]],
                                                                constant int& firstWidth [[buffer(2)]],
                                                                constant float* second [[buffer(3)]],
                                                                constant int& secondWidth[[buffer(4)]],
                                                                constant int& leftOffset [[buffer(5)]],
                                                                constant int& topOffset [[buffer(6)]],
                                                                device float* result [[buffer(7)]],
                                                                uint thread_position_in_grid [[ thread_position_in_grid ]] )
{
    const int rightBorderWidth = secondWidth - leftOffset;
    const int bottomBorderWidth = secondWidth - rightBorderWidth;
    int x, y;
    if( (int)thread_position_in_grid < rightBorderWidth * firstHeight ) {
        y = thread_position_in_grid / rightBorderWidth;
        x = leftOffset + thread_position_in_grid % rightBorderWidth;
    } else if( (int)thread_position_in_grid < rightBorderWidth * firstHeight + bottomBorderWidth * ( firstHeight - topOffset ) ) {
        const int index = thread_position_in_grid - rightBorderWidth * firstHeight;
        y = topOffset + index / bottomBorderWidth;
        x = index % bottomBorderWidth;
    } else {
        return;
    }

    int resultIndex = y * secondWidth + x;
    float res = 0;
    first += y * firstWidth;
    second += x;
    for( int i = 0; i < firstWidth; i++ ) {
        res += *first * *second;
        first += 1;
        second += secondWidth;
    }
    result[resultIndex] = res;
}

// Matrix product with a kernel, then adds a vector to each column of the result
// Each thread calculates 16 elements (a 4*4 block) of the result
kernel void matrixKernelMultiplyMatrixByMatrixAndAddVectorToColumnsThread4x4( constant float* first [[buffer(0)]],
                                                                              constant int& firstHeight [[buffer(1)]],
                                                                              constant int& firstWidth [[buffer(2)]],
                                                                              constant float* second [[buffer(3)]],
                                                                              constant int& secondWidth[[buffer(4)]],
                                                                              device float* result [[buffer(5)]],
                                                                              constant float* vector [[buffer(6)]],
                                                                              uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    int x, y;

    if( pos.GetMetalTaskIndex2D( firstHeight / 4, secondWidth / 4, y, x ) ) {
        constant float *a = first + firstWidth * y * 4;
        constant float *b = second + x * 4;

        float4 acc0 = float4(vector[y * 4]);
        float4 acc1 = float4(vector[y * 4 + 1]);
        float4 acc2 = float4(vector[y * 4 + 2]);
        float4 acc3 = float4(vector[y * 4 + 3]);

        computeMultiplyMatrixByMatrixFor4x4Block( a, b, firstWidth, secondWidth, acc0, acc1, acc2, acc3 );

        result += y * 4 * secondWidth + x * 4;
        result[0] = acc0.x;
        result[1] = acc0.y;
        result[2] = acc0.z;
        result[3] = acc0.w;
        result += secondWidth;
        result[0] = acc1.x;
        result[1] = acc1.y;
        result[2] = acc1.z;
        result[3] = acc1.w;
        result += secondWidth;
        result[0] = acc2.x;
        result[1] = acc2.y;
        result[2] = acc2.z;
        result[3] = acc2.w;
        result += secondWidth;
        result[0] = acc3.x;
        result[1] = acc3.y;
        result[2] = acc3.z;
        result[3] = acc3.w;
    }
}

// Matrix product with a kernel, then adds a vector to each column of the result
// Used to calculate the rest of the elements left after matrixKernelMultiplyMatrixByMatrixAndAddVectorToColumnsThread4x4
kernel void matrixKernelMultiplyMatrixByMatrixAndAddVectorToColumnsThread4x4Borders( constant float* first [[buffer(0)]],
                                                                constant int& firstHeight [[buffer(1)]],
                                                                constant int& firstWidth [[buffer(2)]],
                                                                constant float* second [[buffer(3)]],
                                                                constant int& secondWidth[[buffer(4)]],
                                                                constant int& leftOffset [[buffer(5)]],
                                                                constant int& topOffset [[buffer(6)]],
                                                                device float* result [[buffer(7)]],
                                                                constant float* vector [[buffer(8)]],
                                                                uint thread_position_in_grid [[ thread_position_in_grid ]] )
{
    const int rightBorderWidth = secondWidth - leftOffset;
    const int bottomBorderWidth = secondWidth - rightBorderWidth;
    int x, y;
    if( (int)thread_position_in_grid < rightBorderWidth * firstHeight ) {
        y = thread_position_in_grid / rightBorderWidth;
        x = leftOffset + thread_position_in_grid % rightBorderWidth;
    } else if( (int)thread_position_in_grid < rightBorderWidth * firstHeight + bottomBorderWidth * ( firstHeight - topOffset ) ) {
        const int index = thread_position_in_grid - rightBorderWidth * firstHeight;
        y = topOffset + index / bottomBorderWidth;
        x = index % bottomBorderWidth;
    } else {
        return;
    }

    int resultIndex = y * secondWidth + x;
    float res = vector[y];
    first += y * firstWidth;
    second += x;
    for( int i = 0; i < firstWidth; i++ ) {
        res += *first * *second;
        first += 1;
        second += secondWidth;
    }
    result[resultIndex] = res;
}

// Multiplies a matrix by a transposed matrix, with a kernel
// Each thread calculates one element of the result
kernel void matrixKernelMultiplyMatrixByTransposedMatrixThread1x1Float4( constant float4* first [[buffer(0)]],
                                                                  constant int* firstHeight [[buffer(1)]],
                                                                  constant int* firstWidth [[buffer(2)]],
                                                                  constant float4* second [[buffer(3)]],
                                                                  constant int* secondHeight[[buffer(4)]],
                                                                  device float* result [[buffer(5)]],
                                                                  uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    int heightIndex;
    int widthIndex;
    const int width = ( *firstWidth / 4 );
    if( pos.GetMetalTaskIndex2D( *firstHeight, *secondHeight, heightIndex, widthIndex ) ) {
        int resultIndex = heightIndex * *secondHeight + widthIndex;
        float res = 0;
        first += heightIndex * width;
        second += widthIndex * width;
        for( int i = 0; i < width; i++ ) {
            res += dot( *first, *second );
            first++;
            second++;
        }
        result[resultIndex] = res;
    }
}

// Multiplies a matrix by a transposed matrix, with a kernel
// Each thread calculates one element of the result
kernel void matrixKernelMultiplyMatrixByTransposedMatrixThread1x1( constant float* first [[buffer(0)]],
                                                                  constant int* firstHeight [[buffer(1)]],
                                                                  constant int* firstWidth [[buffer(2)]],
                                                                  constant float* second [[buffer(3)]],
                                                                  constant int* secondHeight[[buffer(4)]],
                                                                  device float* result [[buffer(5)]],
                                                                  uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    int heightIndex;
    int widthIndex;
    if( pos.GetMetalTaskIndex2D( *firstHeight, *secondHeight, heightIndex, widthIndex ) ) {
        int resultIndex = heightIndex * *secondHeight + widthIndex;
        float res = 0;
        first += heightIndex * *firstWidth;
        second += widthIndex * *firstWidth;
        for( int i = 0; i < *firstWidth; i++ ) {
            res += *first * *second;
            first += 1;
            second += 1;
        }
        result[resultIndex] = res;
    }
}

// Multiplies a matrix by a transposed matrix, with a kernel
// Each thread calculates 16 elements (a 4*4 block) of the result
kernel void matrixKernelMultiplyMatrixByTransposedMatrixThread4x4( constant float* first [[buffer(0)]],
                                                                  constant int& firstHeight [[buffer(1)]],
                                                                  constant int& firstWidth [[buffer(2)]],
                                                                  constant float* second [[buffer(3)]],
                                                                  constant int& secondHeight[[buffer(4)]],
                                                                  device float* result [[buffer(5)]],
                                                                  uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    int x, y;

    if( pos.GetMetalTaskIndex2D( firstHeight / 4, secondHeight / 4, y, x ) ) {
        constant float *a = first + firstWidth * y * 4;
        constant float *b = second + firstWidth * x * 4;

        float4 acc0 = 0;
        float4 acc1 = 0;
        float4 acc2 = 0;
        float4 acc3 = 0;

        computeMultiplyMatrixByTransposedMatrixFor4x4Block( a, b, firstWidth, acc0, acc1, acc2, acc3 );

        result += y * 4 * secondHeight + x * 4;
        result[0] = acc0.x;
        result[1] = acc0.y;
        result[2] = acc0.z;
        result[3] = acc0.w;
        result += secondHeight;
        result[0] = acc1.x;
        result[1] = acc1.y;
        result[2] = acc1.z;
        result[3] = acc1.w;
        result += secondHeight;
        result[0] = acc2.x;
        result[1] = acc2.y;
        result[2] = acc2.z;
        result[3] = acc2.w;
        result += secondHeight;
        result[0] = acc3.x;
        result[1] = acc3.y;
        result[2] = acc3.z;
        result[3] = acc3.w;
    }
}

// Multiplies a matrix by a transposed matrix, with a kernel
// Used to calculate the rest of the elements left after matrixKernelMultiplyMatrixByTransposedMatrixThread4x4
kernel void matrixKernelMultiplyMatrixByTransposedMatrixThread4x4Borders( constant float* first [[buffer(0)]],
                                                                    constant int& firstHeight [[buffer(1)]],
                                                                    constant int& firstWidth [[buffer(2)]],
                                                                    constant float* second [[buffer(3)]],
                                                                    constant int& secondHeight[[buffer(4)]],
                                                                    constant int& leftOffset [[buffer(5)]],
                                                                    constant int& topOffset [[buffer(6)]],
                                                                    device float* result [[buffer(7)]],
                                                                    uint thread_position_in_grid [[ thread_position_in_grid ]] )
{
    const int rightBorderWidth = secondHeight - leftOffset;
    const int bottomBorderWidth = secondHeight - rightBorderWidth;
    int x, y;
    if( (int)thread_position_in_grid < rightBorderWidth * firstHeight ) {
        y = thread_position_in_grid / rightBorderWidth;
        x = leftOffset + thread_position_in_grid % rightBorderWidth;
    } else if( (int)thread_position_in_grid < rightBorderWidth * firstHeight + bottomBorderWidth * ( firstHeight - topOffset ) ) {
        const int index = thread_position_in_grid - rightBorderWidth * firstHeight;
        y = topOffset + index / bottomBorderWidth;
        x = index % bottomBorderWidth;
    } else {
        return;
    }

    int resultIndex = y * secondHeight + x;
    float res = 0;
    first += y * firstWidth;
    second += x * firstWidth;
    for( int i = 0; i < firstWidth; i++ ) {
        res += *first * *second;
        first += 1;
        second += 1;
    }
    result[resultIndex] = res;
}

// Multiplies a transposed matrix by a matrix, with a kernel, then adds a vector to each column of the result
// Each thread calculates 1 element of the result
kernel void matrixKernelMultiplyTransposedMatrixByMatrixAndAddThread1x1( constant float* first [[buffer(0)]],
                                                                        constant int* firstHeight [[buffer(1)]],
                                                                        constant int* firstWidth [[buffer(2)]],
                                                                        constant float* second [[buffer(3)]],
                                                                        constant int* secondWidth[[buffer(4)]],
                                                                        device float* result [[buffer(5)]],
                                                                        uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    int heightIndex;
    int widthIndex;
    if( pos.GetMetalTaskIndex2D( *firstHeight, *secondWidth, heightIndex, widthIndex ) ) {
        int resultIndex = heightIndex * *secondWidth + widthIndex;
        first += heightIndex;
        second += widthIndex;
        for( int i = 0; i < *firstHeight; i++ ) {
            result[resultIndex] += *first * *second;
            first += *firstWidth;
            second += *secondWidth;
        }
    }
}

// Multiplies a transposed matrix by a matrix, with a kernel, adding to the result
// Each thread calculates 16 elements (a 4*4 block) of the result
kernel void matrixKernelMultiplyTransposedMatrixByMatrixAndAddThread4x4( constant float* first [[buffer(0)]],
                                                                        constant int& firstHeight [[buffer(1)]],
                                                                        constant int& firstWidth [[buffer(2)]],
                                                                        constant float* second [[buffer(3)]],
                                                                        constant int& secondWidth[[buffer(4)]],
                                                                        device float* result [[buffer(5)]],
                                                                        uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    int x, y;

    if( pos.GetMetalTaskIndex2D( firstWidth / 4, secondWidth / 4, y, x ) ) {
        constant float *a = first + y * 4;
        constant float *b = second + x * 4;

        float4 acc0 = 0;
        float4 acc1 = 0;
        float4 acc2 = 0;
        float4 acc3 = 0;

        computeMultiplyTransposedMatrixByMatrixFor4x4Block( a, b, firstHeight, firstWidth, secondWidth, acc0, acc1, acc2, acc3 );

        result += y * 4 * secondWidth + x * 4;
        result[0] += acc0.x;
        result[1] += acc0.y;
        result[2] += acc0.z;
        result[3] += acc0.w;
        result += secondWidth;
        result[0] += acc1.x;
        result[1] += acc1.y;
        result[2] += acc1.z;
        result[3] += acc1.w;
        result += secondWidth;
        result[0] += acc2.x;
        result[1] += acc2.y;
        result[2] += acc2.z;
        result[3] += acc2.w;
        result += secondWidth;
        result[0] += acc3.x;
        result[1] += acc3.y;
        result[2] += acc3.z;
        result[3] += acc3.w;
    }
}

// Multiplies a transposed matrix by a matrix, with a kernel, adding to the result
// Used to calculate the rest of the elements left after matrixKernelMultiplyTransposedMatrixByMatrixThread4x4
kernel void matrixKernelMultiplyTransposedMatrixByMatrixAndAddThread4x4Borders( constant float* first [[buffer(0)]],
                                                                constant int& firstHeight [[buffer(1)]],
                                                                constant int& firstWidth [[buffer(2)]],
                                                                constant float* second [[buffer(3)]],
                                                                constant int& secondWidth[[buffer(4)]],
                                                                constant int& leftOffset [[buffer(5)]],
                                                                constant int& topOffset [[buffer(6)]],
                                                                device float* result [[buffer(7)]],
                                                                uint thread_position_in_grid [[ thread_position_in_grid ]] )
{
    const int rightBorderWidth = secondWidth - leftOffset;
    const int bottomBorderWidth = secondWidth - rightBorderWidth;
    int x, y;
    if( (int)thread_position_in_grid < rightBorderWidth * firstWidth ) {
        y = thread_position_in_grid / rightBorderWidth;
        x = leftOffset + thread_position_in_grid % rightBorderWidth;
    } else if( (int)thread_position_in_grid < rightBorderWidth * firstWidth + bottomBorderWidth * ( firstWidth - topOffset ) ) {
        const int index = thread_position_in_grid - rightBorderWidth * firstWidth;
        y = topOffset + index / bottomBorderWidth;
        x = index % bottomBorderWidth;
    } else {
        return;
    }

    int resultIndex = y * secondWidth + x;
    float res = 0;
    first += y;
    second += x;
    for( int i = 0; i < firstHeight; i++ ) {
        res += *first * *second;
        first += firstWidth;
        second += secondWidth;
    }
    result[resultIndex] += res;
}

kernel void cubeKernelBatchMultiplyTransposedMatrixByMatrix( constant int* batchSize [[buffer(0)]],
                                                             constant float* first [[buffer(1)]],
                                                             constant int* firstHeight [[buffer(2)]],
                                                             constant int* firstWidth [[buffer(3)]],
                                                             constant float* second [[buffer(4)]],
                                                             constant int* secondWidth [[buffer(5)]],
                                                             device float* result [[buffer(6)]],
                                                             uint3 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C3DPosition pos( thread_position_in_grid );
    int batchIndex;
    int heightIndex;
    int widthIndex;
    if( pos.GetMetalTaskIndex3D( *batchSize, *firstWidth, *secondWidth, batchIndex, heightIndex, widthIndex ) ) {
        int resultIndex = batchIndex * *firstWidth * *secondWidth + heightIndex * *secondWidth + widthIndex;
        float res = 0;
        first += batchIndex * *firstHeight * *firstWidth + heightIndex;
        second += batchIndex * *firstHeight * *secondWidth + widthIndex;
        for( int i = 0; i < *firstHeight; i++ ) {
            res += *first * *second;
            first += *firstWidth;
            second += *secondWidth;
        }
        result[resultIndex] = res;
    }
}

kernel void matrixKernelMultiplyDiagMatrixByMatrix( constant float* first [[buffer(0)]],
                                                    constant int* firstSize [[buffer(1)]],
                                                    constant float* second [[buffer(2)]],
                                                    constant int* secondWidth [[buffer(3)]],
                                                    device float* result [[buffer(4)]],
                                                    uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
	int i;
	int j;
	if( pos.GetMetalTaskIndex2D( *firstSize, *secondWidth, j, i ) ) {
		int index = j * *secondWidth + i;
		result[index] = second[index] * first[j];
	}
}

kernel void matrixKernelMultiply1DiagMatrixByMatrix( constant int* batchSize [[buffer(0)]],
                                                     constant float* first [[buffer(1)]],
                                                     constant int* firstSize [[buffer(2)]],
                                                     constant float* second [[buffer(3)]],
                                                     constant int* secondWidth [[buffer(4)]],
                                                     device float* result [[buffer(5)]],
                                                     constant int* batchNorm [[buffer(6)]],
                                                     uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
	int b;
	int index;
	int matrixSize = *firstSize * *secondWidth;
	if( pos.GetMetalTaskIndex2D( *batchNorm, matrixSize, b, index ) ) {
        b *= 8;
        int bLast = b + 8;
        if(bLast > *batchSize) {
            bLast = *batchSize;
        }

        int count = bLast - b;

        int j = index / *secondWidth;
        index += b * matrixSize;
        result += index;
        second += index;
        float mult = first[j];

        for(int c = 0; c < count; ++c) {
            *result = mult * (*second);
            second += matrixSize;
            result += matrixSize;
        }
    }
}

kernel void cubeKernelMultiplyMatrixByMatrix( constant int* batchSize [[buffer(0)]],
                                              constant float* first [[buffer(1)]],
                                              constant int* firstHeight [[buffer(2)]],
                                              constant int* firstWidth [[buffer(3)]],
                                              constant float* second [[buffer(4)]],
                                              constant int* secondWidth [[buffer(5)]],
                                              device float* result [[buffer(6)]],
                                              uint3 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C3DPosition pos( thread_position_in_grid );
    int batchIndex;
    int heightIndex;
    int widthIndex;
    if( pos.GetMetalTaskIndex3D( *batchSize, *firstHeight, *secondWidth, batchIndex, heightIndex, widthIndex ) ) {
        int resultIndex = batchIndex * *firstHeight * *secondWidth + heightIndex * *secondWidth + widthIndex;
        float res = 0;
        first += batchIndex * *firstHeight * *firstWidth + heightIndex * *firstWidth;
        second += batchIndex * *firstWidth * *secondWidth + widthIndex;
        for( int i = 0; i < *firstWidth; i++ ) {
            res += *first * *second;
            first += 1;
            second += *secondWidth;
        }
        result[resultIndex] = res;
    }
}

kernel void matrixKernelMultiplyMatrixByDiagMatrix( constant float* first [[buffer(0)]],
                                                    constant int* firstHeight [[buffer(1)]],
                                                    constant int* firstWidth [[buffer(2)]],
                                                    constant float* second [[buffer(3)]],
                                                    device float* result [[buffer(4)]],
                                                    uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    
	int i;
	int j;
	if( pos.GetMetalTaskIndex2D( *firstHeight, *firstWidth, j, i ) ) {
		int index = j * *firstWidth + i;
		result[index] = first[index] * second[i];
	}
}

kernel void cubeKernelChannelLastBlobSpatialDropout( constant float* input [[buffer(0)]],
                                                     constant float* mask [[buffer(1)]],
                                                     device float* output [[buffer(2)]],
                                                     constant int* inputObjectCount [[buffer(3)]],
                                                     constant int* inputObjectSize [[buffer(4)]],
                                                     constant int* maskObjectCount [[buffer(5)]],
                                                     constant int* maskObjectSize [[buffer(6)]],
                                                     uint3 thread_position_in_grid [[thread_position_in_grid]] )
{
    C3DPosition pos( thread_position_in_grid );
    
	int obj;
	int row;
	int col;
	if( pos.GetMetalTaskIndex3D( *inputObjectCount, *inputObjectSize / *maskObjectSize, *maskObjectSize,
        obj, row, col ) )
    {
		int pack = obj % *maskObjectCount;
		int index = obj * *inputObjectSize + row * *maskObjectSize + col;
		output[index] = input[index] * mask[*maskObjectSize * pack + col];
	}
}

kernel void vectorKernelTransposeMatrixFloat( constant int* batchSize [[buffer(0)]],
                                              constant float* first [[buffer(1)]],
                                              constant int* height [[buffer(2)]],
                                              constant int* medium [[buffer(3)]],
                                              constant int* width [[buffer(4)]],
                                              constant int* channels [[buffer(5)]],
                                              device float* result [[buffer(6)]],
                                              constant int* size [[buffer(7)]],
                                              uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                              uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                              uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );

	int index;
	int step;
	int count = pos.GetMetalTaskCountAndIndex( *size, 8, index, step );

	for(int i = 0; i < count; ++i) {
		int resChannel = index % *channels;
		int cur = index / *channels;
		int resHeight = cur % *width;
		cur = cur / *width;
		int resMed = cur % *medium;
		cur /= *medium;
		int resWidth = cur % *height;
		int resBatch = cur / *height;

		result[(((resBatch * *width + resHeight) * *medium + resMed) * *height + resWidth) * *channels + resChannel] =
			first[index];

		index += step;
	}
}

kernel void vectorKernelTransposeMatrixInt( constant int* batchSize [[buffer(0)]],
                                            constant int* first [[buffer(1)]],
                                            constant int* height [[buffer(2)]],
                                            constant int* medium [[buffer(3)]],
                                            constant int* width [[buffer(4)]],
                                            constant int* channels [[buffer(5)]],
                                            device int* result [[buffer(6)]],
                                            constant int* size [[buffer(7)]],
                                            uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                            uint threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                            uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C1DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    
    int index;
    int step;
    int count = pos.GetMetalTaskCountAndIndex( *size, 8, index, step );
    
    for(int i = 0; i < count; ++i) {
        int resChannel = index % *channels;
        int cur = index / *channels;
        int resHeight = cur % *width;
        cur = cur / *width;
        int resMed = cur % *medium;
        cur /= *medium;
        int resWidth = cur % *height;
        int resBatch = cur / *height;
        
        result[(((resBatch * *width + resHeight) * *medium + resMed) * *height + resWidth) * *channels + resChannel] =
        first[index];
        
        index += step;
    }
}

kernel void matrixKernelMatrixSpreadRowsFloat( constant float* source [[buffer(0)]],
                                               constant int* height [[buffer(1)]],
                                               constant int* width [[buffer(2)]],
                                               device float* result [[buffer(3)]],
                                               constant int* indices [[buffer(4)]],
                                               uint2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                               uint2 threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                               uint2 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C2DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    
	int j;
	int i;
	if( pos.GetMetalTaskIndex2D( *height, *width, 1, 16, j, i ) ) {
        if(indices[j] >= 0) {
            int step;
            int count = pos.GetMetalWidthTaskCountAndIndex( *width, 16, i, step );
            source += j * *width + i;
            result += indices[j] * *width + i;
            for(int c = 0; c < count; ++c) {
                *result = *source;
                source += step;
                result += step;
            }
        }
    }
}

kernel void matrixKernelMatrixSpreadRowsInt( constant int* source [[buffer(0)]],
                                             constant int* height [[buffer(1)]],
                                             constant int* width [[buffer(2)]],
                                             device int* result [[buffer(3)]],
                                             constant int* indices [[buffer(4)]],
                                             uint2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                             uint2 threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                             uint2 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C2DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );

    int j;
    int i;
    if( pos.GetMetalTaskIndex2D( *height, *width, 1, 16, j, i ) ) {
        if(indices[j] >= 0) {
            int step;
            int count = pos.GetMetalWidthTaskCountAndIndex( *width, 16, i, step );
            source += j * *width + i;
            result += indices[j] * *width + i;
            for(int c = 0; c < count; ++c) {
                *result = *source;
                source += step;
                result += step;
            }
        }
    }
}

kernel void matrixKernelMatrixSpreadRowsAdd( constant float* source [[buffer(0)]],
                                             constant int* height [[buffer(1)]],
                                             constant int* width [[buffer(2)]],
                                             device float* result [[buffer(3)]],
                                             constant int* resultHeight [[buffer(4)]],
                                             constant int* indices [[buffer(5)]],
                                             uint2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                             uint2 threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                             uint2 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C2DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );

	int xPos;
	int yPos;
	if( !pos.GetMetalTaskIndex2D( *resultHeight, *width, 1, 1, yPos, xPos ) ) {
        return;
    }
    
    for( int i = 0; i < *height; i++ ) {
        if( indices[i] == yPos ) {
            result[yPos * *width + xPos] += source[i * *width + xPos];
        }
    }
}

kernel void matrixKernelMultiplyDiagMatrixByMatrixAndAdd( constant int* batchSize [[buffer(0)]],
                                                          constant float* first [[buffer(1)]],
                                                          constant int* firstSize [[buffer(2)]],
                                                          constant float* second [[buffer(3)]],
                                                          constant int* secondWidth [[buffer(4)]],
                                                          device float* result [[buffer(5)]],
                                                          uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    int heightIndex;
    int widthIndex;
    if( pos.GetMetalTaskIndex2D( *firstSize, *secondWidth, heightIndex, widthIndex ) ) {
        int resultIndex = heightIndex * *secondWidth + widthIndex;
        first += heightIndex;
        second += heightIndex * *secondWidth + widthIndex;
        for( int batchIndex = 0; batchIndex < *batchSize; batchIndex++ ) {
            result[resultIndex] += *first * *second;
            first += *firstSize;
            second += *firstSize * *secondWidth;
        }
    }
}

kernel void matrixKernelMatrixLogSumExpByRows( constant float* matrix [[buffer(0)]],
                                               constant int* height [[buffer(1)]],
                                               constant int* width [[buffer(2)]],
                                               device float* result [[buffer(3)]],
                                               threadgroup float* buffer [[threadgroup(4)]],
                                               uint2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                               uint2 threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                               uint2 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    threadgroup float& my = buffer[thread_position_in_threadgroup.y * threads_per_threadgroup.x + thread_position_in_threadgroup.x];
    my = -FLT_MAX;
    
    C2DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );

	int xPos;
	int yPos;
    int step;
    int count = 0;
	if( pos.GetMetalTaskIndex2D( *height, *width, 1, 1, yPos, xPos ) ) {
        matrix += yPos * *width;
        
        int combineCount = ( *width + threads_per_threadgroup.x - 1 ) / threads_per_threadgroup.x;
        count = pos.GetMetalWidthTaskCountAndIndex( *width, combineCount, xPos, step );
        my = matrix[xPos];
        for(int i = 1; i < count; ++i) {
            float value = matrix[xPos + i * step];
            if( value > my ) {
                my = value;
            }
        }
    }
    
    Reduce2DMax( thread_position_in_threadgroup, threads_per_threadgroup, buffer );
    
    const float maxVal = buffer[thread_position_in_threadgroup.y * threads_per_threadgroup.x];
    
    my = 0;
	for(int i = 0; i < count; ++i) {
		my += ExponentFunc(matrix[xPos + i * step] - maxVal);
	}

    Reduce2DSum( thread_position_in_threadgroup, threads_per_threadgroup, buffer );

	if( thread_position_in_threadgroup.x == 0 ) {
        result[yPos] = maxVal + LogFunc(my);
	}
}

kernel void matrixKernelMatrixLogSumExpByColumns( constant float* matrix [[buffer(0)]],
                                                  constant int* height [[buffer(1)]],
                                                  constant int* width [[buffer(2)]],
                                                  device float* result [[buffer(3)]],
                                                  threadgroup float* buffer [[threadgroup(4)]],
                                                  uint2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                                  uint2 threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                                  uint2 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    threadgroup float& my = buffer[thread_position_in_threadgroup.y * threads_per_threadgroup.x + thread_position_in_threadgroup.x];
    my = -FLT_MAX;
    
    C2DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    
    int xPos;
    int yPos;
    int step;
    int count = 0;
    if( pos.GetMetalTaskIndex2D( *height, *width, 1, 1, yPos, xPos ) ) {
        int combine = ( *height + threads_per_threadgroup.y - 1) / threads_per_threadgroup.y;
        count = pos.GetMetalHeightTaskCountAndIndex( *height, combine, yPos, step );
        matrix += xPos;
        yPos *= *width;
        step *= *width;
        my = matrix[yPos];
        for(int i = 1; i < count; ++i) {
            float value = matrix[yPos + i * step];
            if( value > my ) {
                my = value;
            }
        }
    }
    
    Reduce2DMaxTrans( thread_position_in_threadgroup, threads_per_threadgroup, buffer );
    
    const float maxVal = buffer[thread_position_in_threadgroup.x];
 
    my = 0;
    for(int i = 0; i < count; ++i) {
        my += ExponentFunc(matrix[yPos + i * step] - maxVal);
    }
    
    Reduce2DSumTrans( thread_position_in_threadgroup, threads_per_threadgroup, buffer );
    
    if( thread_position_in_threadgroup.y == 0 ) {
        result[xPos] = maxVal + LogFunc(my);
    }
}

kernel void matrixKernelMatrixSoftmaxByRows( constant float* matrix [[buffer(0)]],
                                             constant int* height [[buffer(1)]],
                                             constant int* width [[buffer(2)]],
                                             device float* result [[buffer(3)]],
                                             threadgroup float* buffer [[threadgroup(4)]],
                                             uint2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                             uint2 threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                             uint2 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    threadgroup float& my = buffer[thread_position_in_threadgroup.y * threads_per_threadgroup.x + thread_position_in_threadgroup.x];
    my = -FLT_MAX;
    
    C2DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    
    int xPos;
    int yPos;
    int step;
    int count = 0;
    if( pos.GetMetalTaskIndex2D( *height, *width, 1, 1, yPos, xPos ) ) {
        matrix += yPos * *width;
        result += yPos * *width;
        
        int combine = ( *width + threads_per_threadgroup.x - 1 ) / threads_per_threadgroup.x;
        count = pos.GetMetalWidthTaskCountAndIndex( *width, combine, xPos, step );
        my = matrix[xPos];
        for(int i = 1; i < count; ++i) {
            float value = matrix[xPos + i * step];
            if( value > my ) {
                my = value;
            }
        }
    }
    
    Reduce2DMax( thread_position_in_threadgroup, threads_per_threadgroup, buffer );
    
    const float maxVal = buffer[thread_position_in_threadgroup.y * threads_per_threadgroup.x];
    
    my = 0;
    for( int i = 0; i < count; ++i ) {
        float val = ExponentFunc(matrix[xPos + i * step] - maxVal);
        result[xPos + i * step] = val;
        my += val;
    }
    
    Reduce2DSum( thread_position_in_threadgroup, threads_per_threadgroup, buffer );
    
    for( int i = 0; i < count; ++i ) {
        result[xPos + i * step] *= 1.f / buffer[thread_position_in_threadgroup.y * threads_per_threadgroup.x];
    }
}

kernel void matrixKernelMatrixSoftmaxDiffOpByRows( constant float* first [[buffer(0)]],
                                                   constant float* second [[buffer(1)]],
                                                   constant int* height [[buffer(2)]],
                                                   constant int* width [[buffer(3)]],
                                                   device float* result [[buffer(4)]],
                                                   threadgroup float* buffer [[threadgroup(5)]],
                                                   uint2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                                   uint2 threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                                   uint2 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    threadgroup float& my = buffer[thread_position_in_threadgroup.y * threads_per_threadgroup.x + thread_position_in_threadgroup.x];
    my = 0;
    
    C2DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    
    int xPos;
    int yPos;
    if( !pos.GetMetalTaskIndex2D( *height, *width, 1, 1, yPos, xPos ) ) {
        return;
    }
    
    first += yPos * *width;
    second += yPos * *width;
    result += yPos * *width;
    
    int combine = ( *width + threads_per_threadgroup.x - 1 ) / threads_per_threadgroup.x;
    int step;
    int count = pos.GetMetalWidthTaskCountAndIndex( *width, combine, xPos, step );
    for( int i = 0; i < count; ++i ) {
        my += first[xPos + i * step] * second[xPos + i * step];
    }
    
    Reduce2DSum( thread_position_in_threadgroup, threads_per_threadgroup, buffer );
    
    const float dotProd = buffer[thread_position_in_threadgroup.y * threads_per_threadgroup.x];
    
    for( int i = 0; i < count; ++i ) {
        result[xPos + i * step] = first[xPos + i * step] * ( second[xPos + i * step] - dotProd );
    }
}

kernel void matrixKernelMatrixSoftmaxByColumns( constant float* matrix [[buffer(0)]],
                                                constant int* height [[buffer(1)]],
                                                constant int* width [[buffer(2)]],
                                                device float* result [[buffer(3)]],
                                                threadgroup float* buffer [[threadgroup(4)]],
                                                uint2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                                uint2 threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                                uint2 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]]  )
{
    threadgroup float& my = buffer[thread_position_in_threadgroup.y * threads_per_threadgroup.x + thread_position_in_threadgroup.x];
    my = -FLT_MAX;
    C2DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    
    int xPos;
    int yPos;
    int step;
    int count = 0;
    if( pos.GetMetalTaskIndex2D( *height, *width, 1, 1, yPos, xPos ) ) {
        int combine = ( *height + threads_per_threadgroup.y - 1 ) / threads_per_threadgroup.y;
        count = pos.GetMetalHeightTaskCountAndIndex( *height, combine, yPos, step );
        matrix += xPos;
        result += xPos;
        yPos *= *width;
        step *= *width;
        
        my = matrix[yPos];
        for( int i = 1; i < count; ++i ) {
            float value = matrix[yPos + i * step];
            if( value > my ) {
                my = value;
            }
        }
    }
    
    Reduce2DMaxTrans( thread_position_in_threadgroup, threads_per_threadgroup, buffer );
    
    const float maxVal = buffer[thread_position_in_threadgroup.x];
    
    my = 0;
    for( int i = 0; i < count; ++i ) {
        float value = ExponentFunc( matrix[yPos + i * step] - maxVal );
        result[yPos + i * step] = value;
        my += value;
    }
    
    Reduce2DSumTrans( thread_position_in_threadgroup, threads_per_threadgroup, buffer );
    
    const float sumVal = 1.f / buffer[thread_position_in_threadgroup.x];
    
    for( int i = 0; i < count; ++i ) {
        result[yPos + i * step] *= sumVal;
    }
}

kernel void matrixKernelMatrixSoftmaxDiffOpByColumns( constant float* first [[buffer(0)]],
                                                      constant float* second [[buffer(1)]],
                                                      constant int* height [[buffer(2)]],
                                                      constant int* width [[buffer(3)]],
                                                      device float* result [[buffer(4)]],
                                                      threadgroup float* buffer [[threadgroup(5)]],
                                                      uint2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                                      uint2 threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                                      uint2 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    threadgroup float& my = buffer[thread_position_in_threadgroup.y * threads_per_threadgroup.x + thread_position_in_threadgroup.x];
    my = 0;
    C2DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    
    int xPos;
    int yPos;
    if( !pos.GetMetalTaskIndex2D( *height, *width, 1, 1, yPos, xPos ) ) {
        return;
    }
    
   int combine = ( *height + threads_per_threadgroup.y - 1 ) / threads_per_threadgroup.y;
   int step;
   int count = pos.GetMetalHeightTaskCountAndIndex( *height, combine, yPos, step );
   
   first += xPos;
   second += xPos;
   result += xPos;
   yPos *= *width;
   step *= *width;
    
   for( int i = 0; i < count; ++i ) {
       my += first[yPos + i * step] * second[yPos + i * step];
   }
   
   Reduce2DSumTrans( thread_position_in_threadgroup, threads_per_threadgroup, buffer );

   const float dotProd = buffer[thread_position_in_threadgroup.x];
    
   for( int i = 0; i < count; ++i ) {
       result[yPos + i * step] = first[yPos + i * step] * ( second[yPos + i * step] - dotProd );
   }
}

kernel void matrixKernelMultiplyLookupMatrixByLookupVector( constant int* batchSize [[buffer(0)]],
                                                            constant float* matrixTable [[buffer(1)]],
                                                            constant int* matrixHeight [[buffer(2)]],
                                                            constant int* matrixWidth [[buffer(3)]],
                                                            constant int* rows [[buffer(4)]],
                                                            constant float* vectorTable [[buffer(5)]],
                                                            constant int* vector [[buffer(6)]],
                                                            device float* result [[buffer(7)]],
                                                            threadgroup float* buffer [[threadgroup(8)]],
                                                            uint2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                                            uint2 threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                                            uint2 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    threadgroup float& my = buffer[thread_position_in_threadgroup.y * threads_per_threadgroup.x + thread_position_in_threadgroup.x];
    my = 0;
    C2DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    
    int batchAndRow;
    int column;
    if( !pos.GetMetalTaskIndex2D( *batchSize * *matrixHeight, *matrixWidth, 1, 1, batchAndRow, column) ) {
        return;
    }
    
    int batch = batchAndRow / *matrixHeight;
    
    int combine = ( *matrixWidth + threads_per_threadgroup.x - 1 ) / threads_per_threadgroup.x;
    int step;
    int count = pos.GetMetalWidthTaskCountAndIndex( *matrixWidth, combine, column, step );
    for( int i = 0; i < count; ++i ) {
        my += matrixTable[rows[batchAndRow] * *matrixWidth + column] * vectorTable[vector[batch] * *matrixWidth + column];
        column += step;
    }
    
    Reduce2DSum( thread_position_in_threadgroup, threads_per_threadgroup, buffer );
    
    if( thread_position_in_threadgroup.x == 0 ) {
        result[batchAndRow] = my;
    }
}

kernel void matrixKernelMultiplyTransposedLookupMatrixByVector( constant int* batchSize [[buffer(0)]],
                                                                constant float* matrixTable [[buffer(1)]],
                                                                constant int* matrixHeight [[buffer(2)]],
                                                                constant int* matrixWidth [[buffer(3)]],
                                                                constant int* rows [[buffer(4)]],
                                                                constant float* vector [[buffer(5)]],
                                                                device float* result [[buffer(6)]],
                                                                threadgroup float* buffer [[threadgroup(7)]],
                                                                uint2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                                                uint2 threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                                                uint2 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]]  )
{
    threadgroup float& my = buffer[thread_position_in_threadgroup.y * threads_per_threadgroup.x + thread_position_in_threadgroup.x];
    my = 0;
    C2DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    
    int batchAndRow;
    int column;
    if( !pos.GetMetalTaskIndex2D( *batchSize * *matrixWidth, *matrixHeight, 1, 1, batchAndRow, column) ) {
        return;
    }
    
    int batch = batchAndRow / *matrixWidth;
    int row = batchAndRow % *matrixWidth;
    
    int combine = ( *matrixHeight + threads_per_threadgroup.x - 1 ) / threads_per_threadgroup.x;
    int step;
    int count = pos.GetMetalWidthTaskCountAndIndex( *matrixHeight, combine, column, step );
    for( int i = 0; i < count; ++i ) {
        my += matrixTable[rows[batch * *matrixHeight + column] * *matrixWidth + row] * vector[batch * *matrixHeight + column];
        column += step;
    }
    
    Reduce2DSum( thread_position_in_threadgroup, threads_per_threadgroup, buffer );
    
    if( thread_position_in_threadgroup.x == 0 ) {
        result[batchAndRow] = my;
    }
}

kernel void matrixKernelMultiplyVectorByTransposedLookupVectorAndAddToTable( constant int* batchSize [[buffer(0)]],
                                                                             device float* table [[buffer(1)]],
                                                                             constant int* vectorCount [[buffer(2)]],
                                                                             constant int* vectorSize [[buffer(3)]],
                                                                             constant int* tableIndices [[buffer(4)]],
                                                                             constant float* first [[buffer(5)]],
                                                                             constant int* firstSize [[buffer(6)]],
                                                                             constant float* secondTable [[buffer(7)]],
                                                                             constant int* secondIndices [[buffer(8)]],
                                                                             uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
	int yPos;
	int xPos;
	if( pos.GetMetalTaskIndex2D( *vectorCount, *vectorSize, yPos, xPos) ) {
        for( int b = 0; b < *batchSize; b++ ) {
            for( int i = 0; i < *firstSize; i++ ) {
                if( tableIndices[b * *firstSize + i] == yPos ) {
                    table[yPos * *vectorSize + xPos] += first[b * *firstSize + i] * secondTable[secondIndices[b] * *vectorSize + xPos];
                }
            }
        }
	}
}

kernel void matrixKernelAddVectorToMatrixRows( constant int* batchSize [[buffer(0)]],
                                               constant float* matrix [[buffer(1)]],
                                               device float* result [[buffer(2)]],
                                               constant int* matrixHeight [[buffer(3)]],
                                               constant int* matrixWidth [[buffer(4)]],
                                               constant float* vector [[buffer(5)]],
                                               uint2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                               uint2 threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                               uint2 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C2DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );

	int xPos;
	int yPos;
	if( pos.GetMetalTaskIndex2D( *batchSize * *matrixHeight, *matrixWidth, 1, 4, yPos, xPos ) ) {
		int matrixBaseIndex = yPos * *matrixWidth;
		int batch = yPos / *matrixHeight;
		int vectorBaseIndex = batch * *matrixWidth;

		int index;
		int step;
		int count = pos.GetMetalWidthTaskCountAndIndex( *matrixWidth, 4, index, step );

		for(int i = 0; i < count; ++i) {
			int matrixIndex = matrixBaseIndex + index;
			result[matrixIndex] = matrix[matrixIndex] + vector[vectorBaseIndex + index];
			index += step;
		}
	}
}

kernel void matrixKernelSumMatrixRowsAdd( constant int* batchSize [[buffer(0)]],
                                          constant float* matrix [[buffer(1)]],
                                          device float* result [[buffer(2)]],
                                          constant int* matrixHeight [[buffer(3)]],
                                          constant int* matrixWidth [[buffer(4)]],
                                          uint2 thread_position_in_grid [[ thread_position_in_grid ]]  )
{
    C2DPosition pos( thread_position_in_grid );
    
    int b;
    int x;
    if( pos.GetMetalTaskIndex2D( *batchSize, *matrixWidth, b, x ) ) {
        matrix += b * *matrixHeight * *matrixWidth + x;
        result += b * *matrixWidth + x;
        
        float res = 0;
        for( int i = 0; i < *matrixHeight; ++i ) {
            res += *matrix;
            matrix += *matrixWidth;
        }
        *result = res;
    }
}

kernel void matrixKernelMultiplySparseMatrixByTransposedMatrix( constant int* firstRows [[buffer(0)]],
                                                                constant int* firstColumns [[buffer(1)]],
                                                                constant float* firstValues [[buffer(2)]],
                                                                constant float* second [[buffer(3)]],
                                                                constant int& firstHeight [[buffer(4)]],
                                                                constant int& firstWidth [[buffer(5)]],
                                                                constant int& secondHeight [[buffer(6)]],
                                                                device float* result [[buffer(7)]],
                                                                uint2 thread_position_in_grid [[ thread_position_in_grid ]]  )
{
    C2DPosition pos( thread_position_in_grid );
    
    int col;
    int row;
    if( pos.GetMetalTaskIndex2D( secondHeight, firstHeight, col, row ) ) {
        float resultVal = 0;
		int resultIndex = row * secondHeight + col;
		int secondRowIndex = firstWidth * col;

		for( int ind = firstRows[row]; ind < firstRows[row + 1]; ++ind ) {
			resultVal += firstValues[ind] * second[secondRowIndex + firstColumns[ind]];
		}

		result[resultIndex] = resultVal;
    }
}

kernel void matrixKernelMultiplyTransposedMatrixBySparseMatrix( constant float* first [[buffer(0)]],
                                                                constant int* secondRows [[buffer(1)]],
                                                                constant int* secondColumns [[buffer(2)]],
                                                                constant float* secondValues [[buffer(3)]],
                                                                constant int& firstHeight [[buffer(4)]],
                                                                constant int& firstWidth [[buffer(5)]],
                                                                constant int& secondWidth [[buffer(6)]],
                                                                device float* result [[buffer(7)]],
                                                                uint thread_position_in_grid [[ thread_position_in_grid ]]  )
{
	C1DPosition pos( thread_position_in_grid );
    
	int col;
	if( pos.GetMetalTaskIndex( firstWidth, col ) ) {
		for( int row = 0; row < firstHeight; ++row ) {
			for( int ind = secondRows[row]; ind < secondRows[row + 1]; ++ind ) {
				result[col * secondWidth + secondColumns[ind]] += first[row * firstWidth + col] * secondValues[ind];
			}
		}
	}
}
