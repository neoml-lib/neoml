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

struct CBaseBlobDesc {
    int BatchLength; // maximum sequence length for a recurrent network
    int BatchWidth; // the number of sequences in the blob
    int ListSize; // the number of objects in the list
    int Height;        // image height
    int Width;        // image width
    int Depth;        // image depth
    int Channels;    // the number of channels

    int BlobSize() constant { return BatchLength * BatchWidth * ListSize * Height * Width * Depth * Channels; }
    int ObjectSize() constant { return Height * Width * Depth * Channels; }
    int BatchSize() constant { return BatchLength * BatchWidth * ListSize; }
};
    
inline int GetBlobPos( constant CBaseBlobDesc& blob, int d0, int d1, int d2 )
{
    return (((d0 * blob.Height) + d1) * blob.Width + d2) * blob.Depth * blob.Channels;
}
    
inline constant float* GetFloatBlobPtr( constant CBaseBlobDesc& blob, constant float* data, int d0, int d1, int d2 )
{
    return data + GetBlobPos(blob, d0, d1, d2);
}

inline device float* GetFloatBlobPtr( constant CBaseBlobDesc& blob, device float* data, int d0, int d1, int d2 )
{
    return data + GetBlobPos(blob, d0, d1, d2);
}
    
inline constant int* GetIntBlobPtr( constant CBaseBlobDesc& blob, constant int* data, int d0, int d1, int d2 )
{
    return data + GetBlobPos(blob, d0, d1, d2);
}
    
inline device int* GetIntBlobPtr( constant CBaseBlobDesc& blob, device int* data, int d0, int d1, int d2 )
{
    return data + GetBlobPos(blob, d0, d1, d2);
}
  
//-------------------------------------------------------------------------------------------------------------------

kernel void cubeKernelBlobMaxPooling( constant CBaseBlobDesc* source [[buffer(0)]],
                                      constant float* sourceData [[buffer(1)]],
                                      constant int* filterHeight [[buffer(2)]],
                                      constant int* filterWidth [[buffer(3)]],
                                      constant int* strideHeight [[buffer(4)]],
                                      constant int* strideWidth [[buffer(5)]],
                                      constant CBaseBlobDesc* result [[buffer(6)]],
                                      device float* resultData [[buffer(7)]],
                                      device int* maxIndices [[buffer(8)]],
                                      uint3 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C3DPosition pos( thread_position_in_grid );

	int totalChannels = result->Depth * result->Channels;

	int num, j, i, channel;
	int sourceRowSize;
	int sourceItemSize;

	constant float* sourcePtr;
	int resultPos;

	int startIndexPos;

    if( !pos.GetMetalTaskIndex3D( result->BatchSize(), result->Height, result->Width * totalChannels, num, j, i ) ) {
        return;
    }

    channel = i % totalChannels;
    i /= totalChannels;

    sourceRowSize = source->Width * totalChannels;
    sourceItemSize = totalChannels;

    int sourceJ = j * *strideHeight;
    int sourceI = i * *strideWidth;

    sourcePtr = GetFloatBlobPtr( *source, sourceData, num, sourceJ, sourceI ) + channel;
    resultPos = GetBlobPos( *result, num, j, i ) + channel;

    startIndexPos = GetBlobPos( *source, 0, sourceJ, sourceI ) + channel;

	float resultValue = -FLT_MAX;
	int index = startIndexPos;
	
	for(int jStep = 0; jStep < *filterHeight; ++jStep) {
		constant float* sourceItemPtr = sourcePtr;
		for(int iStep = 0; iStep < *filterWidth; ++iStep) {
			float value = *sourceItemPtr;
			if(resultValue < value) {
				resultValue = value;
				index = startIndexPos + iStep * sourceItemSize;
			}
			sourceItemPtr += sourceItemSize;
		}
		sourcePtr += sourceRowSize;
		startIndexPos += sourceRowSize;
	}

	resultData[resultPos] = resultValue;
	maxIndices[resultPos] = index;
}
    
kernel void cubeKernelBlobMaxPoolingNoIndices( constant CBaseBlobDesc* source [[buffer(0)]],
                                               constant float* sourceData [[buffer(1)]],
                                               constant int* filterHeight [[buffer(2)]],
                                               constant int* filterWidth [[buffer(3)]],
                                               constant int* strideHeight [[buffer(4)]],
                                               constant int* strideWidth [[buffer(5)]],
                                               constant CBaseBlobDesc* result [[buffer(6)]],
                                               device float* resultData [[buffer(7)]],
                                               uint3 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C3DPosition pos( thread_position_in_grid );
    
    int totalChannels = result->Depth * result->Channels;
    
    int num, j, i, channel;
    int sourceRowSize;
    int sourceItemSize;
    
    constant float* sourcePtr;
    int resultPos;
    
    int startIndexPos;
    
    if( !pos.GetMetalTaskIndex3D( result->BatchSize(), result->Height, result->Width * totalChannels, num, j, i ) ) {
        return;
    }
    
    channel = i % totalChannels;
    i /= totalChannels;
    
    sourceRowSize = source->Width * totalChannels;
    sourceItemSize = totalChannels;
    
    int sourceJ = j * *strideHeight;
    int sourceI = i * *strideWidth;
    
    sourcePtr = GetFloatBlobPtr( *source, sourceData, num, sourceJ, sourceI ) + channel;
    resultPos = GetBlobPos( *result, num, j, i ) + channel;
    
    startIndexPos = GetBlobPos( *source, 0, sourceJ, sourceI ) + channel;
    
    float resultValue = -FLT_MAX;
    int index = startIndexPos;
    
    for(int jStep = 0; jStep < *filterHeight; ++jStep) {
        constant float* sourceItemPtr = sourcePtr;
        for(int iStep = 0; iStep < *filterWidth; ++iStep) {
            float value = *sourceItemPtr;
            if(resultValue < value) {
                resultValue = value;
                index = startIndexPos + iStep * sourceItemSize;
            }
            sourceItemPtr += sourceItemSize;
        }
        sourcePtr += sourceRowSize;
        startIndexPos += sourceRowSize;
    }
    
    resultData[resultPos] = resultValue;
}

//-------------------------------------------------------------------------------------------------------------------
// Mean pooling

kernel void cubeKernelBlobMeanPooling( constant CBaseBlobDesc* source [[buffer(0)]],
                                       constant float* sourceData [[buffer(1)]],
                                       constant int* filterHeight [[buffer(2)]],
                                       constant int* filterWidth [[buffer(3)]],
                                       constant int* strideHeight [[buffer(4)]],
                                       constant int* strideWidth [[buffer(5)]],
                                       constant CBaseBlobDesc* result [[buffer(6)]],
                                       device float* resultData [[buffer(7)]],
                                       uint3 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C3DPosition pos( thread_position_in_grid );
    
	const int totalChannels = result->Depth * result->Channels;
	const int sourceRowSize = source->Width * totalChannels;

    int b;
    int j;
    int ic;
    if( !pos.GetMetalTaskIndex3D(result->BatchSize(), result->Height, result->Width * totalChannels, b, j, ic) ) {
        return;
    }

    int channel = ic % totalChannels;
    int i = ic / totalChannels;

    constant float* sourcePtr = GetFloatBlobPtr(*source, sourceData, b, j * *strideHeight, i * *strideWidth) + channel;
    device float* resultPtr = GetFloatBlobPtr(*result, resultData, b, j, i) + channel;

    float resultValue = 0;

	for(int jStep = 0; jStep < *filterHeight; ++jStep) {
		constant float* sourceItemPtr = sourcePtr;
		for(int iStep = 0; iStep < *filterWidth; ++iStep) {
			resultValue+= *sourceItemPtr;
			sourceItemPtr += totalChannels;
		}
		sourcePtr += sourceRowSize;
	}

	*resultPtr = resultValue / (*filterHeight * *filterWidth);
}
    
//-------------------------------------------------------------------------------------------------------------------
// Global max pooling

inline void MergeBuffers( int maxCount, threadgroup float* buffer, threadgroup float* maxIndexBuffer,
	threadgroup float* buffer0, threadgroup float* maxIndexBuffer0, threadgroup float* buffer1, threadgroup float* maxIndexBuffer1 )
{
	while( maxCount-- > 0 ) {
		if( *buffer0 > *buffer1 ) {
			*buffer++ = *buffer0++;
			*maxIndexBuffer++ = *maxIndexBuffer0++;
		} else {
			*buffer++ = *buffer1++;
			*maxIndexBuffer++ = *maxIndexBuffer1++;
		}
	}
}
    
kernel void matrixKernelBlobGlobalMaxPooling( constant CBaseBlobDesc& source [[buffer(0)]],
                                              constant float* sourceData [[buffer(1)]],
                                              constant CBaseBlobDesc& maxIndices [[buffer(2)]],
                                              device int* maxIndicesData [[buffer(3)]],
                                              constant CBaseBlobDesc& result [[buffer(4)]],
                                              device float* resultData [[buffer(5)]],
                                              constant int& poolSize [[buffer(6)]],
                                              constant int& maxCount [[buffer(7)]],
                                              threadgroup float* sharedData [[threadgroup(8)]],
                                              uint2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                              uint2 threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                              uint2 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C2DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    
	int bufferStep = 2 * maxCount;
    int bufferIndex = thread_position_in_threadgroup.y * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
	threadgroup float* buffer = sharedData + bufferIndex * bufferStep;
	threadgroup float* maxIndexBuffer = buffer + maxCount;
	// The pointer to the end of the buffer (everything else will be used when merging)
	threadgroup float* bufferEnd = sharedData + threads_per_threadgroup.y * threads_per_threadgroup.x * bufferStep;

	for( int i = 0; i < maxCount; ++i ) {
		buffer[i] = -FLT_MAX;
		maxIndexBuffer[i] = -1.f;
	}

	// Find the position and the other indices
	int bc;
	int index;
	if( pos.GetMetalTaskIndex2D( source.BatchSize() * source.Channels, poolSize, 1, 1, bc, index ) ) {
		int b = bc / source.Channels;
		int c = bc - b * source.Channels;
		// Find maximum values among the 'combine' values and put them into the buffer
		int combine = ( poolSize + threads_per_threadgroup.x - 1 ) / threads_per_threadgroup.x;
		int step;
		int count = pos.GetMetalWidthTaskCountAndIndex( poolSize, combine, index, step );

		sourceData += b * poolSize * source.Channels + index * source.Channels + c;
		for( int i = 0; i < count; ++i ) {
			float nextValue = *sourceData;
			float nextIndex = (float)index;
			for(int j = 0; j < maxCount; ++j) {
				if(nextValue >= buffer[j]) {
					float preValue = buffer[j];
					float preIndex = maxIndexBuffer[j];
					buffer[j] = nextValue;
					maxIndexBuffer[j] = nextIndex;
					nextValue = preValue;
					nextIndex = preIndex;
				}
			}

			index += step;
			sourceData += step * source.Channels;
		}
	}

    uint s = 1;
    while( s * 2 < threads_per_threadgroup.x ) {
        s = s * 2;
    }
    
    for( uint i = s; i >= 1; i = i >> 1 ) {
        if( thread_position_in_threadgroup.x < i && thread_position_in_threadgroup.x + i < threads_per_threadgroup.x ) {
            threadgroup float* resultBuffer = bufferEnd + bufferStep * bufferIndex;
            threadgroup float* resultMaxIndexBuffer = resultBuffer + maxCount;
            MergeBuffers( maxCount, resultBuffer, resultMaxIndexBuffer, buffer, maxIndexBuffer, buffer + i * bufferStep, maxIndexBuffer + i * bufferStep );
            for(int j = 0; j < maxCount; ++j) {
                buffer[j] = resultBuffer[j];
                maxIndexBuffer[j] = resultMaxIndexBuffer[j];
            }
        }
    }
    
	if( thread_position_in_threadgroup.x == 0 ) {
        int channelNum = bc % source.Channels;
        int batchNum = bc / source.Channels;
        if( batchNum < result.BatchSize() && channelNum < result.Channels ) {
			device float* resultDataPtr = GetFloatBlobPtr( result, resultData, batchNum, 0, 0 ) + channelNum;
			device int* maxIndexDataPtr = GetIntBlobPtr( maxIndices, maxIndicesData, batchNum, 0, 0 ) + channelNum;
			for(int i = 0; i < maxCount; ++i) {
				*resultDataPtr = *buffer++;
				*maxIndexDataPtr = *maxIndexBuffer++;

				resultDataPtr += result.Channels;
				maxIndexDataPtr += maxIndices.Channels;
			}
		}
	}
}

//-------------------------------------------------------------------------------------------------------------------
// 3D max pooling
     
kernel void cubeKernelBlob3dMaxPooling( constant CBaseBlobDesc& source [[buffer(0)]],
                                        constant float* sourceData [[buffer(1)]],
                                        constant int& filterHeight [[buffer(2)]],
                                        constant int& filterWidth [[buffer(3)]],
                                        constant int& filterDepth [[buffer(4)]],
                                        constant int& strideHeight [[buffer(5)]],
                                        constant int& strideWidth [[buffer(6)]],
                                        constant int& strideDepth [[buffer(7)]],
                                        constant CBaseBlobDesc& result [[buffer(8)]],
                                        device float* resultData [[buffer(9)]],
                                        device int* maxIndices [[buffer(10)]],
                                        uint3 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C3DPosition pos( thread_position_in_grid );

	int b;
	int channel;
	int p;

	int resultGeomSize = result.Depth * result.Height * result.Width;

	if( !pos.GetMetalTaskIndex3D( result.BatchSize(), result.Channels, resultGeomSize, b, channel, p ) ) {
		return;
	}

	int inputWDC = source.Depth * source.Width * source.Channels;
	sourceData += b * source.ObjectSize();

	// Output position
	int xOut = p % result.Width;
	p /= result.Width;
	int yOut = p % result.Height;
	int zOut = p / result.Height;

    int resultShift = b * result.ObjectSize() + yOut * result.Width * result.Depth * result.Channels + 
    	xOut * result.Depth * result.Channels + zOut * result.Channels + channel;

	// Input position
	int xStart = xOut * strideWidth;
	int yStart = yOut * strideHeight;
	int zStart = zOut * strideDepth;

	float maxValue = -FLT_MAX;
	int maxIndex = 0;

    int iIndex = yStart * inputWDC + channel;
    for(int i = 0; i < filterHeight; i++) {
    	int jIndex = iIndex + xStart * source.Depth * source.Channels;
    	for(int j = 0; j < filterWidth; j++) {
    		int index = jIndex + zStart * source.Channels;
    		for(int k = 0; k < filterDepth; k++) {
    			float value = sourceData[index];
                if(value >= maxValue) {
                	maxIndex = index;
                    maxValue = value;
                }
                index += source.Channels;
    		}
    		jIndex += source.Depth * source.Channels;
    	}
    	iIndex += inputWDC;
    }

	resultData[resultShift] = maxValue;
    maxIndices[resultShift] = maxIndex;
}
    
kernel void cubeKernelBlob3dMaxPoolingNoIndices( constant CBaseBlobDesc& source [[buffer(0)]],
                                                 constant float* sourceData [[buffer(1)]],
                                                 constant int& filterHeight [[buffer(2)]],
                                                 constant int& filterWidth [[buffer(3)]],
                                                 constant int& filterDepth [[buffer(4)]],
                                                 constant int& strideHeight [[buffer(5)]],
                                                 constant int& strideWidth [[buffer(6)]],
                                                 constant int& strideDepth [[buffer(7)]],
                                                 constant CBaseBlobDesc& result [[buffer(8)]],
                                                 device float* resultData [[buffer(9)]],
                                                 uint3 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C3DPosition pos( thread_position_in_grid );

	int b;
	int channel;
	int p;

	int resultGeomSize = result.Depth * result.Height * result.Width;

	if( !pos.GetMetalTaskIndex3D( result.BatchSize(), result.Channels, resultGeomSize, b, channel, p ) ) {
		return;
	}

	int inputWDC = source.Depth * source.Width * source.Channels;
	sourceData += b * source.ObjectSize();

	// Output position
	int xOut = p % result.Width;
	p /= result.Width;
	int yOut = p % result.Height;
	int zOut = p / result.Height;

    int resultShift = b * result.ObjectSize() + yOut * result.Width * result.Depth * result.Channels + 
    	xOut * result.Depth * result.Channels + zOut * result.Channels + channel;

	// Input position
	int xStart = xOut * strideWidth;
	int yStart = yOut * strideHeight;
	int zStart = zOut * strideDepth;

	float maxValue = -FLT_MAX;

    int iIndex = yStart * inputWDC + channel;
    for(int i = 0; i < filterHeight; i++) {
    	int jIndex = iIndex + xStart * source.Depth * source.Channels;
    	for(int j = 0; j < filterWidth; j++) {
    		int index = jIndex + zStart * source.Channels;
    		for(int k = 0; k < filterDepth; k++) {
    			float value = sourceData[index];
                if(value >= maxValue) {
                    maxValue = value;
                }
                index += source.Channels;
    		}
    		jIndex += source.Depth * source.Channels;
    	}
    	iIndex += inputWDC;
    }

	resultData[resultShift] = maxValue;
}
    
//-------------------------------------------------------------------------------------------------------------------
// 3D mean pooling

kernel void cubeKernelBlob3dMeanPooling( constant CBaseBlobDesc& source [[buffer(0)]],
                                         constant float* sourceData [[buffer(1)]],
                                         constant int& filterHeight [[buffer(2)]],
                                         constant int& filterWidth [[buffer(3)]],
                                         constant int& filterDepth [[buffer(4)]],
                                         constant int& strideHeight [[buffer(5)]],
                                         constant int& strideWidth [[buffer(6)]],
                                         constant int& strideDepth [[buffer(7)]],
                                         constant CBaseBlobDesc& result [[buffer(8)]],
                                         device float* resultData [[buffer(9)]],
                                         uint3 thread_position_in_grid [[ thread_position_in_grid ]]  )
{
    C3DPosition pos( thread_position_in_grid );

	int b;
	int channel;
	int p;

	int resultGeomSize = result.Depth * result.Height * result.Width;

	if( !pos.GetMetalTaskIndex3D( result.BatchSize(), result.Channels, resultGeomSize, b, channel, p ) ) {
		return;
	}

	int inputWDC = source.Depth * source.Width * source.Channels;
	sourceData += b * source.ObjectSize();

	// Output position
	int xOut = p % result.Width;
	p /= result.Width;
	int yOut = p % result.Height;
	int zOut = p / result.Height;

    int resultShift = b * result.ObjectSize() + yOut * result.Width * result.Depth * result.Channels + 
    	xOut * result.Depth * result.Channels + zOut * result.Channels + channel;

	// Input position
	int xStart = xOut * strideWidth;
	int yStart = yOut * strideHeight;
	int zStart = zOut * strideDepth;

	float sumValue = 0;

    int iIndex = yStart * inputWDC + channel;
    for(int i = 0; i < filterHeight; i++) {
    	int jIndex = iIndex + xStart * source.Depth * source.Channels;
    	for(int j = 0; j < filterWidth; j++) {
    		int index = jIndex + zStart * source.Channels;
    		for(int k = 0; k < filterDepth; k++) {
                sumValue += sourceData[index];
                index += source.Channels;
    		}
    		jIndex += source.Depth * source.Channels;
    	}
    	iIndex += inputWDC;
    }

	resultData[resultShift] = sumValue / filterHeight / filterWidth / filterDepth;
}

//--------------------------------------------------------------------------------------------------------------
// MaxOverTime Pooling

kernel void matrixKernelBlobMaxOverTimePooling( constant CBaseBlobDesc* source [[buffer(0)]],
                                                constant float* sourceData [[buffer(1)]],
                                                constant int* filterLen [[buffer(2)]],
                                                constant int* strideLen [[buffer(3)]],
                                                constant CBaseBlobDesc* result [[buffer(4)]],
                                                device float* resultData [[buffer(5)]],
                                                threadgroup float* bufferValues [[threadgroup(6)]],
                                                device int* maxIndicesData [[buffer(7)]],
                                                threadgroup int* bufferIndexes [[threadgroup(8)]],
                                                uint2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                                uint2 threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                                uint2 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C2DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    
	int bufferIndex = thread_position_in_threadgroup.y * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
    threadgroup float& value = bufferValues[bufferIndex];
    value = -FLT_MAX;
    threadgroup int& index = bufferIndexes[bufferIndex];
    
    int x;
	int p;
	if( pos.GetMetalTaskIndex2D( result->BlobSize(), *filterLen, 1, 1, p, x ) ) {
        int objectSize = source->ObjectSize();
        int seqElemSize = source->BatchWidth * objectSize;
        int seqNum = p / seqElemSize;
        int srcPos = p % seqElemSize;
        int srcSeqNum = seqNum * *strideLen + x;
        int srcSeqNumEnd = seqNum * *strideLen + *filterLen;

        index = srcSeqNum;
        value = *(sourceData + srcSeqNum * seqElemSize + srcPos);
        
        srcSeqNum += threads_per_threadgroup.x;
        while( srcSeqNum < srcSeqNumEnd ) {
            float candidate = *(sourceData + srcSeqNum * seqElemSize + srcPos);
            if( candidate > value ) {
                value = candidate;
                index = srcSeqNum;
            }
            srcSeqNum += threads_per_threadgroup.x;
        }
    }

    Reduce2DMax( thread_position_in_threadgroup, threads_per_threadgroup, bufferValues, bufferIndexes );

    if( p < result->BlobSize() && thread_position_in_threadgroup.x == 0 ) {
        resultData[p] = value;
        maxIndicesData[p] = index;
    }
}

kernel void matrixKernelBlobMaxOverTimePoolingNoIndexes( constant CBaseBlobDesc* source [[buffer(0)]],
                                                         constant float* sourceData [[buffer(1)]],
                                                         constant int* filterLen [[buffer(2)]],
                                                         constant int* strideLen [[buffer(3)]],
                                                         constant CBaseBlobDesc* result [[buffer(4)]],
                                                         device float* resultData [[buffer(5)]],
                                                         threadgroup float* buffer [[threadgroup(6)]],
                                                         uint2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                                         uint2 threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                                         uint2 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C2DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    
    threadgroup float& value = buffer[thread_position_in_threadgroup.y * threads_per_threadgroup.x + thread_position_in_threadgroup.x];
    value = -FLT_MAX;

	int x;
	int p;
	if( !pos.GetMetalTaskIndex2D( result->BlobSize(), *filterLen, 1, 1, p, x ) ) {
		return;
	}
    
    int objectSize = source->ObjectSize();
    int seqElemSize = source->BatchWidth * objectSize;
	int seqNum = p / seqElemSize;
	int srcPos = p % seqElemSize;
	int srcSeqNum = seqNum * *strideLen + x;
	int srcSeqNumEnd = seqNum * *strideLen + *filterLen;

    value = *(sourceData + srcSeqNum * seqElemSize + srcPos);
    
	srcSeqNum += threads_per_threadgroup.x;
	while( srcSeqNum < srcSeqNumEnd ) {
		float candidate = *(sourceData + srcSeqNum * seqElemSize + srcPos);
		if( candidate > value ) {
			value = candidate;
		}
		srcSeqNum += threads_per_threadgroup.x;
	}
    
    Reduce2DMax( thread_position_in_threadgroup, threads_per_threadgroup, buffer );

    if( thread_position_in_threadgroup.x == 0 ) {
        resultData[p] = value;
    }
}
    
//----------------------------------------------------------------------------------------------------------------------
// Max-over-time pooling

kernel void vectorKernelBlobGlobalMaxOverTimePoolingWithIndex( constant CBaseBlobDesc* source [[buffer(0)]],
                                                               constant float* sourceData [[buffer(1)]],
                                                               constant CBaseBlobDesc* result [[buffer(2)]],
                                                               device float* resultData [[buffer(3)]],
                                                               device int* maxIndicesData [[buffer(4)]],
                                                               uint thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C1DPosition pos( thread_position_in_grid );
    
    const int objectCount = source->BatchLength;
    const int objectSize = source->BlobSize() / objectCount;
    
    int objectNum;
    if( !pos.GetMetalTaskIndex( objectSize, objectNum ) ) {
        return;
    }
    
    int curIndex = objectNum;
    int maxIndex = curIndex;
    float maxVal = *(sourceData + curIndex);
    
    for(int i = 1; i < objectCount; ++i) {
        curIndex += objectSize;
        float candidate = *(sourceData + curIndex);
        if( candidate > maxVal ) {
            maxVal = candidate;
            maxIndex = curIndex;
        }
    }
    
    resultData[objectNum] = maxVal;
    maxIndicesData[objectNum] = maxIndex;
}

kernel void vectorKernelBlobGlobalMaxOverTimePooling( constant CBaseBlobDesc* source [[buffer(0)]],
                                                      constant float* sourceData [[buffer(1)]],
                                                      constant CBaseBlobDesc* result [[buffer(2)]],
                                                      device float* resultData [[buffer(3)]],
                                                      uint thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C1DPosition pos( thread_position_in_grid );
    
    const int objectCount = source->BatchLength;
    const int objectSize = source->BlobSize() / objectCount;
    
    int objectNum;
    if( !pos.GetMetalTaskIndex( objectSize, objectNum ) ) {
        return;
    }
    
    int curIndex = objectNum;
    float maxVal = -FLT_MAX;
    
    for(int i = 0; i < objectCount; ++i) {
        float candidate = *(sourceData + curIndex);
        if( candidate > maxVal ) {
            maxVal = candidate;
        }
        curIndex += objectSize;
    }
    
    resultData[objectNum] = maxVal;
}

//----------------------------------------------------------------------------------------------------------------------
// Max-over-time pooling

kernel void matrixLrn( constant float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant int* vectorCount [[buffer(2)]],
                       constant int* vectorSize [[buffer(3)]],
                       constant int* windowSize [[buffer(4)]],
                       constant float* bias [[buffer(5)]],
                       constant float* alpha [[buffer(6)]],
                       constant float* beta [[buffer(7)]],
                       uint2 thread_position_in_grid [[thread_position_in_grid]] )
{
    C2DPosition pos( thread_position_in_grid );

    int vectorIndex;
    int channelIndex;
    if( !pos.GetMetalTaskIndex2D( *vectorCount, *vectorSize, vectorIndex, channelIndex ) ) {
        return;
    }

    const int firstC = max( 0, channelIndex - ( *windowSize - 1 ) / 2 );
    const int lastC = min( *vectorSize - 1, channelIndex + *windowSize / 2 );

    input += vectorIndex * *vectorSize;
    output += vectorIndex * *vectorSize + channelIndex;

    float res = 0;

    for( int i = firstC; i <= lastC; ++i ) {
        res += input[i] * input[i];
    }

    res = *bias + *alpha * res / *windowSize;
    *output = pow( 1.f / res, *beta ) * input[channelIndex];
}
