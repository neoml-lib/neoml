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
    
    int ObjectSize() constant { return Height * Width * Depth * Channels; }
    int BatchSize() constant { return BatchLength * BatchWidth * ListSize; }
};
 
//-------------------------------------------------------------------------------------------------------------------
// Convolution

kernel void matrixKernelBlobConvolutionPrepare( constant CBaseBlobDesc* source [[buffer(0)]],
                                                constant float* sourceData [[buffer(1)]],
                                                constant int* paddingHeight [[buffer(2)]],
                                                constant int* paddingWidth [[buffer(3)]],
                                                constant int* strideHeight [[buffer(4)]],
                                                constant int* strideWidth [[buffer(5)]],
                                                constant int* dilationHeight [[buffer(6)]],
                                                constant int* dilationWidth [[buffer(7)]],
                                                constant CBaseBlobDesc* filt [[buffer(8)]],
                                                constant CBaseBlobDesc* result [[buffer(9)]],
                                                constant int* resultOffset [[buffer(10)]],
                                                constant int* resultSize [[buffer(11)]],
                                                device float* resultData [[buffer(12)]],
                                                uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );

    int x;
    int y;
    int xy;
    int c;
    if( pos.GetMetalTaskIndex2D( *resultSize, source->Depth * source->Channels, xy, c ) ) {
        sourceData += c;
        resultData += xy * filt->ObjectSize() + c;
        xy += *resultOffset;
        x = xy % result->Width;
        y = xy / result->Width;
        
        int startX = *strideWidth * x + -*paddingWidth;
        int startY = *strideHeight * y + -*paddingHeight;
        int inputY = startY;
        for( int fy = 0; fy < filt->Height; fy++ ) {
            if( 0 <= inputY && inputY < source->Height ) {
                int inputX = startX;
                constant float* sourceDataPtr = sourceData + inputY * source->Width * source->Channels * source->Depth;
                for( int fx = 0; fx < filt->Width; fx++ ) {
                    if( 0 <= inputX && inputX < source->Width ) {
                        *resultData = sourceDataPtr[inputX * source->Channels * source->Depth];
                    } else {
                        *resultData = 0;
                    }
                    resultData += source->Channels * source->Depth;
                    inputX += *dilationWidth;
                }
            } else {
                for( int fx = 0; fx < filt->Width; fx++ ) {
                    *resultData = 0;
                    resultData += source->Channels * source->Depth;
                }
            }
            inputY += *dilationHeight;
        }
    }
}

kernel void cubeKernelBlobConvolution( constant CBaseBlobDesc& source [[buffer(0)]],
                                        constant float* sourceData [[buffer(1)]],
                                        constant int& paddingHeight [[buffer(2)]],
                                        constant int& paddingWidth [[buffer(3)]],
                                        constant int& strideHeight [[buffer(4)]],
                                        constant int& strideWidth [[buffer(5)]],
                                        constant int& dilationHeight [[buffer(6)]],
                                        constant int& dilationWidth [[buffer(7)]],
                                        constant CBaseBlobDesc& filter [[buffer(8)]],
                                        constant float* filterData [[buffer(9)]],
                                        constant CBaseBlobDesc& result [[buffer(10)]],
                                        device float* resultData [[buffer(11)]],
                                        uint3 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C3DPosition pos( thread_position_in_grid );

    int x;
    int y;
    int bc;
    const int totalInputChannels = source.Channels * source.Depth;
    if( pos.GetMetalTaskIndex3D( result.Height, result.Width, result.Channels * result.BatchSize(), y, x, bc ) ) {
        const int b = bc / result.Channels;
        const int c = bc - b * result.Channels;

        sourceData += b * source.ObjectSize();
        resultData += b * result.ObjectSize() + y * result.Width * result.Channels + x * result.Channels + c;
        filterData += c * filter.ObjectSize();

        const int inputHStart = y * strideHeight - paddingHeight;
        const int inputHEnd = inputHStart + filter.Height * dilationHeight;
        const int inputWStart = x * strideWidth - paddingWidth;
        const int inputWEnd = inputWStart + filter.Width * dilationWidth;

        float resultVal = 0.0;
        for( int k = 0; k < totalInputChannels; k++) {
            int filterY = 0;
            for( int j = inputHStart; j < inputHEnd; j += dilationHeight ) {
                int filterX = 0;
                int sourceHOffset = j * source.Width * totalInputChannels;
                int filterHOffset = filterY * filter.Width * totalInputChannels;
                for( int i = inputWStart; i < inputWEnd; i += dilationWidth ) {
                    if(j >= 0 && j < source.Height && i >= 0 && i < source.Width) {
                        const float srcVal = sourceData[sourceHOffset + i * totalInputChannels];
                        const float fltVal = filterData[filterHOffset + filterX * totalInputChannels];
                        resultVal = fma(srcVal, fltVal, resultVal );
                    }
                    filterX++;
                }
                filterY++;
            }
            sourceData++;
            filterData++;
        }

        *resultData = resultVal;
    }
}

kernel void cubeKernelBlobConvolutionWithFreeTerm( constant CBaseBlobDesc& source [[buffer(0)]],
                                                    constant float* sourceData [[buffer(1)]],
                                                    constant int& paddingHeight [[buffer(2)]],
                                                    constant int& paddingWidth [[buffer(3)]],
                                                    constant int& strideHeight [[buffer(4)]],
                                                    constant int& strideWidth [[buffer(5)]],
                                                    constant int& dilationHeight [[buffer(6)]],
                                                    constant int& dilationWidth [[buffer(7)]],
                                                    constant CBaseBlobDesc& filter [[buffer(8)]],
                                                    constant float* filterData [[buffer(9)]],
                                                    constant float* freeTermData [[buffer(10)]],
                                                    constant CBaseBlobDesc& result [[buffer(11)]],
                                                    device float* resultData [[buffer(12)]],
                                                    uint3 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C3DPosition pos( thread_position_in_grid );

    int x;
    int y;
    int bc;
    const int totalInputChannels = source.Channels * source.Depth;
    if( pos.GetMetalTaskIndex3D( result.Height, result.Width, result.Channels * result.BatchSize(), y, x, bc ) ) {
        const int b = bc / result.Channels;
        const int c = bc - b * result.Channels;

        sourceData += b * source.ObjectSize();
        resultData += b * result.ObjectSize() + y * result.Width * result.Channels + x * result.Channels + c;
        filterData += c * filter.ObjectSize();

        const int inputHStart = y * strideHeight - paddingHeight;
        const int inputHEnd = inputHStart + filter.Height * dilationHeight;
        const int inputWStart = x * strideWidth - paddingWidth;
        const int inputWEnd = inputWStart + filter.Width * dilationWidth;

        float resultVal = freeTermData[c];
        for( int k = 0; k < totalInputChannels; k++) {
            int filterY = 0;
            for( int j = inputHStart; j < inputHEnd; j += dilationHeight ) {
                int filterX = 0;
                int sourceHOffset = j * source.Width * totalInputChannels;
                int filterHOffset = filterY * filter.Width * totalInputChannels;
                for( int i = inputWStart; i < inputWEnd; i += dilationWidth ) {
                    if(j >= 0 && j < source.Height && i >= 0 && i < source.Width) {
                        const float srcVal = sourceData[sourceHOffset + i * totalInputChannels];
                        const float fltVal = filterData[filterHOffset + filterX * totalInputChannels];
                        resultVal = fma(srcVal, fltVal, resultVal );
                    }
                    filterX++;
                }
                filterY++;
            }
            sourceData++;
            filterData++;
        }

        *resultData = resultVal;
    }
}

kernel void vectorKernelBlobConvolutionBackward( constant CBaseBlobDesc* outputDiff [[buffer(0)]],
                                                 constant int* paddingHeight [[buffer(1)]],
                                                 constant int* paddingWidth [[buffer(2)]],
                                                 constant int* strideHeight [[buffer(3)]],
                                                 constant int* strideWidth [[buffer(4)]],
                                                 constant int* dilationHeight [[buffer(5)]],
                                                 constant int* dilationWidth [[buffer(6)]],
                                                 constant CBaseBlobDesc* filt [[buffer(7)]],
                                                 constant float* filtData [[buffer(8)]],
                                                 constant CBaseBlobDesc* inputDiff [[buffer(9)]],
                                                 device float* inputDiffData [[buffer(10)]],
                                                 uint thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C1DPosition pos( thread_position_in_grid );

    int row;
    if( pos.GetMetalTaskIndex( inputDiff->Height, row ) ) {
        const int channelsCount = inputDiff->Channels * inputDiff->Depth;
        inputDiffData += row * inputDiff->Width * channelsCount;

        for( int y = 0; y < outputDiff->Height; y++ ) {
            const int firstFilterRow = y * *strideHeight - *paddingHeight;
            const int lastFilterRow = firstFilterRow + *dilationHeight * ( filt->Height - 1 );
            if( firstFilterRow <= row && row <= lastFilterRow && ( row - firstFilterRow ) % *dilationHeight == 0 ) {
                const int filtY = ( row - firstFilterRow ) / *dilationHeight;
                for( int x = 0; x < outputDiff->Width; x++ ) {
                    int column = x * *strideWidth - *paddingWidth;
                    device float* inputDiffDataPtr = inputDiffData + column * channelsCount;
                    constant float* filtDataPtr = filtData
                        + ( y * outputDiff->Width + x ) * filt->ObjectSize() + filtY * filt->Width * channelsCount;

                    for( int filtX = 0; filtX < filt->Width; filtX++ ) {
                        if( 0 <= column && column < inputDiff->Width ) {
                            for( int filtC = 0; filtC < channelsCount; filtC++ ) {
                                inputDiffDataPtr[filtC] += filtDataPtr[filtC];
                            }
                        }
                        column += *dilationWidth;
                        inputDiffDataPtr += channelsCount * *dilationWidth;
                        filtDataPtr += channelsCount;
                    }
                }
            }
        }
    }
}

//-------------------------------------------------------------------------------------------------------------------
// 3D Convolution

kernel void cubeKernelBlob3DConvolution(constant CBaseBlobDesc& source [[buffer(0)]],
                                        constant float* sourceData [[buffer(1)]],
                                        constant int& paddingDepth [[buffer(2)]],
                                        constant int& paddingHeight [[buffer(3)]],
                                        constant int& paddingWidth [[buffer(4)]],
                                        constant int& strideDepth [[buffer(5)]],
                                        constant int& strideHeight [[buffer(6)]],
                                        constant int& strideWidth [[buffer(7)]],
                                        constant CBaseBlobDesc& filter [[buffer(8)]],
                                        constant float* filterData [[buffer(9)]],
                                        constant CBaseBlobDesc& result [[buffer(10)]],
                                        device float* resultData [[buffer(11)]],
                                        uint3 thread_position_in_grid [[ thread_position_in_grid ]])
{
    C3DPosition pos( thread_position_in_grid );

    int x;
    int y;
    int bcz;
    if( pos.GetMetalTaskIndex3D( result.Height, result.Width, result.Channels * result.BatchSize() * result.Depth, y, x, bcz ) ) {
        const int b = bcz / ( result.Channels * result.Depth );
        const int cz = bcz - b * result.Channels * result.Depth;
        const int c = cz / result.Depth;
        const int z = cz - c * result.Depth;

        sourceData += b * source.ObjectSize();
        resultData += b * result.ObjectSize() + y * result.Width * result.Depth * result.Channels + x * result.Depth * result.Channels + z * result.Channels + c;
        filterData += c * filter.ObjectSize();

        const int inputDStart = z * strideDepth - paddingDepth;
        const int inputDEnd = inputDStart + filter.Depth;
        const int inputHStart = y * strideHeight - paddingHeight;
        const int inputHEnd = inputHStart + filter.Height;
        const int inputWStart = x * strideWidth - paddingWidth;
        const int inputWEnd = inputWStart + filter.Width;

        float resultVal = 0.0;
        for( int k = 0; k < source.Channels; k++) {
            int filterY = 0;
            for( int j = inputHStart; j < inputHEnd; j++ ) {
                if( j < 0 || j >= source.Height ) {
                    filterY++;
                    continue;
                }
                
                int sourceHOffset = j * source.Width * source.Depth * source.Channels;
                int filterHOffset = filterY * filter.Width * filter.Depth * filter.Channels;
                int filterX = 0;
                for( int i = inputWStart; i < inputWEnd; i++ ) {
                    if( i < 0 || i >= source.Width ) {
                        filterX++;
                        continue;
                    }
                    
                    int sourceWOffset = sourceHOffset + i * source.Depth * source.Channels;
                    int filterWOffset = filterHOffset + filterX * filter.Depth * filter.Channels;
                    int filterZ = 0;
                    for( int l = inputDStart; l < inputDEnd; l++ ) {
                        if( l < 0 || l >= source.Depth ) {
                            filterZ++;
                            continue;
                        }
                        const float srcVal = sourceData[sourceWOffset + l * source.Channels];
                        const float fltVal = filterData[filterWOffset + filterZ * filter.Channels];
                        resultVal = fma(srcVal, fltVal, resultVal );
                        filterZ++;
                    }
                    filterX++;
                }
                filterY++;
            }
            sourceData++;
            filterData++;
        }

        *resultData = resultVal;
    }
}

kernel void cubeKernelBlob3DConvolutionWithFreeTerm(constant CBaseBlobDesc& source [[buffer(0)]],
                                                    constant float* sourceData [[buffer(1)]],
                                                    constant int& paddingDepth [[buffer(2)]],
                                                    constant int& paddingHeight [[buffer(3)]],
                                                    constant int& paddingWidth [[buffer(4)]],
                                                    constant int& strideDepth [[buffer(5)]],
                                                    constant int& strideHeight [[buffer(6)]],
                                                    constant int& strideWidth [[buffer(7)]],
                                                    constant CBaseBlobDesc& filter [[buffer(8)]],
                                                    constant float* filterData [[buffer(9)]],
                                                    constant float* freeTerm [[buffer(10)]],
                                                    constant CBaseBlobDesc& result [[buffer(11)]],
                                                    device float* resultData [[buffer(12)]],
                                                    uint3 thread_position_in_grid [[ thread_position_in_grid ]])
{
    C3DPosition pos( thread_position_in_grid );

    int x;
    int y;
    int bcz;
    if( pos.GetMetalTaskIndex3D( result.Height, result.Width, result.Channels * result.BatchSize() * result.Depth, y, x, bcz ) ) {
        const int b = bcz / ( result.Channels * result.Depth );
        const int cz = bcz - b * result.Channels * result.Depth;
        const int c = cz / result.Depth;
        const int z = cz - c * result.Depth;

        sourceData += b * source.ObjectSize();
        resultData += b * result.ObjectSize() + y * result.Width * result.Depth * result.Channels + x * result.Depth * result.Channels + z * result.Channels + c;
        filterData += c * filter.ObjectSize();

        const int inputDStart = z * strideDepth - paddingDepth;
        const int inputDEnd = inputDStart + filter.Depth;
        const int inputHStart = y * strideHeight - paddingHeight;
        const int inputHEnd = inputHStart + filter.Height;
        const int inputWStart = x * strideWidth - paddingWidth;
        const int inputWEnd = inputWStart + filter.Width;

        float resultVal = freeTerm[c];
        for( int k = 0; k < source.Channels; k++) {
            int filterY = 0;
            for( int j = inputHStart; j < inputHEnd; j++ ) {
                if( j < 0 || j >= source.Height ) {
                    filterY++;
                    continue;
                }
                
                int sourceHOffset = j * source.Width * source.Depth * source.Channels;
                int filterHOffset = filterY * filter.Width * filter.Depth * filter.Channels;
                int filterX = 0;
                for( int i = inputWStart; i < inputWEnd; i++ ) {
                    if( i < 0 || i >= source.Width ) {
                        filterX++;
                        continue;
                    }
                    
                    int sourceWOffset = sourceHOffset + i * source.Depth * source.Channels;
                    int filterWOffset = filterHOffset + filterX * filter.Depth * filter.Channels;
                    int filterZ = 0;
                    for( int l = inputDStart; l < inputDEnd; l++ ) {
                        if( l < 0 || l >= source.Depth ) {
                            filterZ++;
                            continue;
                        }
                        const float srcVal = sourceData[sourceWOffset + l * source.Channels];
                        const float fltVal = filterData[filterWOffset + filterZ * filter.Channels];
                        resultVal = fma(srcVal, fltVal, resultVal );
                        filterZ++;
                    }
                    filterX++;
                }
                filterY++;
            }
            sourceData++;
            filterData++;
        }

        *resultData = resultVal;
    }
}


kernel void cubeKernelBlob3DConvolutionPrepare( constant CBaseBlobDesc* source [[buffer(0)]],
                                                constant float* sourceData [[buffer(1)]],
                                                constant int* paddingDepth [[buffer(2)]],
                                                constant int* paddingHeight [[buffer(3)]],
                                                constant int* paddingWidth [[buffer(4)]],
                                                constant int* strideDepth [[buffer(5)]],
                                                constant int* strideHeight [[buffer(6)]],
                                                constant int* strideWidth [[buffer(7)]],
                                                constant CBaseBlobDesc* filt [[buffer(8)]],
                                                constant CBaseBlobDesc* result [[buffer(9)]],
                                                device float* resultData [[buffer(10)]],
                                                uint3 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C3DPosition pos( thread_position_in_grid );
    
    int x;
    int y;
    int z;
    if( pos.GetMetalTaskIndex3D( result->Depth, result->Height, result->Width, z, y, x ) ) {
        resultData += ( ( z * result->Height + y ) * result->Width + x ) * filt->ObjectSize();
        
        int startX = *strideWidth * x + -*paddingWidth;
        int startY = *strideHeight * y + -*paddingHeight;
        int startZ = *strideDepth * z + -*paddingDepth;
        for( int c = 0; c < source->Channels; c++ ) {
            int inputZ = startZ;
            for( int fz = 0; fz < filt->Depth; fz++ ) {
                if( 0 <= inputZ && inputZ < source->Depth ) {
                    int inputY = startY;
                    for( int fy = 0; fy < filt->Height; fy++ ) {
                        if( 0 <= inputY && inputY < source->Height ) {
                            int inputX = startX;
                            constant float* sourceDataPtr = sourceData + ( inputZ * source->Height + inputY ) * source->Width;
                            for( int fx = 0; fx < filt->Width; fx++ ) {
                                if( 0 <= inputX && inputX < source->Width ) {
                                    *resultData = sourceDataPtr[inputX];
                                } else {
                                    *resultData = 0;
                                }
                                resultData++;
                                inputX++;
                            }
                        } else {
                            for( int fx = 0; fx < filt->Width; fx++ ) {
                                *resultData = 0;
                                resultData++;
                            }
                        }
                        inputY++;
                    }
                } else {
                    for( int fxfy = 0; fxfy < filt->Width * filt->Height; fxfy++ ) {
                        *resultData = 0;
                        resultData++;
                    }
                }
                inputZ++;
            }
            sourceData += source->Depth * source->Height * source->Width;
        }
    }
}
    
kernel void matrixKernelBlob3DConvolutionBackward( constant CBaseBlobDesc* outputDiff [[buffer(0)]],
                                                   constant int* paddingDepth [[buffer(1)]],
                                                   constant int* paddingHeight [[buffer(2)]],
                                                   constant int* paddingWidth [[buffer(3)]],
                                                   constant int* strideDepth [[buffer(4)]],
                                                   constant int* strideHeight [[buffer(5)]],
                                                   constant int* strideWidth [[buffer(6)]],
                                                   constant CBaseBlobDesc* filt [[buffer(7)]],
                                                   constant float* filtData [[buffer(8)]],
                                                   constant CBaseBlobDesc* inputDiff [[buffer(9)]],
                                                   device float* inputDiffData [[buffer(10)]],
                                                   uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    
    int row;
    int layer;
    if( pos.GetMetalTaskIndex2D( inputDiff->Depth, inputDiff->Height, layer, row ) ) {
        inputDiffData += layer * inputDiff->Channels + row * inputDiff->Width * inputDiff->Channels * inputDiff->Depth;
        
        for( int z = 0; z < outputDiff->Depth; z++ ) {
            const int firstFilterLayer = z * *strideDepth - *paddingDepth;
            const int lastFilterLayer = firstFilterLayer + filt->Depth - 1;
            if( firstFilterLayer <= layer && layer <= lastFilterLayer ) {
                const int filtZ = layer - firstFilterLayer;
                for( int y = 0; y < outputDiff->Height; y++ ) {
                    const int firstFilterRow = y * *strideHeight - *paddingHeight;
                    const int lastFilterRow = firstFilterRow + filt->Height - 1;
                    if( firstFilterRow <= row && row <= lastFilterRow ) {
                        const int filtY = row - firstFilterRow;
                        for( int x = 0; x < outputDiff->Width; x++ ) {
                            device float* inputDiffDataPtr = inputDiffData;
                            constant float* filtDataPtr = filtData + ( y * outputDiff->Depth * outputDiff->Width + x * outputDiff->Depth + z ) * filt->ObjectSize() + filtY * filt->Width * filt->Depth * filt->Channels + filtZ * filt->Channels; // ( filtZ * filt->Height + filtY ) * filt->Width;

                            for( int filtC = 0; filtC < filt->Channels; filtC++ ) {
                                int column = x * *strideWidth - *paddingWidth;
                                for( int filtX = 0; filtX < filt->Width; filtX++ ) {
                                    if( 0 <= column && column < inputDiff->Width ) {
                                        inputDiffDataPtr[column * inputDiff->Depth * inputDiff->Channels] += filtDataPtr[filtX * filt->Depth * filt->Channels];
                                    }
                                    column++;
                                }
                                inputDiffDataPtr += 1;
                                filtDataPtr += 1;
                            }
                        }
                    }
                }
            }
        }
    }
}
    
//-------------------------------------------------------------------------------------------------------------------
// Time convolution:

static constant int BlobTimeConvolutionPrepareCombine = 16;
kernel void cubeKernelBlobTimeConvolutionPrepare( constant CBaseBlobDesc* source [[buffer(0)]],
                                                  constant float* sourceData [[buffer(1)]],
                                                  constant int* strideSize [[buffer(2)]],
                                                  constant int* paddingSize [[buffer(3)]],
                                                  constant int* dilation [[buffer(4)]],
                                                  constant CBaseBlobDesc* filt [[buffer(5)]],
                                                  constant float* filtData [[buffer(6)]],
                                                  constant CBaseBlobDesc* result [[buffer(7)]],
                                                  constant float* resultData [[buffer(8)]],
                                                  device float* preparedData [[buffer(9)]],
                                                  uint3 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                                  uint3 threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                                  uint3 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]] )
{
    C3DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );

    int h;
	int seqNumber;
	int x;
	if( !pos.GetMetalTaskIndex3D( filt->Height, result->BatchLength, source->BatchWidth * source->ObjectSize(),
                                 1, 1, BlobTimeConvolutionPrepareCombine, h, seqNumber, x ) ) {
		return;
	}

	int inputSeqNumber = seqNumber * *strideSize + h * *dilation - *paddingSize;

	int objectSize = source->ObjectSize();

	int sourceShift = inputSeqNumber * source->BatchWidth * objectSize;

	int resultShift = objectSize * filt->Height * result->BatchWidth * seqNumber + objectSize * h;
	int resultStep = objectSize * filt->Height;

	constant float* inputData = (0 <= inputSeqNumber && inputSeqNumber < source->BatchLength) ? (sourceData + sourceShift) : 0;
	device float* outputData = preparedData + resultShift;

	// Passing over x
	int index;
	int step;
	int count = pos.GetMetalWidthTaskCountAndIndex( source->BatchWidth * objectSize, BlobTimeConvolutionPrepareCombine, index, step );

	for( int i = 0; i < count; ++i ) {
		int batch = index / objectSize;
		int pos = index % objectSize;
		outputData[batch * resultStep + pos] = (inputData == 0) ? 0 : *(inputData + index);
        index += step;
	}
}

//-------------------------------------------------------------------------------------------------------------------
// Channelwise Convolution
    
inline float getData( int x, int y, int width, int height, constant float* sourceData, int index )
{
    return ( 0 <= x && x < width && 0 <= y && y < height ) ? sourceData[index] : 0;
}

inline float3 get3floats( int x, int y, int width, int height, constant float* sourceData, int index, int step )
{
    float3 res = 0;
    res.x = getData(x, y, width, height, sourceData, index);
    res.y = getData(x + 1, y, width, height, sourceData, index + step);
    res.z = getData(x + 2, y, width, height, sourceData, index + 2 * step);
    
    return res;
}

inline float2 get2floats( int x, int y, int width, int height, constant float* sourceData, int index, int step )
{
    float2 res = 0;
    res.x = getData(x, y, width, height, sourceData, index);
    res.y = getData(x + 1, y, width, height, sourceData, index + step);
    
    return res;
}

inline void set4floats( float4 val, int x, int y, int width, int height, device float* dest, int step )
{
    if( y < height ) {
        int i = 0;
        while ( i < 4 && x + i < width ) {
            dest[i * step] = val[i];
            i++;
        }
    }
}

inline void set2floats( float2 val, int x, int y, int width, int height, device float* dest, int step )
{
    if( y < height ) {
        int i = 0;
        while ( i < 2 && x + i < width ) {
            dest[i * step] = val[i];
            i++;
        }
    }
}
    
kernel void cubeKernelBlobChannelwiseConvolutionBase( constant CBaseBlobDesc& source [[buffer(0)]],
                                                      constant float* sourceData [[buffer(1)]],
                                                      constant CBaseBlobDesc& filt [[buffer(2)]],
                                                      constant float* filtData [[buffer(3)]],
                                                      constant int& paddingHeight [[buffer(4)]],
                                                      constant int& paddingWidth [[buffer(5)]],
                                                      constant int& strideHeight [[buffer(6)]],
                                                      constant int& strideWidth [[buffer(7)]],
                                                      constant CBaseBlobDesc& result [[buffer(8)]],
                                                      device float* resultData [[buffer(9)]],
                                                      uint3 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C3DPosition pos( thread_position_in_grid );

    int b;
    int g;
    int c;
    if( pos.GetMetalTaskIndex3D( result.BatchSize(), result.Height * result.Width, result.Channels, b, g, c ) ) {
        const int sourceX0 = -paddingWidth + strideWidth * ( g % result.Width );
        
        int sourceY = -paddingHeight + strideHeight * ( g / result.Width );
        int sourceGY = sourceY * source.Width;
        
        filtData += c;
        sourceData += b * source.ObjectSize() + c;
        resultData += b * result.ObjectSize() + g * result.Channels + c;
        
        float result = 0;
        for( int i = 0; i < filt.Height; i++ ) {
            int sourceX = sourceX0;
            for( int j = 0; j < filt.Width; j++ ) {
                result += getData( sourceX, sourceY, source.Width, source.Height, sourceData, ( sourceGY + sourceX ) * source.Channels ) * *filtData;
                filtData += filt.Channels;
                sourceX++;
            }
            sourceY++;
            sourceGY += source.Width;
        }
        
        *resultData = result;
    }
}

kernel void cubeKernelBlobChannelwiseConvolutionBaseWithFreeTerm( constant CBaseBlobDesc& source [[buffer(0)]],
                                                                  constant float* sourceData [[buffer(1)]],
                                                                  constant CBaseBlobDesc& filt [[buffer(2)]],
                                                                  constant float* filtData [[buffer(3)]],
                                                                  constant int& paddingHeight [[buffer(4)]],
                                                                  constant int& paddingWidth [[buffer(5)]],
                                                                  constant int& strideHeight [[buffer(6)]],
                                                                  constant int& strideWidth [[buffer(7)]],
                                                                  constant float* freeTerm [[buffer(8)]],
                                                                  constant CBaseBlobDesc& result [[buffer(9)]],
                                                                  device float* resultData [[buffer(10)]],
                                                                  uint3 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C3DPosition pos( thread_position_in_grid );

    int b;
    int g;
    int c;
    if( pos.GetMetalTaskIndex3D( result.BatchSize(), result.Height * result.Width, result.Channels, b, g, c ) ) {
        const int sourceX0 = -paddingWidth + strideWidth * ( g % result.Width );
        
        int sourceY = -paddingHeight + strideHeight * ( g / result.Width );
        int sourceGY = sourceY * source.Width;
        
        filtData += c;
        sourceData += b * source.ObjectSize() + c;
        resultData += b * result.ObjectSize() + g * result.Channels + c;
        
        float result = freeTerm[c];
        for( int i = 0; i < filt.Height; i++ ) {
            int sourceX = sourceX0;
            for( int j = 0; j < filt.Width; j++ ) {
                result += getData( sourceX, sourceY, source.Width, source.Height, sourceData, ( sourceGY + sourceX ) * source.Channels ) * *filtData;
                filtData += filt.Channels;
                sourceX++;
            }
            sourceY++;
            sourceGY += source.Width;
        }
        
        *resultData = result;
    }
}

#define CONVOLUTION1X3(acc, src0, src1, weights) \
    ({                                           \
      acc.x += src0.x * weights.x;               \
      acc.y += src0.y * weights.x;               \
      acc.z += src0.z * weights.x;               \
      acc.w += src1.x * weights.x;               \
        \
      acc.x += src0.y * weights.y;               \
      acc.y += src0.z * weights.y;               \
      acc.z += src1.x * weights.y;               \
      acc.w += src1.y * weights.y;               \
        \
      acc.x += src0.z * weights.z;               \
      acc.y += src1.x * weights.z;               \
      acc.z += src1.y * weights.z;               \
      acc.w += src1.z * weights.z;               \
    }) 

kernel void cubeKernelBlobChannelwiseConvolution3x3( constant CBaseBlobDesc& source [[buffer(0)]],
                                                     constant float* sourceData [[buffer(1)]],
                                                     constant CBaseBlobDesc& filt [[buffer(2)]],
                                                     constant float* filtData [[buffer(3)]],
                                                     constant int& paddingHeight [[buffer(4)]],
                                                     constant int& paddingWidth [[buffer(5)]],
                                                     constant CBaseBlobDesc& result [[buffer(6)]],
                                                     device float* resultData [[buffer(7)]],
                                                     uint3 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C3DPosition pos( thread_position_in_grid );

    const int combineW = 4;
    const int combineH = 3;

    int channelBlocksCount = ( result.Height / combineH + ( result.Height % combineH != 0 ) ) * ( result.Width / combineW + ( result.Width % combineW != 0 ) );
    
    int b;
    int g;
    int c;
    if( pos.GetMetalTaskIndex3D( result.BatchSize(), channelBlocksCount, result.Channels, b, g, c ) ) {
        const int blocksInRow = result.Width / combineW + ( result.Width % combineW != 0 );

        filtData += c;

        sourceData += b * source.ObjectSize() + c;

        const int sourceX0 = -paddingWidth + ( g % blocksInRow ) * combineW;
        int sourceY = -paddingHeight + ( g / blocksInRow ) * combineH;
        int sourceGY = sourceY * source.Width;

        const int resultX = ( g % blocksInRow ) * combineW;
        const int resultY = ( g / blocksInRow ) * combineH;

        resultData += b * result.ObjectSize() + ( result.Width * resultY + resultX ) * result.Channels + c;

        float4 values0 = 0;
        float4 values1 = 0;
        float4 values2 = 0;

        float3 weights0 = float3(filtData[0], filtData[filt.Channels], filtData[2 * filt.Channels]);
        float3 weights1 = float3(filtData[3 * filt.Channels], filtData[4 * filt.Channels], filtData[5 * filt.Channels]);
        float3 weights2 = float3(filtData[6 * filt.Channels], filtData[7 * filt.Channels], filtData[8 * filt.Channels]);

        float3 src0 = get3floats(sourceX0, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0) * source.Channels, source.Channels);
        float3 src1 = get3floats(sourceX0 + 3, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0 + 3) * source.Channels, source.Channels);

        CONVOLUTION1X3(values0, src0, src1, weights0);
        
        sourceY += 1;
        sourceGY += source.Width;
        src0 = get3floats(sourceX0, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0) * source.Channels, source.Channels);
        src1 = get3floats(sourceX0 + 3, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0 + 3) * source.Channels, source.Channels);

        CONVOLUTION1X3(values0, src0, src1, weights1);
        CONVOLUTION1X3(values1, src0, src1, weights0);

        sourceY += 1;
        sourceGY += source.Width;
        src0 = get3floats(sourceX0, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0) * source.Channels, source.Channels);
        src1 = get3floats(sourceX0 + 3, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0 + 3) * source.Channels, source.Channels);

        CONVOLUTION1X3(values0, src0, src1, weights2);
        CONVOLUTION1X3(values1, src0, src1, weights1);
        CONVOLUTION1X3(values2, src0, src1, weights0);

        sourceY += 1;
        sourceGY += source.Width;
        src0 = get3floats(sourceX0, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0) * source.Channels, source.Channels);
        src1 = get3floats(sourceX0 + 3, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0 + 3) * source.Channels, source.Channels);

        CONVOLUTION1X3(values1, src0, src1, weights2);
        CONVOLUTION1X3(values2, src0, src1, weights1);

        sourceY += 1;
        sourceGY += source.Width;
        src0 = get3floats(sourceX0, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0) * source.Channels, source.Channels);
        src1 = get3floats(sourceX0 + 3, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0 + 3) * source.Channels, source.Channels);

        CONVOLUTION1X3(values2, src0, src1, weights2);

        set4floats( values0, resultX, resultY, result.Width, result.Height, resultData, result.Channels );
        resultData += result.Width * result.Channels;
        set4floats( values1, resultX, resultY + 1, result.Width, result.Height, resultData, result.Channels );
        resultData += result.Width * result.Channels;
        set4floats( values2, resultX, resultY + 2, result.Width, result.Height, resultData, result.Channels );
    }
}

kernel void cubeKernelBlobChannelwiseConvolution3x3WithFreeTerm( constant CBaseBlobDesc& source [[buffer(0)]],
                                                                 constant float* sourceData [[buffer(1)]],
                                                                 constant CBaseBlobDesc& filt [[buffer(2)]],
                                                                 constant float* filtData [[buffer(3)]],
                                                                 constant int& paddingHeight [[buffer(4)]],
                                                                 constant int& paddingWidth [[buffer(5)]],
                                                                 constant float* freeTerm [[buffer(6)]],
                                                                 constant CBaseBlobDesc& result [[buffer(7)]],
                                                                 device float* resultData [[buffer(8)]],
                                                                 uint3 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C3DPosition pos( thread_position_in_grid );

    const int combineW = 4;
    const int combineH = 3;

    int channelBlocksCount = ( result.Height / combineH + ( result.Height % combineH != 0 ) ) * ( result.Width / combineW + ( result.Width % combineW != 0 ) );
    
    int b;
    int g;
    int c;
    if( pos.GetMetalTaskIndex3D( result.BatchSize(), channelBlocksCount, result.Channels, b, g, c ) ) {
        const int blocksInRow = result.Width / combineW + ( result.Width % combineW != 0 );

        filtData += c;

        sourceData += b * source.ObjectSize() + c;

        const int sourceX0 = -paddingWidth + ( g % blocksInRow ) * combineW;
        int sourceY = -paddingHeight + ( g / blocksInRow ) * combineH;
        int sourceGY = sourceY * source.Width;

        const int resultX = ( g % blocksInRow ) * combineW;
        const int resultY = ( g / blocksInRow ) * combineH;

        resultData += b * result.ObjectSize() + ( result.Width * resultY + resultX ) * result.Channels + c;

        float4 values0 = freeTerm[c];
        float4 values1 = freeTerm[c];
        float4 values2 = freeTerm[c];

        float3 weights0 = float3(filtData[0], filtData[filt.Channels], filtData[2 * filt.Channels]);
        float3 weights1 = float3(filtData[3 * filt.Channels], filtData[4 * filt.Channels], filtData[5 * filt.Channels]);
        float3 weights2 = float3(filtData[6 * filt.Channels], filtData[7 * filt.Channels], filtData[8 * filt.Channels]);

        float3 src0 = get3floats(sourceX0, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0) * source.Channels, source.Channels);
        float3 src1 = get3floats(sourceX0 + 3, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0 + 3) * source.Channels, source.Channels);

        CONVOLUTION1X3(values0, src0, src1, weights0);
        
        sourceY += 1;
        sourceGY += source.Width;
        src0 = get3floats(sourceX0, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0) * source.Channels, source.Channels);
        src1 = get3floats(sourceX0 + 3, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0 + 3) * source.Channels, source.Channels);

        CONVOLUTION1X3(values0, src0, src1, weights1);
        CONVOLUTION1X3(values1, src0, src1, weights0);

        sourceY += 1;
        sourceGY += source.Width;
        src0 = get3floats(sourceX0, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0) * source.Channels, source.Channels);
        src1 = get3floats(sourceX0 + 3, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0 + 3) * source.Channels, source.Channels);

        CONVOLUTION1X3(values0, src0, src1, weights2);
        CONVOLUTION1X3(values1, src0, src1, weights1);
        CONVOLUTION1X3(values2, src0, src1, weights0);

        sourceY += 1;
        sourceGY += source.Width;
        src0 = get3floats(sourceX0, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0) * source.Channels, source.Channels);
        src1 = get3floats(sourceX0 + 3, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0 + 3) * source.Channels, source.Channels);

        CONVOLUTION1X3(values1, src0, src1, weights2);
        CONVOLUTION1X3(values2, src0, src1, weights1);

        sourceY += 1;
        sourceGY += source.Width;
        src0 = get3floats(sourceX0, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0) * source.Channels, source.Channels);
        src1 = get3floats(sourceX0 + 3, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0 + 3) * source.Channels, source.Channels);

        CONVOLUTION1X3(values2, src0, src1, weights2);

        set4floats( values0, resultX, resultY, result.Width, result.Height, resultData, result.Channels );
        resultData += result.Width * result.Channels;
        set4floats( values1, resultX, resultY + 1, result.Width, result.Height, resultData, result.Channels );
        resultData += result.Width * result.Channels;
        set4floats( values2, resultX, resultY + 2, result.Width, result.Height, resultData, result.Channels );
    }
}

#define CONVOLUTION1X3_STRIDE2(acc, src0, src1, weights) \
    ({                                           \
      acc.x += src0.x * weights.x;               \
      acc.y += src0.z * weights.x;               \
        \
      acc.x += src0.y * weights.y;               \
      acc.y += src1.x * weights.y;               \
        \
      acc.x += src0.z * weights.z;               \
      acc.y += src1.y * weights.z;               \
    }) 

kernel void cubeKernelBlobChannelwiseConvolution3x3Stride2x2( constant CBaseBlobDesc& source [[buffer(0)]],
                                                              constant float* sourceData [[buffer(1)]],
                                                              constant CBaseBlobDesc& filt [[buffer(2)]],
                                                              constant float* filtData [[buffer(3)]],
                                                              constant int& paddingHeight [[buffer(4)]],
                                                              constant int& paddingWidth [[buffer(5)]],
                                                              constant CBaseBlobDesc& result [[buffer(6)]],
                                                              device float* resultData [[buffer(7)]],
                                                              uint3 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C3DPosition pos( thread_position_in_grid );
    
    const int combineW = 2;
    const int combineH = 2;

    int channelBlocksCount = ( result.Height / combineH + ( result.Height % combineH != 0 ) ) * ( result.Width / combineW + ( result.Width % combineW != 0 ) );
    
    int b;
    int g;
    int c;
    if( pos.GetMetalTaskIndex3D( result.BatchSize(), channelBlocksCount, result.Channels, b, g, c ) ) {
        
        const int blocksInRow = result.Width / combineW + ( result.Width % combineW != 0 );

        filtData += c;

        sourceData += b * source.ObjectSize() + c;

        const int sourceX0 = -paddingWidth + 2 * ( g % blocksInRow ) * combineW;
        int sourceY = -paddingHeight + 2 * ( g / blocksInRow ) * combineH;
        int sourceGY = sourceY * source.Width;

        const int resultX = ( g % blocksInRow ) * combineW;
        const int resultY = ( g / blocksInRow ) * combineH;
        resultData += b * result.ObjectSize() + ( result.Width * resultY + resultX ) * result.Channels + c;

        float2 values0 = 0;
        float2 values1 = 0;

        float3 weights0 = float3(filtData[0], filtData[filt.Channels], filtData[2 * filt.Channels]);
        float3 weights1 = float3(filtData[3 * filt.Channels], filtData[4 * filt.Channels], filtData[5 * filt.Channels]);
        float3 weights2 = float3(filtData[6 * filt.Channels], filtData[7 * filt.Channels], filtData[8 * filt.Channels]);

        float3 src0 = get3floats(sourceX0, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0) * source.Channels, source.Channels);
        float2 src1 = get2floats(sourceX0 + 3, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0 + 3) * source.Channels, source.Channels);

        CONVOLUTION1X3_STRIDE2(values0, src0, src1, weights0);
        
        sourceY += 1;
        sourceGY += source.Width;
        src0 = get3floats(sourceX0, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0) * source.Channels, source.Channels);
        src1 = get2floats(sourceX0 + 3, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0 + 3) * source.Channels, source.Channels);

        CONVOLUTION1X3_STRIDE2(values0, src0, src1, weights1);

        sourceY += 1;
        sourceGY += source.Width;
        src0 = get3floats(sourceX0, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0) * source.Channels, source.Channels);
        src1 = get2floats(sourceX0 + 3, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0 + 3) * source.Channels, source.Channels);

        CONVOLUTION1X3_STRIDE2(values0, src0, src1, weights2);
        CONVOLUTION1X3_STRIDE2(values1, src0, src1, weights0);

        sourceY += 1;
        sourceGY += source.Width;
        src0 = get3floats(sourceX0, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0) * source.Channels, source.Channels);
        src1 = get2floats(sourceX0 + 3, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0 + 3) * source.Channels, source.Channels);

        CONVOLUTION1X3_STRIDE2(values1, src0, src1, weights1);

        sourceY += 1;
        sourceGY += source.Width;
        src0 = get3floats(sourceX0, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0) * source.Channels, source.Channels);
        src1 = get2floats(sourceX0 + 3, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0 + 3) * source.Channels, source.Channels);

        CONVOLUTION1X3_STRIDE2(values1, src0, src1, weights2);

        set2floats( values0, resultX, resultY, result.Width, result.Height, resultData, result.Channels );
        resultData += result.Width * result.Channels;
        set2floats( values1, resultX, resultY + 1, result.Width, result.Height, resultData, result.Channels );
    }
}

kernel void cubeKernelBlobChannelwiseConvolution3x3Stride2x2WithFreeTerm( constant CBaseBlobDesc& source [[buffer(0)]],
                                                                          constant float* sourceData [[buffer(1)]],
                                                                          constant CBaseBlobDesc& filt [[buffer(2)]],
                                                                          constant float* filtData [[buffer(3)]],
                                                                          constant int& paddingHeight [[buffer(4)]],
                                                                          constant int& paddingWidth [[buffer(5)]],
                                                                          constant float* freeTerm [[buffer(6)]],
                                                                          constant CBaseBlobDesc& result [[buffer(7)]],
                                                                          device float* resultData [[buffer(8)]],
                                                                          uint3 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C3DPosition pos( thread_position_in_grid );
    
    const int combineW = 2;
    const int combineH = 2;

    int channelBlocksCount = ( result.Height / combineH + ( result.Height % combineH != 0 ) ) * ( result.Width / combineW + ( result.Width % combineW != 0 ) );
    
    int b;
    int g;
    int c;
    if( pos.GetMetalTaskIndex3D( result.BatchSize(), channelBlocksCount, result.Channels, b, g, c ) ) {
        
        const int blocksInRow = result.Width / combineW + ( result.Width % combineW != 0 );

        filtData += c;

        sourceData += b * source.ObjectSize() + c;

        const int sourceX0 = -paddingWidth + 2 * ( g % blocksInRow ) * combineW;
        int sourceY = -paddingHeight + 2 * ( g / blocksInRow ) * combineH;
        int sourceGY = sourceY * source.Width;

        const int resultX = ( g % blocksInRow ) * combineW;
        const int resultY = ( g / blocksInRow ) * combineH;
        resultData += b * result.ObjectSize() + ( result.Width * resultY + resultX ) * result.Channels + c;

        float2 values0 = freeTerm[c];
        float2 values1 = freeTerm[c];

        float3 weights0 = float3(filtData[0], filtData[filt.Channels], filtData[2 * filt.Channels]);
        float3 weights1 = float3(filtData[3 * filt.Channels], filtData[4 * filt.Channels], filtData[5 * filt.Channels]);
        float3 weights2 = float3(filtData[6 * filt.Channels], filtData[7 * filt.Channels], filtData[8 * filt.Channels]);

        float3 src0 = get3floats(sourceX0, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0) * source.Channels, source.Channels);
        float2 src1 = get2floats(sourceX0 + 3, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0 + 3) * source.Channels, source.Channels);

        CONVOLUTION1X3_STRIDE2(values0, src0, src1, weights0);
        
        sourceY += 1;
        sourceGY += source.Width;
        src0 = get3floats(sourceX0, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0) * source.Channels, source.Channels);
        src1 = get2floats(sourceX0 + 3, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0 + 3) * source.Channels, source.Channels);

        CONVOLUTION1X3_STRIDE2(values0, src0, src1, weights1);

        sourceY += 1;
        sourceGY += source.Width;
        src0 = get3floats(sourceX0, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0) * source.Channels, source.Channels);
        src1 = get2floats(sourceX0 + 3, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0 + 3) * source.Channels, source.Channels);

        CONVOLUTION1X3_STRIDE2(values0, src0, src1, weights2);
        CONVOLUTION1X3_STRIDE2(values1, src0, src1, weights0);

        sourceY += 1;
        sourceGY += source.Width;
        src0 = get3floats(sourceX0, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0) * source.Channels, source.Channels);
        src1 = get2floats(sourceX0 + 3, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0 + 3) * source.Channels, source.Channels);

        CONVOLUTION1X3_STRIDE2(values1, src0, src1, weights1);

        sourceY += 1;
        sourceGY += source.Width;
        src0 = get3floats(sourceX0, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0) * source.Channels, source.Channels);
        src1 = get2floats(sourceX0 + 3, sourceY, source.Width, source.Height, sourceData, (sourceGY + sourceX0 + 3) * source.Channels, source.Channels);

        CONVOLUTION1X3_STRIDE2(values1, src0, src1, weights2);

        set2floats( values0, resultX, resultY, result.Width, result.Height, resultData, result.Channels );
        resultData += result.Width * result.Channels;
        set2floats( values1, resultX, resultY + 1, result.Width, result.Height, resultData, result.Channels );
    }
}
