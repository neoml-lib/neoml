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

//-------------------------------------------------------------------------------------------------------------------

static constant int BlobMergeByDimCombine = 16;
    
kernel void matrixKernelBlobMergeByDim( constant int* height [[buffer(0)]],
                                        constant int* width [[buffer(1)]],
                                        constant CBaseBlobDesc* from [[buffer(2)]],
                                        constant float* fromData [[buffer(3)]],
                                        constant CBaseBlobDesc* to [[buffer(4)]],
                                        device float* toData [[buffer(5)]],
                                        constant int* heightNorm [[buffer(6)]],
                                        constant int* wStart [[buffer(7)]],
                                        constant int* wLen [[buffer(8)]],
                                        uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    
    int j;
    int i;
    if( !pos.GetMetalTaskIndex2D( *heightNorm, *wLen, j, i ) ) {
        return;
    }
    
    j *= BlobMergeByDimCombine;
    int jLast = j + BlobMergeByDimCombine;
    if( jLast > *height ) {
        jLast = *height;
    }
    
    int count = jLast - j;
    
    toData += j * *width + *wStart + i;
    fromData += j * *wLen + i;
    
    for( int k = 0; k < count; ++k ) {
        *toData = *fromData;
        toData += *width;
        fromData += *wLen;
    }
}
 
inline int Repack( int fromIndex, int channels, int height, int width  ) {
	int x = fromIndex % width;
	fromIndex /= width;
	int y = fromIndex % height;
	fromIndex /= height;
	int c = fromIndex % channels;
	int b = fromIndex / channels;
	return c + channels * ( x + width * ( y + height * b ) );
}

kernel void blobReorgFloat( constant float* input [[buffer(0)]],
                            constant int& width [[buffer(1)]],
                            constant int& height [[buffer(2)]],
                            constant int& channels [[buffer(3)]],
                            constant int& objectCount [[buffer(4)]],
                            constant int& stride [[buffer(5)]],
                            constant bool& isForward [[buffer(6)]],
                            device float* output [[buffer(7)]],
                            uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
	C2DPosition pos( thread_position_in_grid );

	int i;
	int j;
	if( !pos.GetMetalTaskIndex2D( objectCount * height, channels * width, i, j ) ) {
		return;
	}

	int b = i / height;
	int h = i - b * height;
	int c = j / width;
	int w = j - c * width;

	int outputChannels = channels / ( stride * stride );
	int inputIndex = w + width * ( h + height * ( c + channels * b ) );
	inputIndex = Repack( inputIndex, channels * stride * stride, height / stride, width / stride );
	int offset = c / outputChannels;
	int outputW = w * stride + offset % stride;
	int outputH = h * stride + offset / stride;
	int outputChannelId = c % outputChannels;
	int outputIndex = outputW + width * stride * ( outputH + height * stride * ( outputChannelId + outputChannels * b ) );
	outputIndex = Repack( outputIndex, channels, height, width );
	if( isForward ) {
		output[inputIndex] = input[outputIndex];
	} else {
		output[outputIndex] = input[inputIndex];
	}
}

kernel void blobReorgInt( constant int* input [[buffer(0)]],
                          constant int& width [[buffer(1)]],
                          constant int& height [[buffer(2)]],
                          constant int& channels [[buffer(3)]],
                          constant int& objectCount [[buffer(4)]],
                          constant int& stride [[buffer(5)]],
                          constant bool& isForward [[buffer(6)]],
                          device int* output [[buffer(7)]],
                          uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
	C2DPosition pos( thread_position_in_grid );
    
	int i;
	int j;
	if( !pos.GetMetalTaskIndex2D( objectCount * height, channels * width, i, j ) ) {
		return;
	}

	int b = i / height;
	int h = i - b * height;
	int c = j / width;
	int w = j - c * width;

	int outputChannels = channels / ( stride * stride );
	int inputIndex = w + width * ( h + height * ( c + channels * b ) );
	inputIndex = Repack( inputIndex, channels * stride * stride, height / stride, width / stride );
	int offset = c / outputChannels;
	int outputW = w * stride + offset % stride;
	int outputH = h * stride + offset / stride;
	int outputChannelId = c % outputChannels;
	int outputIndex = outputW + width * stride * ( outputH + height * stride * ( outputChannelId + outputChannels * b ) );
	outputIndex = Repack( outputIndex, channels, height, width );
	if( isForward ) {
		output[inputIndex] = input[outputIndex];
	} else {
		output[outputIndex] = input[inputIndex];
	}
}

kernel void depthToSpaceFloat( constant float* input [[buffer(0)]],
                               constant int& dataRowCount [[buffer(1)]],
                               constant int& dataRowWidth [[buffer(2)]],
                               constant int& blockChannels [[buffer(3)]],
                               constant int& blockSize [[buffer(4)]],
                               constant bool& isForward [[buffer(5)]],
                               device float* output [[buffer(6)]],
                               uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );

    // number of elements in the single data row
    const int dataRowSize = blockSize * ( dataRowWidth * blockSize ) * blockChannels;

    int dataRowIndex;
    int elementIndex;
    if( !pos.GetMetalTaskIndex2D( dataRowCount, dataRowSize, dataRowIndex, elementIndex ) ) {
        return;
    }

    // number of elements in a single row inside 3d-block
    const int blockRowSize = blockChannels * blockSize;

    // offset for switching to the next data row
    // const int dataRowSize = blockSize * ( dataRowWidth * blockSize ) * blockChannels;
    // offset for switching to the next block inside data row
    const int sourceBlockOffset = isForward ? blockRowSize : blockSize * blockRowSize;
    const int resultBlockOffset = isForward ? blockSize * blockRowSize : blockRowSize;
    // offset for switching to the next row inside the 3d-block
    const int sourceBlockRowOffset = isForward ? dataRowWidth * blockRowSize : blockRowSize;
    const int resultBlockRowOffset = isForward ? blockRowSize : dataRowWidth * blockRowSize;

    const int pixelIndex = elementIndex / blockChannels;
    elementIndex %= blockChannels;
    const int inBlockX = pixelIndex % blockSize;
    const int inBlockY = ( pixelIndex / blockSize ) % blockSize;
    const int blockX = ( pixelIndex / blockSize / blockSize );

    source += dataRowIndex * dataRowSize + blockX * sourceBlockOffset + inBlockY * sourceBlockRowOffset
        + inBlockX * blockChannels + elementIndex;
    result += dataRowIndex * dataRowSize + blockX * resultBlockOffset + inBlockY * resultBlockRowOffset
        + inBlockX * blockChannels + elementIndex;
    *result = *source;
}

kernel void depthToSpaceInt( constant int* input [[buffer(0)]],
                             constant int& dataRowCount [[buffer(1)]],
                             constant int& dataRowWidth [[buffer(2)]],
                             constant int& blockChannels [[buffer(3)]],
                             constant int& blockSize [[buffer(4)]],
                             constant bool& isForward [[buffer(5)]],
                             device int* output [[buffer(6)]],
                             uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );

    // number of elements in the single data row
    const int dataRowSize = blockSize * ( dataRowWidth * blockSize ) * blockChannels;

    int dataRowIndex;
    int elementIndex;
    if( !pos.GetMetalTaskIndex2D( dataRowCount, dataRowSize, dataRowIndex, elementIndex ) ) {
        return;
    }

    // number of elements in a single row inside 3d-block
    const int blockRowSize = blockChannels * blockSize;

    // offset for switching to the next data row
    // const int dataRowSize = blockSize * ( dataRowWidth * blockSize ) * blockChannels;
    // offset for switching to the next block inside data row
    const int sourceBlockOffset = isForward ? blockRowSize : blockSize * blockRowSize;
    const int resultBlockOffset = isForward ? blockSize * blockRowSize : blockRowSize;
    // offset for switching to the next row inside the 3d-block
    const int sourceBlockRowOffset = isForward ? dataRowWidth * blockRowSize : blockRowSize;
    const int resultBlockRowOffset = isForward ? blockRowSize : dataRowWidth * blockRowSize;

    const int pixelIndex = elementIndex / blockChannels;
    elementIndex %= blockChannels;
    const int inBlockX = pixelIndex % blockSize;
    const int inBlockY = ( pixelIndex / blockSize ) % blockSize;
    const int blockX = ( pixelIndex / blockSize / blockSize );

    source += dataRowIndex * dataRowSize + blockX * sourceBlockOffset + inBlockY * sourceBlockRowOffset
        + inBlockX * blockChannels + elementIndex;
    result += dataRowIndex * dataRowSize + blockX * resultBlockOffset + inBlockY * resultBlockRowOffset
        + inBlockX * blockChannels + elementIndex;
    *result = *source;
}

static constant int BlobSplitByDimCombine = 16;
kernel void matrixKernelBlobSplitByDim( constant int* height [[buffer(0)]],
                                        constant int* width [[buffer(1)]],
                                        constant CBaseBlobDesc* from [[buffer(2)]],
                                        constant float* fromData [[buffer(3)]],
                                        constant CBaseBlobDesc* to [[buffer(4)]],
                                        device float* toData [[buffer(5)]],
                                        constant int* heightNorm [[buffer(6)]],
                                        constant int* wStart [[buffer(7)]],
                                        constant int* wLen [[buffer(8)]],
                                        uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
    
	int j;
	int i;
	if( !pos.GetMetalTaskIndex2D( *heightNorm, *wLen, j, i ) ) {
		return;
	}

	j *= BlobSplitByDimCombine;
	int jLast = j + BlobSplitByDimCombine;
	if( jLast > *height ) {
		jLast = *height;
	}

	int count = jLast - j;

	fromData += j * *width + *wStart + i;
	toData += j * *wLen + i;
    
	for(int k = 0; k < count; ++k) {
		*toData = *fromData;
		fromData += *width;
		toData += *wLen;
	}
}

struct CMetalRleStroke {
    short Start;
    short End;
};

struct CMetalRleImage {
    int StrokesCount;
    int Height;
    int Width;
    CMetalRleStroke Stub;
    CMetalRleStroke Lines[1];
};

kernel void matrixKernelBlobConvertFromRle( constant CBaseBlobDesc* source [[buffer(0)]],
                                            device float* sourceData [[buffer(1)]],
                                            constant int* objectSize [[buffer(2)]],
                                            constant float* stroke [[buffer(3)]],
                                            constant float* nonStroke [[buffer(4)]],
                                            constant CBaseBlobDesc* result [[buffer(5)]],
                                            device float* resultData [[buffer(6)]],
                                            uint2 thread_position_in_grid [[ thread_position_in_grid ]] )
{
    C2DPosition pos( thread_position_in_grid );
   
	int num;
	int line;
	if( !pos.GetMetalTaskIndex2D( source->BatchSize(), source->Height, num, line) ) {
		return;
	}

	device CMetalRleImage* image = reinterpret_cast<device CMetalRleImage*>( (device char*)sourceData + num * *objectSize );
	device float* output = GetFloatBlobPtr(*result, resultData, num, line, 0);

	int imageStart = (source->Height - image->Height) / 2;
	int imageStop = imageStart + image->Height;

	if(line < imageStart || line >= imageStop) {
		// Empty row
		for(int i = 0; i < result->Width; ++i) {
			*output++ = *nonStroke;
		}
		return;
	}

	// Find the row in the RLE image
	int lineToPass = line - imageStart;
	device const CMetalRleStroke* rleStroke = image->Lines;
	while(lineToPass > 0) {
		if(rleStroke->End < 0) {
			--lineToPass;
		}
		++rleStroke;
	}

	// Fill the row start with the empty value
	int startPos = (source->Width - image->Width) / 2;
	for(int i = 0; i < startPos; ++i) {
		*output++ = *nonStroke;
	}

	// Draw the strokes
	int position = 0;
	while(rleStroke->End >= 0) {
		for(; position < rleStroke->Start; ++position) {
			*output++ = *nonStroke;
		}
		for(; position < rleStroke->End; ++position) {
			*output++ = *stroke;
		}
		++rleStroke;
	}

	// Fill the row end with the empty value
	int rest = result->Width - position - startPos;
	for(int i = 0; i < rest; ++i) {
		*output++ = *nonStroke;
	}
}

constant static const int BlobResizeImageCombine = 16;

kernel void cubeKernelBlobResizeImage( constant CBaseBlobDesc& from [[buffer(0)]],
                                       device float* fromData [[buffer(1)]],
                                       constant int& deltaLeft [[buffer(2)]],
                                       constant int& deltaRight [[buffer(3)]],
                                       constant int& deltaTop [[buffer(4)]],
                                       constant int& deltaBottom [[buffer(5)]],
                                       constant float& defaultValue [[buffer(6)]],
                                       constant CBaseBlobDesc& to [[buffer(7)]],
                                       device float* toData [[buffer(8)]],
                                       uint3 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                       uint3 threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                       uint3 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]]  )
{
    C3DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    
	int totalChannels = to.Channels * to.Depth;
	int geom = to.Height * to.Width;
	int num;
	int channel;
	int hw;
	if( !pos.GetMetalTaskIndex3D( to.BatchSize(), totalChannels, geom, 1, 1, BlobResizeImageCombine, num, channel, hw ) ) {
		return;
	}

	int hwStep;
	int count = pos.GetMetalWidthTaskCountAndIndex( geom, BlobResizeImageCombine, hw, hwStep );

    fromData += num * from.ObjectSize() + channel;
    toData += num * to.ObjectSize() + channel;

	for(int k = 0; k < count; ++k) {
		int j = hw / to.Width;
		int i = hw % to.Width;
		int jFrom = j - deltaTop;
		int iFrom = i - deltaLeft;
		if(0 <= jFrom && jFrom < from.Height && 0 <= iFrom && iFrom < from.Width) {
			toData[hw * totalChannels] = fromData[(jFrom * from.Width + iFrom) * totalChannels];
		} else {
			toData[hw * totalChannels] = defaultValue;
		}
		hw += hwStep;
	}
}

kernel void cubeKernelBlobGetSubSequence( constant CBaseBlobDesc* from [[buffer(0)]],
                                          device const float* fromData [[buffer(1)]],
                                          constant CBaseBlobDesc* to [[buffer(2)]],
                                          device float* toData [[buffer(3)]],
                                          constant int* startPos [[buffer(4)]],
                                          constant int* isRev [[buffer(5)]],
                                          device int* index [[buffer(6)]],
                                          uint3 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                          uint3 threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                          uint3 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]]  )
{
    C3DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    
	int seqPos;
	int seqNum;
	int i;
    
    if( !pos.GetMetalTaskIndex3D( to->BatchLength, to->BatchWidth * to->ListSize, from->ObjectSize(), 1, 1, 16, seqPos, seqNum, i ) ) {
        return;
	}

	int objectSize = from->ObjectSize();

	int fromSeqPos = (*isRev != 0) ? ( *startPos - seqPos ) : ( *startPos + seqPos );
	int fromPos = fromSeqPos * from->BatchWidth * from->ListSize + seqNum;
	fromData += fromPos * objectSize;
	int toPos = seqPos * to->BatchWidth * to->ListSize + seqNum;
	toData += toPos * objectSize;

	int step;
	int count = pos.GetMetalWidthTaskCountAndIndex( objectSize, 16, i, step );

	if( i == 0 && count > 0 ) {
		index[toPos] = fromPos;
	}

	for( int k = 0; k < count; ++k ) {
		toData[i] = *(fromData + i);
		i += step;
	}
}
    
kernel void cubeKernelBlobGetSubSequenceNoIndex( constant CBaseBlobDesc* from [[buffer(0)]],
                                                 device const float* fromData [[buffer(1)]],
                                                 constant CBaseBlobDesc* to [[buffer(2)]],
                                                 device float* toData [[buffer(3)]],
                                                 constant int* startPos [[buffer(4)]],
                                                 constant int* isRev [[buffer(5)]],
                                                 uint3 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                                 uint3 threads_per_threadgroup        [[ threads_per_threadgroup ]],
                                                 uint3 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]]  )
{
    C3DCombinePosition pos( thread_position_in_threadgroup, threads_per_threadgroup, threadgroup_position_in_grid );
    
    int seqPos;
    int seqNum;
    int i;
    
    if( !pos.GetMetalTaskIndex3D( to->BatchLength, to->BatchWidth * to->ListSize, from->ObjectSize(), 1, 1, 16, seqPos, seqNum, i ) ) {
        return;
    }
    
    int objectSize = from->ObjectSize();
    
    int fromSeqPos = (*isRev != 0) ? (*startPos - seqPos) : (*startPos + seqPos);
    int fromPos = fromSeqPos * from->BatchWidth * from->ListSize + seqNum;
    fromData += fromPos * objectSize;
    int toPos = seqPos * to->BatchWidth * to->ListSize + seqNum;
    toData += toPos * objectSize;
    
    int step;
    int count = pos.GetMetalWidthTaskCountAndIndex( objectSize, 16, i, step );
    
    for( int k = 0; k < count; ++k ) {
        toData[i] = *(fromData + i);
        i += step;
    }
}
