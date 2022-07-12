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

#pragma once

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_CUDA

#include <NeoMathEngine/CrtAllocatedObject.h>
#include <NeoMathEngine/BlobDesc.h>

namespace NeoML {

// CUDA blob descriptor
class CCudaBlobDesc : public CCrtAllocatedObject {
public:
	static const int MaxDimensions = BD_Count;

	__host__ CCudaBlobDesc( const CBlobDesc& blob )
	{
		blob.GetDimSizes( dimensions );
	}

	__host__ __device__ CCudaBlobDesc()
	{
		for( int i = 0; i < MaxDimensions; i++ ) {
			dimensions[i] = 1;
		}
	}

	// Maximum sequence length for a recurrent network
	__host__ __device__ int BatchLength() const { return dimensions[BD_BatchLength]; }
	// The number of sequences in the blob
	__host__ __device__ int BatchWidth() const { return dimensions[BD_BatchWidth]; }
	// Object list size
	__host__ __device__ int ListSize() const { return dimensions[BD_ListSize]; }
	// Image height
	__host__ __device__ int Height() const { return dimensions[BD_Height]; }
	// Image width
	__host__ __device__ int Width() const { return dimensions[BD_Width]; }
	// Image depth
	__host__ __device__ int Depth() const { return dimensions[BD_Depth]; }
	// The number of "color" channels
	__host__ __device__ int Channels() const { return dimensions[BD_Channels]; }
	// The blob size
	__host__ __device__ int BlobSize() const
	{
		int blobSize = 1;
		for( int i = 0; i < MaxDimensions; i++ ) {
			blobSize *= dimensions[i];
		}
		return blobSize;
	}
	// The size of an object in the blob
	__host__ __device__ int ObjectSize() const { return Height() * Width() * Depth() * Channels(); }
	// The number of objects in the blob
	__host__ __device__ int ObjectCount() const { return BatchLength() * BatchWidth() * ListSize(); }
	// The geometrical size
	__host__ __device__ int GeometricalSize() const { return Height() * Width() * Depth(); }
	
	// The size of the specified dimension
	__host__ __device__ int DimSize( int d ) const
	{
		return dimensions[d];
	}

private:
	int dimensions[BD_Count];
};

//------------------------------------------------------------------------------------------------------------

inline __device__ int GetBlobPos( const CCudaBlobDesc& blob, int b, int h, int w, int c )
{
	const int maxDim = CCudaBlobDesc::MaxDimensions;
	return ((b * blob.DimSize(maxDim - 4) + h) * blob.DimSize(maxDim - 3) + w) * blob.DimSize(maxDim - 2) * blob.DimSize(maxDim - 1) + c;
}

template<class T>
inline __device__ T* GetBlobPtr( const CCudaBlobDesc& __restrict__ blob, T* data, int b, int h, int w, int c )
{
	return data + GetBlobPos(blob, b, h, w, c);
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
