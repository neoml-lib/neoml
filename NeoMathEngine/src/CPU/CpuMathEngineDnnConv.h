/* Copyright © 2017-2023 ABBYY

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

namespace NeoML {

// The algorithm used to calculate a 2D convolution
enum TConvAlgo {
	CA_Auto,	// choose automatically
	CA_1,		// use a temporary matrix to store the data in another order
	CA_2,		// work with the data directly (only for stride = 1 and padding = 0)
				// most efficient when the image is large and especially when it has many channels
				
	CA_1x1		// for convolution with a 1*1 filter, no padding and dilation (both 2D and 3D)
};

constexpr int BlobConvolutionCacheSize = 256 * 1024;

// Convolution descriptor
struct CCpuConvolutionDesc : public CCommonConvolutionDesc {
	TConvAlgo ForwardAlgo;
	TConvAlgo BackwardAlgo;
	std::unique_ptr<CConvolutionDesc> SimdConvolutionDesc;

	CCpuConvolutionDesc( const CBlobDesc& source, const CBlobDesc& result, const CBlobDesc& filter,
			int paddingHeight, int paddingWidth, int strideHeight, int strideWidth, int dilationHeight, int dilationWidth ) :
		CCommonConvolutionDesc( source, result, filter, paddingHeight, paddingWidth, strideHeight, strideWidth, dilationHeight, dilationWidth ),
		ForwardAlgo( getActualForwardAlgo() ),
		BackwardAlgo( getActualBackwardAlgo() )
	{
	}

	TConvAlgo getActualForwardAlgo() const;
	TConvAlgo getActualBackwardAlgo() const;
};

// Gets the algorithm to be used for this convolution
inline TConvAlgo CCpuConvolutionDesc::getActualForwardAlgo() const
{
	if( PaddingHeight == 0 && PaddingWidth == 0
		&& DilationHeight == 1 && DilationWidth == 1
		&& Filter.ObjectSize() == Filter.Channels() )
	{
		return CA_1x1;
	}

	if( DilationHeight == 1 && DilationWidth == 1 && StrideHeight == 1 && StrideWidth == 1 ) {
		if( PaddingHeight > 0 || PaddingWidth > 0 ) {
			if( ( Source.Height() >= 64 && Source.Width() >= 64 && Source.Depth() * Source.Channels() >= 8 ) ||
				( Source.Height() >= 32 && Source.Width() >= 32 && Source.Depth() * Source.Channels() >= 16 ) )
			{
				return CA_2;
			}
		} else {
			if( ( Source.Height() >= 64 && Source.Width() >= 64 && Source.Depth() * Source.Channels() >= 4 ) ||
				( Source.Height() >= 32 && Source.Width() >= 32 && Source.Depth() * Source.Channels() >= 8 ) )
			{
				return CA_2;
			}
		}
	}
	return CA_1;
}

inline TConvAlgo CCpuConvolutionDesc::getActualBackwardAlgo() const
{
	TConvAlgo ret = getActualForwardAlgo();
	if( ret == CA_2 && ( PaddingHeight != 0 || PaddingWidth != 0 ) ) {
		ret = CA_1;
	}
	return ret;
}

inline int ceilTo( int val, int discret )
{
	if( val > 0 ) {
		return ( ( val + discret - 1 ) / discret ) * discret;
	}
	return ( val / discret ) * discret;
}

} // namespace NeoML
