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

#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoMathEngine/CrtAllocatedObject.h>

namespace NeoML {

// The general convolution descriptor
struct CCommonConvolutionDesc : public CConvolutionDesc {
	CBlobDesc Source;
	CBlobDesc Result;
	CBlobDesc Filter;
	int PaddingHeight;
	int PaddingWidth;
	int StrideHeight;
	int StrideWidth;
	int DilationHeight;
	int DilationWidth;

	CCommonConvolutionDesc( const CBlobDesc& source, const CBlobDesc& result, const CBlobDesc& filter,
			int paddingHeight, int paddingWidth, int strideHeight, int strideWidth, int dilationHeight, int dilationWidth ) :
		Source( source ),
		Result( result ),
        Filter( filter ),
		PaddingHeight( paddingHeight ),
		PaddingWidth( paddingWidth ),
		StrideHeight( strideHeight ),
		StrideWidth( strideWidth ),
		DilationHeight( dilationHeight ),
		DilationWidth( dilationWidth )
	{
	}
};

// The general 3d convolution descriptor
struct CCommon3dConvolutionDesc : public C3dConvolutionDesc {
	CBlobDesc Source;
	CBlobDesc Result;
	CBlobDesc Filter;
	int PaddingHeight;
	int PaddingWidth;
	int PaddingDepth;
	int StrideHeight;
	int StrideWidth;
	int StrideDepth;

	CCommon3dConvolutionDesc( const CBlobDesc& source, const CBlobDesc& result, const CBlobDesc& filter,
			int paddingHeight, int paddingWidth, int paddingDepth, int strideHeight, int strideWidth, int strideDepth ) :
		Source( source ),
        Result( result ),
        Filter( filter ),
 		PaddingHeight( paddingHeight ),
		PaddingWidth( paddingWidth ),
		PaddingDepth( paddingDepth ),
		StrideHeight( strideHeight ),
		StrideWidth( strideWidth ),
		StrideDepth( strideDepth )
	{
	}
};

// The general time convolution descriptor
struct CCommonTimeConvolutionDesc : public CTimeConvolutionDesc {
	CBlobDesc Source;
	CBlobDesc Filter;
	CBlobDesc Result;
	int Stride;
	int Padding;
	int Dilation;

	CCommonTimeConvolutionDesc( const CBlobDesc& source, const CBlobDesc& filter, const CBlobDesc& result,
			int stride, int padding, int dilation ) :
		Source( source ),
        Filter( filter ),
		Result( result ),
        Stride( stride ),
        Padding( padding ),
        Dilation( dilation )
	{
	}
};

// The general channelwise convolution descriptor
struct CCommonChannelwiseConvolutionDesc : public CChannelwiseConvolutionDesc {
	int PaddingHeight;
	int PaddingWidth;
	int StrideHeight;
	int StrideWidth;
	CBlobDesc Source;
	CBlobDesc Filter;
	CBlobDesc Result;

	CCommonChannelwiseConvolutionDesc( int paddingHeight, int paddingWidth, int strideHeight, int strideWidth,
			const CBlobDesc& source, const CBlobDesc& filter, const CBlobDesc& result ) : 
		PaddingHeight( paddingHeight ),
		PaddingWidth( paddingWidth ),
		StrideHeight( strideHeight ),
		StrideWidth( strideWidth ),
		Source( source ),
		Filter( filter ),
		Result( result )
	{
	}
};

} // namespace NeoML
