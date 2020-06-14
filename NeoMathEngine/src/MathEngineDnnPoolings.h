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

// The general max pooling descriptor
struct CCommonMaxPoolingDesc : public CMaxPoolingDesc {
	CBlobDesc Source;
	CBlobDesc Result;
	int FilterHeight;
	int FilterWidth;
	int StrideHeight;
	int StrideWidth;

	CCommonMaxPoolingDesc( const CBlobDesc& source, const CBlobDesc& result,
			int filterHeight, int filterWidth, int strideHeight, int strideWidth ) :
		Source( source ),
		Result( result ),
		FilterHeight( filterHeight ),
		FilterWidth( filterWidth ),
		StrideHeight( strideHeight ),
		StrideWidth( strideWidth )
	{
	}
};

// The general mean pooling descriptor
struct CCommonMeanPoolingDesc : public CMeanPoolingDesc {
	CBlobDesc Source;
	CBlobDesc Result;
	int FilterHeight;
	int FilterWidth;
	int StrideHeight;
	int StrideWidth;

	CCommonMeanPoolingDesc( const CBlobDesc& source, const CBlobDesc& result,
			int filterHeight, int filterWidth, int strideHeight, int strideWidth ) :
		Source( source ),
		Result( result ),
		FilterHeight( filterHeight ),
		FilterWidth( filterWidth ),
		StrideHeight( strideHeight ),
		StrideWidth( strideWidth )
	{
	}
};

// The general max over time pooling descriptor
struct CCommonGlobalMaxOverTimePoolingDesc : public CGlobalMaxOverTimePoolingDesc {
	CBlobDesc Source;
	CBlobDesc Result;

	CCommonGlobalMaxOverTimePoolingDesc( const CBlobDesc& source, const CBlobDesc& result ) :
		Source( source ),
		Result( result )
	{
	}
};

// The general global max pooling descriptor
struct CCommonGlobalMaxPoolingDesc : public CGlobalMaxPoolingDesc {
	CBlobDesc Source;
	CBlobDesc MaxIndices;
	CBlobDesc Result;

	CCommonGlobalMaxPoolingDesc( const CBlobDesc& source, const CBlobDesc& maxIndices, const CBlobDesc& result ) :
		Source( source ),
		MaxIndices( maxIndices ),
		Result( result )
	{
	}
};

// The general 3d max pooling descriptor
struct CCommon3dMaxPoolingDesc : public C3dMaxPoolingDesc {
	CBlobDesc Source;
	CBlobDesc Result;
	int FilterHeight;
	int FilterWidth;
	int FilterDepth;
	int StrideHeight;
	int StrideWidth;
	int StrideDepth;

	CCommon3dMaxPoolingDesc( const CBlobDesc& source, const CBlobDesc& result, int filterHeight, int filterWidth,
			int filterDepth, int strideHeight, int strideWidth, int strideDepth ) :
		Source( source ),
		Result( result ),
		FilterHeight( filterHeight ),
		FilterWidth( filterWidth ),
		FilterDepth( filterDepth ),
		StrideHeight( strideHeight ),
		StrideWidth( strideWidth ),
		StrideDepth( strideDepth )
	{
	}
};

// The general 3d mean pooling descriptor
struct CCommon3dMeanPoolingDesc : public C3dMeanPoolingDesc {
	CBlobDesc Source;
	CBlobDesc Result;
	int FilterHeight;
	int FilterWidth;
	int FilterDepth;
	int StrideHeight;
	int StrideWidth;
	int StrideDepth;

	CCommon3dMeanPoolingDesc( const CBlobDesc& source, const CBlobDesc& result, int filterHeight, int filterWidth, int filterDepth,
			int strideHeight, int strideWidth, int strideDepth ) :
		Source( source ),
		Result( result ),
		FilterHeight( filterHeight ),
		FilterWidth( filterWidth ),
		FilterDepth( filterDepth ),
		StrideHeight( strideHeight ),
		StrideWidth( strideWidth ),
		StrideDepth( strideDepth )
	{
	}
};

// The general max over time pooling descriptor
struct CCommonMaxOverTimePoolingDesc : public CMaxOverTimePoolingDesc {
	CBlobDesc Source;
	CBlobDesc Result;
	int FilterLen;
	int StrideLen;

	CCommonMaxOverTimePoolingDesc( const CBlobDesc& source, const CBlobDesc& result, int filterLen, int strideLen ) :
		Source( source ),
		Result( result ),
		FilterLen( filterLen ),
		StrideLen( strideLen )
	{
	}
};

} // namespace NeoML
