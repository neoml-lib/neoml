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

#include <memory>

#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoMathEngine/CrtAllocatedObject.h>

namespace NeoML {

// Empirical upper limit of a matrix size to effective JIT optimization
static constexpr int SMMD_MaxHeight = 128 + 1;

// The array of descriptors of the small matrices multiplication optimization by MKL JIT
// Enabled only for x86/x64 platform, and for CPU mathEngine
template<int Size = SMMD_MaxHeight>
class CCpuSmallMatricesMultiplyDescsArray : public CSmallMatricesMultiplyDescsArray {
public:
	CCpuSmallMatricesMultiplyDescsArray( IMathEngine& mathEngine ) : mathEngine( mathEngine ) {}
	CCpuSmallMatricesMultiplyDescsArray( const CCpuSmallMatricesMultiplyDescsArray& ) = delete;
	CCpuSmallMatricesMultiplyDescsArray( CCpuSmallMatricesMultiplyDescsArray&& ) = default;
	~CCpuSmallMatricesMultiplyDescsArray() override = default;

	// Destroy all the optimization descriptors in the array.
	// Should be called on a reshape of a layer.
	void DestroyAll();

	// Method to get exact optimization descriptor from the array.
	// If desired descriptor by index is not created, creates it using other method's arguments.
	// Returns nullptr if optimization descriptor impossible to create, otherwise returns pointer to it.
	const CSmallMatricesMultiplyDesc* Get( int index,
		int firstHeight, int firstWidth, int secondWidth, int secondRowSize, int resultWidth,
		bool resultAdd, bool trans1, bool trans2 ) const;
	// Method to get exact optimization descriptor from the array.
	const CSmallMatricesMultiplyDesc* Get( int index,
		int firstHeight, int firstWidth, int secondWidth, int resultWidth,
		bool resultAdd = false, bool trans1 = false, bool trans2 = true ) const
	{ return Get( index, firstHeight, firstWidth, secondWidth, secondWidth, resultWidth, resultAdd, trans1, trans2 ); }

private:
	IMathEngine& mathEngine; // CPU mathEngine
	mutable std::unique_ptr<CSmallMatricesMultiplyDesc> descs[Size]{}; // Array of descriptors
};

template<int Size>
inline const CSmallMatricesMultiplyDesc* CCpuSmallMatricesMultiplyDescsArray<Size>::Get( int index,
	int firstHeight, int firstWidth, int secondWidth, int secondRowSize, int resultWidth,
	bool resultAdd, bool trans1, bool trans2 ) const
{
	if( index >= Size ) {
		return nullptr;
	}
	if( descs[index] == nullptr ) {
		CSmallMatricesMultiplyDesc* ptr = mathEngine.InitSmallMatricesMultiplyDesc(
			firstHeight, firstWidth, secondWidth, secondRowSize, resultWidth, resultAdd, trans1, trans2 );
		PRESUME_EXPR( ptr != nullptr );
		descs[index].reset( ptr );
	}
	return descs[index].get();
}

template<int Size>
inline void CCpuSmallMatricesMultiplyDescsArray<Size>::DestroyAll()
{
	for( int i = 0; i < Size; ++i ) {
		descs[i].reset( nullptr );
	}
}

//--------------------------------------------------------------------------------------------------------------------

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

	enum { TSMMD_3DForward, TSMMD_3DBackward, TSMMD_3DLearn, /*...*/ TSMMD_3DCount_ };
	// The array of matrices multiplication optimization descriptors by enum above
	mutable CCpuSmallMatricesMultiplyDescsArray<TSMMD_3DCount_> smallMatricesMultiplyDescs;

	CCommon3dConvolutionDesc( IMathEngine& mathEngine,
			const CBlobDesc& source, const CBlobDesc& result, const CBlobDesc& filter,
			int paddingHeight, int paddingWidth, int paddingDepth,
			int strideHeight, int strideWidth, int strideDepth ) :
		Source( source ),
        Result( result ),
        Filter( filter ),
 		PaddingHeight( paddingHeight ),
		PaddingWidth( paddingWidth ),
		PaddingDepth( paddingDepth ),
		StrideHeight( strideHeight ),
		StrideWidth( strideWidth ),
		StrideDepth( strideDepth ),
		smallMatricesMultiplyDescs( mathEngine )
	{}
};

// The general time convolution descriptor
struct CCommonTimeConvolutionDesc : public CTimeConvolutionDesc {
	CBlobDesc Source;
	CBlobDesc Filter;
	CBlobDesc Result;
	int Stride;
	int PaddingFront;
	int PaddingBack;
	int Dilation;

	enum { TSMMD_Forward_MxMT, TSMMD_Forward_MxMT_Add, TSMMD_Backward_MxM_Add, TSMMD_Learn_MTxM_Add,
		/*...*/ TSMMD_Count_ };
	// The array of matrices multiplication optimization descriptors by enum above
	mutable CCpuSmallMatricesMultiplyDescsArray<TSMMD_Count_> smallMatricesMultiplyDescs;

	CCommonTimeConvolutionDesc( IMathEngine& mathEngine,
			const CBlobDesc& source, const CBlobDesc& result, const CBlobDesc& filter,
			int stride, int paddingFront, int paddingBack, int dilation ) :
		Source( source ),
        Filter( filter ),
		Result( result ),
        Stride( stride ),
		PaddingFront( paddingFront ),
		PaddingBack( paddingBack ),
        Dilation( dilation ),
		smallMatricesMultiplyDescs( mathEngine )
	{}
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
