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

#include <common.h>
#pragma hdrstop

#include <DnnConv.h>
#include <AvxDll.h>

namespace NeoML {

class CAvxDll : public IAvxDll {
public:
	CAvxDll( int filterCount_, int channelCount_, int filterHeight_, int filterWidth_,
		int sourceHeight_, int sourceWidth_, int strideHeight_, int strideWidth_,
		int dilationHeight_, int dilationWidth_, int resultHeight_, int resultWidth_ ) :
			filterCount( filterCount_ ), channelCount( channelCount_ ), filterHeight( filterHeight_ ), filterWidth( filterWidth_ ),
			sourceHeight( sourceHeight_ ), sourceWidth( sourceWidth_ ), strideHeight( strideHeight_ ), strideWidth( strideWidth_ ),
			dilationHeight( dilationHeight_ ), dilationWidth( dilationWidth_ ), resultHeight( resultHeight_ ), resultWidth( resultWidth_ )
	{
	}
	bool IsBlobConvolutionAvailable() const override;
	void BlobConvolution( int threadCount,	const float* sourceData, const float* filterData, const float* freeTermData, float* resultData ) const override;
private:
	const int filterCount;
	const int channelCount;
	const int filterHeight;
	const int filterWidth;
	const int sourceHeight;
	const int sourceWidth;
	const int strideHeight;
	const int strideWidth;
	const int dilationHeight;
	const int dilationWidth;
	const int resultHeight;
	const int resultWidth;
};

bool CAvxDll::IsBlobConvolutionAvailable() const
{
	return CBlobConvolutionFabric::IsBlobConvolutionAvailable( filterCount, channelCount, filterHeight, filterWidth );
}

void CAvxDll::BlobConvolution( int threadCount,
	const float* sourceData, const float* filterData, const float* freeTermData, float* resultData ) const
{
	// Data should be alligned
	ASSERT_EXPR( reinterpret_cast<std::uintptr_t>( sourceData ) % 32 == 0 );
	ASSERT_EXPR( reinterpret_cast<std::uintptr_t>( resultData ) % 32 == 0 );

	auto blobConvolutionInst = CBlobConvolutionFabric::GetProperInstance( filterCount,
		channelCount, filterHeight, filterWidth,
		sourceHeight, sourceWidth, strideHeight, strideWidth,
		dilationHeight, dilationWidth, resultHeight, resultWidth,
		sourceData, filterData, freeTermData, resultData );

	ASSERT_EXPR( blobConvolutionInst != nullptr );
	
	blobConvolutionInst->ProcessConvolution( threadCount );

}

extern "C"
FME_DLL_EXPORT
IAvxDll* GetAvxDllInstance(
	int filterCount, int channelCount, int filterHeight, int filterWidth,
	int sourceHeight, int sourceWidth, int strideHeight, int strideWidth,
	int dilationHeight, int dilationWidth, int resultHeight, int resultWidth )
{
	return new CAvxDll(
		filterCount, channelCount, filterHeight, filterWidth,
		sourceHeight, sourceWidth, strideHeight, strideWidth,
		dilationHeight, dilationWidth, resultHeight, resultWidth );
}

}
