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

namespace NeoML {

extern "C"
FME_DLL_EXPORT bool IsBlobConvolutionAvailable( int filterCount, int channelCount, int filterHeight, int filterWidth )
{
	return CBlobConvolutionFabric::IsBlobConvolutionAvailable( filterCount, channelCount, filterHeight, filterWidth );
}

extern "C"
FME_DLL_EXPORT bool BlobConvolution( int filterCount, int channelCount, int filterHeight, int filterWidth, int threadCount,
	int sourceHeight, int sourceWidth, int strideHeight, int strideWidth,
	int dilationHeight, int dilationWidth, int resultHeight, int resultWidth,
	const float* sourceData, const float* filterData, const float* freeTermData, float* resultData )
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

	if( blobConvolutionInst != nullptr ) {
		blobConvolutionInst->ProcessConvolution( threadCount );
		return true;
	}
	return false;
}

}
