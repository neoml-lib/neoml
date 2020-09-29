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

#include <AvxDll.h>
#include <MathEngineCommon.h>
#include <MemoryHandleInternal.h>

namespace NeoML {

CAvxDll::CAvxDll() : isLoaded( false )
{
	if( !Load( libName ) ) {
		return;
	}

	loadFunction( TFunctionPointers::BlobConvolution_avx_f9x9_c24_fc24, "BlobConvolution_avx_f9x9_c24_fc24" );

	isLoaded = true;
}

CAvxDll& CAvxDll::GetInstance()
{
	static CAvxDll instance;
	return instance;
}

void CAvxDll::loadFunction( TFunctionPointers functionType, const char* functionName )
{
	void* functionAdress = reinterpret_cast<void*>( GetProcAddress( functionName ) );
	ASSERT_EXPR( functionAdress != nullptr );

	functionAdresses[static_cast<size_t>(functionType)] = functionAdress;
}

void CAvxDll::CallBlobConvolution_avx_f9x9_c24_fc24( IMathEngine& mathEngine, int threadCount, const CCommonConvolutionDesc& desc,
	const CFloatHandle& sourceData, const CFloatHandle& filterData, const CFloatHandle* freeTermData, const CFloatHandle& resultData ) const
{
	typedef void ( *FuncType )( IMathEngine& mathEngine, int threadCount, const CCommonConvolutionDesc&, const CFloatHandle&, const CFloatHandle&, const CFloatHandle*, const CFloatHandle& );
	FuncType func = reinterpret_cast<FuncType>( functionAdresses.at( static_cast<size_t>( TFunctionPointers::BlobConvolution_avx_f9x9_c24_fc24 ) ) );

	ASSERT_EXPR( func != nullptr );

	ASSERT_EXPR( desc.Filter.Channels() == 24 );
	ASSERT_EXPR( desc.Filter.ObjectCount() == 24 );
	ASSERT_EXPR( desc.PaddingWidth == desc.PaddingHeight );
	ASSERT_EXPR( desc.DilationWidth == desc.DilationHeight );
	ASSERT_EXPR( desc.PaddingWidth == desc.DilationHeight );
	ASSERT_EXPR( desc.StrideWidth == desc.StrideHeight );
	ASSERT_EXPR( desc.Filter.Width() == 3 );
	ASSERT_EXPR( desc.Filter.Height() == 3 );

	ASSERT_EXPR( reinterpret_cast<unsigned long>( GetRaw( sourceData ) ) % 32 == 0 );
	ASSERT_EXPR( reinterpret_cast<unsigned long>(  GetRaw( resultData )  ) % 32 == 0 );

	func( mathEngine, threadCount, desc, sourceData, filterData, freeTermData, resultData );
}

}
