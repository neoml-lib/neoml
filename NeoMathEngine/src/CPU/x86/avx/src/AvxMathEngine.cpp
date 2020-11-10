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

#include <NeoMathEngine/SimdMathEngine.h>
#include <BlobConvolution.h>

namespace NeoML {

struct CAvxConvolutionDesc : public CConvolutionDesc {
	~CAvxConvolutionDesc() override {}

	CAvxConvolutionDesc( IMathEngine* mathEngine, const CBlobDesc& source, const CBlobDesc& result, const CBlobDesc& filter,
		int paddingHeight, int paddingWidth, int strideHeight, int strideWidth, int dilationHeight, int dilationWidth );

	std::unique_ptr<CBlobConvolutionBase> BlobConvolution;
};

CAvxConvolutionDesc::CAvxConvolutionDesc( IMathEngine* mathEngine, const CBlobDesc& source, const CBlobDesc& result, const CBlobDesc& filter,
	int paddingHeight, int paddingWidth, int strideHeight, int strideWidth, int dilationHeight, int dilationWidth ) :
	BlobConvolution( CBlobConvolutionFabric::GetProperInstance( mathEngine,
		filter.BatchWidth(), filter.Channels(), filter.Height(), filter.Width(), source.Height(), source.Width(), 
		paddingHeight, paddingWidth, strideHeight, strideWidth,
		dilationHeight, dilationWidth, result.Height(), result.Width() ) )
{
}

class CAvxMathEngine : public ISimdMathEngine {
public:
	CAvxMathEngine( IMathEngine* _mathEngine, int _threadCount ) : mathEngine( _mathEngine ), threadCount( _threadCount ) {}

	CConvolutionDesc* InitBlobConvolution( const CBlobDesc& source, int paddingHeight, int paddingWidth,
		int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, const CBlobDesc& filter,
		const CBlobDesc& result ) const override;

	void BlobConvolution( const CConvolutionDesc& convDesc, const float* source,
		const float* filter, const float* freeTerm, float* result ) const override;

private:
	IMathEngine* mathEngine;
	int threadCount;
};

CConvolutionDesc* CAvxMathEngine::InitBlobConvolution( const CBlobDesc& source, int paddingHeight, int paddingWidth,
	int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, const CBlobDesc& filter,
	const CBlobDesc& result ) const
{
	if( CBlobConvolutionFabric::IsBlobConvolutionAvailable( filter.BatchWidth() , filter.Height(), filter.Width() ) ) {
		return new CAvxConvolutionDesc( mathEngine, source, result, filter, paddingHeight, paddingWidth, strideHeight, strideWidth, dilationHeight, dilationWidth );
	}
	return nullptr;
}

void CAvxMathEngine::BlobConvolution( const CConvolutionDesc& convDesc, const float* source,
	const float* filter, const float* freeTerm, float* result ) const
{
	const CAvxConvolutionDesc& desc = static_cast<const CAvxConvolutionDesc&>( convDesc );
	
	desc.BlobConvolution->ProcessConvolution( threadCount, source, filter, freeTerm, result );

}

extern "C"
FME_DLL_EXPORT
ISimdMathEngine* CreateSimdMathEngine( IMathEngine* mathEngine, int threadCount )
{
	try {
		return new CAvxMathEngine( mathEngine, threadCount );
	} catch( ... ) {
		// We cannot throw any exception from C function
		return nullptr;
	}
}

}
