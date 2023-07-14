/* Copyright © 2017-2024 ABBYY

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
#include <PrimitivesJit.h>
#include <CPUInfo.h>

namespace NeoML {

void AvxMultiplyMatrix( bool transA, bool transB,
	IMathEngine *engine,
	const float* aPtr, size_t aRowSize,
	const float* bPtr, size_t bRowSize,
	float* cPtr, size_t cRowSize,
	size_t m, size_t n, size_t k );

struct CAvxConvolutionDesc : public CConvolutionDesc {
	~CAvxConvolutionDesc() override = default;

	CAvxConvolutionDesc( IMathEngine* mathEngine, const CBlobDesc& source, const CBlobDesc& result, const CBlobDesc& filter,
		int paddingHeight, int paddingWidth, int strideHeight, int strideWidth, int dilationHeight, int dilationWidth );

	std::unique_ptr<CBlobConvolutionBase> BlobConvolution;
};

CAvxConvolutionDesc::CAvxConvolutionDesc( IMathEngine* mathEngine, const CBlobDesc& source, const CBlobDesc& result, const CBlobDesc& filter,
	int paddingHeight, int paddingWidth, int strideHeight, int strideWidth, int dilationHeight, int dilationWidth ) :
	BlobConvolution( CBlobConvolutionFabric::GetProperInstance( mathEngine,
		filter.BatchWidth(), filter.Channels() * filter.Depth(), filter.Height(), filter.Width(), source.Height(), source.Width(), 
		paddingHeight, paddingWidth, strideHeight, strideWidth,
		dilationHeight, dilationWidth, result.Height(), result.Width(), result.ObjectCount() ) )
{
}

class CAvxMathEngine : public ISimdMathEngine {
public:
	explicit CAvxMathEngine( IMathEngine* _mathEngine ) :
		mathEngine( _mathEngine ), primitives( _mathEngine ) {}

	CConvolutionDesc* InitBlobConvolution( const CBlobDesc& source, int paddingHeight, int paddingWidth,
		int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, const CBlobDesc& filter,
		const CBlobDesc& result ) const override;

	void BlobConvolution( const CConvolutionDesc& convDesc, const float* source,
		const float* filter, const float* freeTerm, float* result ) const override;
	void BlobConvolutionRowwise( const CConvolutionDesc& convDesc, const float* source,
		int sourceRowIndex, const float* filter, const float* freeTerm, float* result,
		int resultRowIndex, int resultRowCount ) const override;

	SgemmFunc GetSgemmFunction() const override;

	void Tanh( float* dst, const float* src, size_t dataSize ) override;
	void Exp( float* dst, const float* src, size_t dataSize ) override;
	void RunOnceRestOfLstm( CMathEngineLstmDesc* desc, int sequenceCount, float* fullyConnectedResult,
		const float* inputStateBackLink, float* outputStateBackLink, float* outputMainBackLink ) override;

private:
	IMathEngine* const mathEngine;
	CPrimitivesJit primitives;
};

CConvolutionDesc* CAvxMathEngine::InitBlobConvolution( const CBlobDesc& source, int paddingHeight, int paddingWidth,
	int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, const CBlobDesc& filter,
	const CBlobDesc& result ) const
{
	if( !CCPUInfo::HasAvx512
		&& CBlobConvolutionFabric::IsBlobConvolutionAvailable( source.ObjectCount() * source.Height() * source.Width(),
			filter.BatchWidth() , filter.Height(), filter.Width() ) )
	{
		return new CAvxConvolutionDesc( mathEngine, source, result, filter, paddingHeight, paddingWidth, strideHeight, strideWidth, dilationHeight, dilationWidth );
	}
	return nullptr;
}

void CAvxMathEngine::BlobConvolution( const CConvolutionDesc& convDesc, const float* source,
	const float* filter, const float* freeTerm, float* result ) const
{
	const CAvxConvolutionDesc& desc = static_cast<const CAvxConvolutionDesc&>( convDesc );
	desc.BlobConvolution->ProcessConvolution( source, filter, freeTerm, result );
}

void CAvxMathEngine::BlobConvolutionRowwise( const CConvolutionDesc& convDesc, const float* source,
	int sourceRowIndex, const float* filter, const float* freeTerm, float* result,
	int resultRowIndex, int resultRowCount ) const
{
	const CAvxConvolutionDesc& desc = static_cast<const CAvxConvolutionDesc&>( convDesc );
	desc.BlobConvolution->ProcessConvolutionRowwise( source, sourceRowIndex, filter, freeTerm,
		result, resultRowIndex, resultRowCount );
}

SgemmFunc CAvxMathEngine::GetSgemmFunction() const
{
	return AvxMultiplyMatrix;
}

void CAvxMathEngine::Tanh( float* dst, const float* src, size_t dataSize )
{
	primitives.Tanh( dst, src, dataSize );
}

void CAvxMathEngine::Exp( float* dst, const float* src, size_t dataSize )
{
	primitives.Exp( dst, src, dataSize );
}

void CAvxMathEngine::RunOnceRestOfLstm( CMathEngineLstmDesc* desc, int sequenceCount, float* fullyConnectedResult,
	const float* inputStateBackLink, float* outputStateBackLink, float* outputMainBackLink )
{
	primitives.RestOfLstm( desc, sequenceCount, fullyConnectedResult, inputStateBackLink, outputStateBackLink,
		outputMainBackLink );
}

extern "C"
FME_DLL_EXPORT
ISimdMathEngine* CreateSimdMathEngine( IMathEngine* mathEngine )
{
	try {
		return new CAvxMathEngine( mathEngine );
	} catch( ... ) {
		// We cannot throw any exception from C function
		return nullptr;
	}
}

}
