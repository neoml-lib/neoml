/* Copyright © 2017-2022 ABBYY Production LLC

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

#include <NeoMathEngine/SimdMathEngine.h>
#include <PrimitivesJit.h>

namespace NeoML {

class CAvxMathEngine : public ISimdMathEngine {
public:
	CAvxMathEngine( IMathEngine* _mathEngine, int _threadCount ) :
		mathEngine( _mathEngine ), threadCount( _threadCount ), primitives( _mathEngine, _threadCount ) {}

	CConvolutionDesc* InitBlobConvolution( const CBlobDesc& source, int paddingHeight, int paddingWidth,
		int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, const CBlobDesc& filter,
		const CBlobDesc& result ) const override;

	void BlobConvolution( const CConvolutionDesc& convDesc, const float* source,
		const float* filter, const float* freeTerm, float* result ) const override;

	virtual CConvolutionDesc* InitBlockedConvolution( const CBlobDesc& source, int paddingHeight, int paddingWidth,
		int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, const CBlobDesc& filter,
		const CBlobDesc& result ) const override;
	void PackBlockedData( const CBlobDesc& desc, const float* source, float* result ) const override;
	void UnpackBlockedData( const CBlobDesc& desc, const float* source, float* result ) const override;
	void PackBlockedFilter( const CBlobDesc& desc, const float* source, float* result ) const override;
	void BlockedConvolution( const CConvolutionDesc& convDesc, const float* packedSource,
		const float* packedFilter, const float* freeTerm, float* packedResult ) const override;

	SgemmFunc GetSgemmFunction() const override;

	void Tanh( float* dst, const float* src, size_t dataSize, bool isMultithread ) override;
	void Sigmoid( float* dst, const float* src, size_t dataSize, bool isMultithread ) override;
	void Exp( float* dst, const float* src, size_t dataSize, bool isMultithread ) override;
	void RunOnceRestOfLstm( CMathEngineLstmDesc* desc, const CConstFloatHandle& inputStateBackLink,
		const CFloatHandle& outputStateBackLink, const CFloatHandle& outputMainBackLink, bool isMultithread ) override;

private:
	IMathEngine* mathEngine;
	int threadCount;
	CPrimitivesJit primitives;
};

}

