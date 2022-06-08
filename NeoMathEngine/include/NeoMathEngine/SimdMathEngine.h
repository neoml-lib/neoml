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

namespace NeoML {

typedef void ( *SgemmFunc )( bool transA, bool transB,
	IMathEngine *engine,
	const float* aPtr, size_t aRowSize,
	const float* bPtr, size_t bRowSize,
	float* cPtr, size_t cRowSize,
	size_t m, size_t n, size_t k );

struct CMathEngineLstmDesc;

class ISimdMathEngine : public CCrtAllocatedObject {
public:
	virtual ~ISimdMathEngine() = default;

	// Convolution
	// The descriptor should be destroyed using the standard delete operator after use.
	virtual CConvolutionDesc* InitBlobConvolution( const CBlobDesc& source, int paddingHeight, int paddingWidth,
		int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, const CBlobDesc& filter,
        const CBlobDesc& result ) const = 0;

	virtual void BlobConvolution( const CConvolutionDesc& convDesc, const float* source,
		const float* filter, const float* freeTerm, float* result ) const = 0;

	virtual SgemmFunc GetSgemmFunction() const = 0;

	virtual void Tanh( float* dst, const float* src, size_t dataSize, bool isMultithread = true ) = 0;
	virtual void Sigmoid( float* dst, const float* src, size_t dataSize, bool isMultithread = true ) = 0;
	virtual void Exp( float* dst, const float* src, size_t dataSize, bool isMultithread = true ) = 0;
	virtual void RunOnceRestOfLstm( CMathEngineLstmDesc* desc, const CConstFloatHandle& inputStateBackLink,
		const CFloatHandle& outputStateBackLink, const CFloatHandle& outputMainBackLink, bool isMultithread = true ) = 0;

	using vectorAddFunc = void (*)( const float* first, const float* second, float* result, int vectorSize );
	using alignedVectorAdd = void (*)( const float* first, float* second, int vectorSize );
	using vectorEltwiseMax = void (*)( const float* first, const float* second, float* result, int vectorSize );
	using vectorReLU = void (*)( const float* first, float* result, int vectorSize );
	using vectorReLUTreshold = void (*)( const float* first, float* result, int vectorSize, float threshold );
	using alignedVectorMultiplyAndAdd = void (*)( const float* first, const float* second,
		float* result, int vectorSize, const float* mult );
	using vectorMultiply = void (*)( const float* first, float multiplier, float* result, int vectorSize );
	using vectorEltwiseMultiply = void (*)( const float* first, const float* second, float* result, int vectorSize );
	using vectorEltwiseMultiplyAdd = void (*)( const float* first, const float* second, float* result, int vectorSize );
	using vectorAddValue = void (*)( const float* first, float value, float* result, int vectorSize );
	using vectorDotProduct = void (*)( const float* first, const float* second, float* result, int vectorSize );
	using vectorMinMax = void (*)( const float* first, float* result, int vectorSize, const float minValue, const float maxValue );

	virtual vectorAddFunc GetVectorAddFunc() = 0;
	virtual alignedVectorAdd GetAlignedVectorAddFunc() = 0;
	virtual vectorEltwiseMax GetVectorMaxFunc() = 0;
	virtual vectorReLU GetVectorReLUFunc() = 0;
	virtual vectorReLUTreshold GetVectorReLUTresholdFunc() = 0;
	virtual alignedVectorMultiplyAndAdd GetAlignedVectorMultiplyAndAddFunc() = 0;
	virtual vectorMultiply GetVectorMultiplyFunc() = 0;
	virtual vectorEltwiseMultiply GetVectorEltwiseMultiplyFunc() = 0;
	virtual vectorEltwiseMultiplyAdd GetVectorEltwiseMultiplyAddFunc() = 0;
	virtual vectorAddValue GetVectorAddValueFunc() = 0;
	virtual vectorDotProduct GetVectorDotProductFunc() = 0;
	virtual vectorMinMax GetVectorMinMaxFunc() = 0;
};

}
