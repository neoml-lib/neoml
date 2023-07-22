/* Copyright © 2023 ABBYY

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

#include "CudaRowwiseInterface.h"
#include "../CudaMathEngine.h"

namespace NeoML {

class CCudaRowwiseConv : public ICudaRowwiseImpl, public CRowwiseOperationDesc {
public:
	CCudaRowwiseConv( int padH, int padW, int strideH, int strideW, int dilH, int dilW, const CBlobDesc& filterDesc,
			const CConstFloatHandle& filter, const CConstFloatHandle* freeTerm ) :
		padH( padH ),
		padW( padW ),
		strideH( strideH ),
		strideW( strideW ),
		dilH( dilH ),
		dilW( dilW ),
		filterDesc( filterDesc ),
		filter( filter ),
		freeTerm( freeTerm == nullptr ? CConstFloatHandle() : *freeTerm ),
		outputBlobSize( 0 )
	{
	}

	// ICudaRowwiseImpl
	CBlobDesc Reshape( const CBlobDesc& inputSize ) override;
	int OutputSize() const override { return outputBlobSize; }
	bool IsInPlace() const override { return false; }
	void Process( const CFloatHandle& input, const CFloatHandle& output ) const override;

private:
	const int padH;
	const int padW;
	const int strideH;
	const int strideW;
	const int dilH;
	const int dilW;
	const CBlobDesc filterDesc;
	const CConstFloatHandle filter;
	const CConstFloatHandle freeTerm;
	std::unique_ptr<CConvolutionDesc> convDesc;
	int outputBlobSize;
};

//---------------------------------------------------------------------------------------------------------------------

inline CBlobDesc CCudaRowwiseConv::Reshape( const CBlobDesc& inputSize )
{
	auto convOutputSize = [] ( int input, int filter, int padding,
		int stride, int dilation ) -> int
	{
		return 1 + ( input - ( filter - 1 ) * dilation + 2 * padding - 1 ) / stride;
	};

	CBlobDesc outputSize = inputSize;
	outputSize.SetDimSize( BD_Height, convOutputSize( inputSize.Height(), filterDesc.Height(), padH, strideH, dilH ) );
	outputSize.SetDimSize( BD_Width, convOutputSize( inputSize.Width(), filterDesc.Width(), padW, strideW, dilW ) );
	outputSize.SetDimSize( BD_Channels, filterDesc.ObjectCount() );

	IMathEngine& mathEngine = *filter.GetMathEngine();
	convDesc.reset( mathEngine.InitBlobConvolution( inputSize, padH, padW, strideH, strideW, dilH, dilW,
		filterDesc, outputSize ) );

	outputBlobSize = outputSize.BlobSize();

	return outputSize;
}

inline void CCudaRowwiseConv::Process( const CFloatHandle& input, const CFloatHandle& output ) const
{
	filter.GetMathEngine()->BlobConvolution( *convDesc, input, filter,
		freeTerm.IsNull() ? nullptr : &freeTerm, output);
}

} // namespace NeoML
