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

class CCudaRowwiseChConv : public ICudaRowwiseImpl, public CRowwiseOperationDesc {
public:
	CCudaRowwiseChConv( int paddingHeight, int paddingWidth, int strideHeight,
			int strideWidth, const CBlobDesc& filterDesc, const CConstFloatHandle& filter,
			const CConstFloatHandle* freeTerm ) :
		paddingHeight( paddingHeight ),
		paddingWidth( paddingWidth ),
		strideHeight( strideHeight ),
		strideWidth( strideWidth ),
		filterDesc( filterDesc ),
		filter( filter ),
		freeTerm( freeTerm == nullptr ? CConstFloatHandle() : *freeTerm )
	{
	}

	// ICudaRowwiseImpl
	CBlobDesc Reshape( const CBlobDesc& inputSize ) override;
	int OutputSize() const override { return outputBlobSize; }
	bool IsInPlace() const override { return false; }
	void Process( const CFloatHandle& input, const CFloatHandle& output ) const override;

private:
	const int paddingHeight;
	const int paddingWidth;
	const int strideHeight;
	const int strideWidth;
	const CBlobDesc filterDesc;
	const CConstFloatHandle filter;
	const CConstFloatHandle freeTerm;
	std::unique_ptr<CChannelwiseConvolutionDesc> convDesc;
	int outputBlobSize;
};

//---------------------------------------------------------------------------------------------------------------------

inline CBlobDesc CCudaRowwiseChConv::Reshape( const CBlobDesc& inputSize )
{
	auto convOutputSize = [] ( int input, int filter, int padding, int stride ) -> int
	{
		return 1 + ( input + 2 * padding - filter ) / stride;
	};

	CBlobDesc outputSize = inputSize;
	outputSize.SetDimSize( BD_Height, convOutputSize( inputSize.Height(), filterDesc.Height(),
		paddingHeight, strideHeight ) );
	outputSize.SetDimSize( BD_Width, convOutputSize( inputSize.Width(), filterDesc.Width(),
		paddingWidth, strideWidth ) );

	CBlobDesc freeTermDesc( { inputSize.Channels() } );
	IMathEngine& mathEngine = *filter.GetMathEngine();
	convDesc.reset( mathEngine.InitBlobChannelwiseConvolution( inputSize, paddingHeight, paddingWidth,
		strideHeight, strideWidth, filterDesc, freeTerm.IsNull() ? nullptr : &freeTermDesc, outputSize ) );
	outputBlobSize = outputSize.BlobSize();
	return outputSize;
}

inline void CCudaRowwiseChConv::Process( const CFloatHandle& input, const CFloatHandle& output ) const
{
	filter.GetMathEngine()->BlobChannelwiseConvolution( *convDesc, input, filter,
		freeTerm.IsNull() ? nullptr : &freeTerm, output );
}

} // namespace NeoML
