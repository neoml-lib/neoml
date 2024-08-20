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

class CCudaRowwise2DPooling : public ICudaRowwiseImpl, public CRowwiseOperationDesc {
public:
	CCudaRowwise2DPooling( CCudaMathEngine& mathEngine, bool isMax, int filterHeight, int filterWidth,
				int strideHeight, int strideWidth ) :
		mathEngine( mathEngine ),
		isMax( isMax ),
		filterHeight( filterHeight ),
		filterWidth( filterWidth ),
		strideHeight( strideHeight ),
		strideWidth( strideWidth )
	{
	}

	// ICudaRowwiseImpl
	CBlobDesc Reshape( const CBlobDesc& inputSize ) override;
	int OutputSize() const override { return outputBlobSize; }
	bool IsInPlace() const override { return false; }
	void Process( const CFloatHandle& input, const CFloatHandle& output ) const override;

private:
	CCudaMathEngine& mathEngine;
	const bool isMax;
	const int filterHeight;
	const int filterWidth;
	const int strideHeight;
	const int strideWidth;
	std::unique_ptr<CMaxPoolingDesc> maxPoolDesc;
	std::unique_ptr<CMeanPoolingDesc> meanPoolDesc;
	int outputBlobSize;
};

//---------------------------------------------------------------------------------------------------------------------

inline CBlobDesc CCudaRowwise2DPooling::Reshape( const CBlobDesc& inputSize )
{
	auto poolOutputSize = [] ( int input, int filter, int stride ) -> int
	{
		return 1 + ( input - filter ) / stride;
	};

	CBlobDesc outputSize = inputSize;
	outputSize.SetDimSize( BD_Height, poolOutputSize( outputSize.Height(), filterHeight, strideHeight ) );
	outputSize.SetDimSize( BD_Width, poolOutputSize( outputSize.Width(), filterWidth, strideWidth ) );
	if( isMax ) {
		maxPoolDesc.reset( mathEngine.InitMaxPooling( inputSize, filterHeight, filterWidth,
			strideHeight, strideWidth, outputSize ) );
	} else {
		meanPoolDesc.reset( mathEngine.InitMeanPooling( inputSize, filterHeight, filterWidth,
			strideHeight, strideWidth, outputSize ) );
	}
	outputBlobSize = outputSize.BlobSize();
	return outputSize;
}

inline void CCudaRowwise2DPooling::Process( const CFloatHandle& input, const CFloatHandle& output ) const
{
	if( isMax ) {
		mathEngine.BlobMaxPooling( *maxPoolDesc, input, nullptr, output );
	} else {
		mathEngine.BlobMeanPooling( *meanPoolDesc, input, output );
	}
}

} // namespace NeoML
