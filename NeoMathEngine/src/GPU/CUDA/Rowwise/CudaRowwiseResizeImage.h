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

class CCudaRowwiseImageResize : public ICudaRowwiseImpl, public CRowwiseOperationDesc {
public:
	CCudaRowwiseImageResize( TBlobResizePadding padding, float defaultValue, int deltaLeft, int deltaRight,
		int deltaTop, int deltaBottom ) :
			padding( padding ),
			defaultValue( defaultValue ),
			deltaLeft( deltaLeft ),
			deltaRight( deltaRight ),
			deltaTop( deltaTop ),
			deltaBottom( deltaBottom )
	{
	}

	// ICudaRowwiseImpl
	CBlobDesc Reshape( const CBlobDesc& inputSize ) override;
	int OutputSize() const override { return to.BlobSize(); }
	bool IsInPlace() const override { return false; }
	void Process( const CFloatHandle& input, const CFloatHandle& output ) const override;

private:
	TBlobResizePadding padding;
	float defaultValue;
	int deltaLeft;
	int deltaRight;
	int deltaTop;
	int deltaBottom;
	CBlobDesc from;
	CBlobDesc to;
};

//---------------------------------------------------------------------------------------------------------------------

inline CBlobDesc CCudaRowwiseImageResize::Reshape( const CBlobDesc& inputSize )
{
	from = inputSize;
	to = from;
	to.SetDimSize( BD_Height, from.Height() + deltaTop + deltaBottom );
	to.SetDimSize( BD_Width, from.Width() + deltaLeft + deltaRight );
	return to;
}

inline void CCudaRowwiseImageResize::Process( const CFloatHandle& input, const CFloatHandle& output ) const
{
	input.GetMathEngine()->BlobResizeImage( from, input, deltaLeft, deltaRight, deltaTop, deltaBottom,
		padding, defaultValue, to, output );
}

} // namespace NeoML
