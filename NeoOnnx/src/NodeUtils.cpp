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

#include "common.h"
#pragma hdrstop

#include "NodeUtils.h"

namespace NeoOnnx {

void CalculatePadding( const CString& autoPad, const CTensorShape& inputShape,
	const CTensorShape& kernelShape, CFastArray<int, 8>& pads )
{
	const int padDims = static_cast<int>( kernelShape.Size() );
	const int skipDims = static_cast<int>( inputShape.Size() ) - padDims;
	NeoAssert( skipDims >= 0 );

	for( int padDimIndex = 0; padDimIndex < padDims; ++padDimIndex ) {
		const int totalPadSize = inputShape[padDimIndex + skipDims] + kernelShape[padDimIndex] - 1;
		if( totalPadSize % 2 == 1 ) {
			// This case can be supported in NeoML only if mode is SAME_LOWER.
			NeoAssert( autoPad == "SAME_LOWER" );
		}
		pads[padDimIndex] = ( totalPadSize + 1 ) / 2;
		pads[padDims + padDimIndex] = ( totalPadSize + 1 ) / 2;
	}
}

} // namespace NeoOnnx