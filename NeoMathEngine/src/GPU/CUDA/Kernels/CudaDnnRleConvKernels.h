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

#include <CudaMathEngineDnnConvs.h>
#include <Kernels/CudaGrid.h>
#include <CudaCommon.h>

namespace NeoML {

__global__ void BlobConvertFromRleKernel( const CCudaConvolutionDescInternal convDesc, float strokeValue,
	float nonStrokeValue, const void* sourceData, int objectSize, float* resultData )
{
	const CCudaBlobDesc& source = convDesc.Source;

	int num;
	int line;
	if(!GetCudaTaskIndex2D(source.ObjectCount(), source.Height(), num, line)) {
		return;
	}
	const CCudaRleImage* __restrict__ image = reinterpret_cast<const CCudaRleImage*>(
		(const char*)sourceData + num * objectSize);
	float* output = GetBlobPtr(source, resultData, num, line, 0, 0);

	int imageStart = (source.Height() - image->Height) / 2;
	int imageStop = imageStart + image->Height;

	if(line < imageStart || line >= imageStop) {
		// Empty row
		for(int i = 0; i < source.Width(); ++i) {
			*output++ = nonStrokeValue;
		}
		return;
	}

	// Find the needed row in the RLE image
	int lineToPass = line - imageStart;
	const CCudaRleStroke* __restrict__ rleStroke = image->Lines;
	while(lineToPass > 0) {
		if(rleStroke->End < 0) {
			--lineToPass;
		}
		++rleStroke;
	}

	// Fill the row start with empty values
	int startPos = (source.Width() - image->Width) / 2;
	for(int i = 0; i < startPos; ++i) {
		*output++ = nonStrokeValue;
	}

	// Draw the strokes
	int pos = 0;
	while(rleStroke->End >= 0) {
		for(; pos < rleStroke->Start; ++pos) {
			*output++ = nonStrokeValue;
		}
		for(; pos < rleStroke->End; ++pos) {
			*output++ = strokeValue;
		}
		++rleStroke;
	}

	// Fill the rest of the row with empty values
	int rest = source.Width() - pos - startPos;
	for(int i = 0; i < rest; ++i) {
		*output++ = nonStrokeValue;
	}
}

} // namespace NeoML
