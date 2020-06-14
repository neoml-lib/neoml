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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_CUDA

#include <NeoMathEngine/CrtAllocatedObject.h>
#include <CudaBlobDesc.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

// CUDA does not copy the structures correctly in the case of multiple inheritance
// So all operation descriptors are stored in the internal part

struct CCudaConvolutionDescInternal {
	CCudaBlobDesc Source;
	CCudaBlobDesc Filter;
	CCudaBlobDesc Result;

	int StrideHeight;
	int StrideWidth;

	int PaddingHeight;
	int PaddingWidth;

	int DilationHeight;
	int DilationWidth;
};

struct CCudaConvolutionDesc : public CConvolutionDesc {
	CCudaConvolutionDescInternal Internal;
};

struct CCuda3dConvolutionDescInternal {
	CCudaBlobDesc Source;
	CCudaBlobDesc Filter;
	CCudaBlobDesc Result;

	int StrideHeight;
	int StrideWidth;
	int StrideDepth;

	int PaddingHeight;
	int PaddingWidth;
	int PaddingDepth;
};

struct CCuda3dConvolutionDesc : public C3dConvolutionDesc {
	CCuda3dConvolutionDescInternal Internal;
};

// Channelwise convolution
struct CCudaChannelwiseConvolutionDescInternal {
	int PaddingHeight;
	int PaddingWidth;
	int StrideHeight;
	int StrideWidth;
	CCudaBlobDesc Source;
	CCudaBlobDesc Filter;
	CCudaBlobDesc Result;
};

struct CCudaChannelwiseConvolutionDesc : public CChannelwiseConvolutionDesc {
	CCudaChannelwiseConvolutionDescInternal Internal;
};

struct CCudaTimeConvolutionDescInternal {
	CCudaBlobDesc Source;
	CCudaBlobDesc Filter;
	CCudaBlobDesc Result;
	int Stride;
	int Padding;
	int Dilation;
};

struct CCudaTimeConvolutionDesc : public CTimeConvolutionDesc {
	CCudaTimeConvolutionDescInternal Internal;
};

// RLE convolution descriptor. Should NOT be copied to the GPU
struct CCudaRleConvolutionDesc : public CRleConvolutionDesc {
	float StrokeValue;
	float NonStrokeValue;

	CCudaConvolutionDesc* ConvDesc;
};

} // namespace NeoML

#endif // NEOML_USE_CUDA
