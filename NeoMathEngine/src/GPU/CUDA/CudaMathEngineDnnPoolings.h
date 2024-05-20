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

#pragma once

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_CUDA

#include <NeoMathEngine/CrtAllocatedObject.h>
#include <CudaBlobDesc.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

// CUDA cannot copy structures correctly in cases of multiple inheritance
// So all operation descriptors are in the internal part

struct CCudaMaxPoolingDescInternal final {
	CCudaBlobDesc Source{};
	CCudaBlobDesc Result{};
	int FilterWidth = 0;
	int FilterHeight = 0;
	int StrideHeight = 0;
	int StrideWidth = 0;
};

struct CCudaMaxPoolingDesc : public CMaxPoolingDesc {
	CCudaMaxPoolingDescInternal Internal{};
};

struct CCudaMeanPoolingDescInternal {
	CCudaBlobDesc Source{};
	CCudaBlobDesc Result{};
	int FilterHeight = 0;
	int FilterWidth = 0;
	int StrideHeight = 0;
	int StrideWidth = 0;
};

struct CCudaMeanPoolingDesc : public CMeanPoolingDesc {
	CCudaMeanPoolingDescInternal Internal{};
};

struct CCudaGlobalMaxOverTimePoolingDescInternal {
	CCudaBlobDesc Source{};
	CCudaBlobDesc Result{};
};

struct CCudaGlobalMaxOverTimePoolingDesc : public CGlobalMaxOverTimePoolingDesc {
	CCudaGlobalMaxOverTimePoolingDescInternal Internal;
};

struct CCudaGlobalMaxPoolingDescInternal {
	CCudaBlobDesc Source{};
	CCudaBlobDesc MaxIndices{};
	CCudaBlobDesc Result{};
};

struct CCudaGlobalMaxPoolingDesc : public CGlobalMaxPoolingDesc {
	CCudaGlobalMaxPoolingDescInternal Internal{};
};

struct CCuda3dMaxPoolingDescInternal {
	CCudaBlobDesc Source{};
	CCudaBlobDesc Result{};
	int FilterHeight = 0;
	int FilterWidth = 0;
	int FilterDepth = 0;
	int StrideHeight = 0;
	int StrideWidth = 0;
	int StrideDepth = 0;
};

struct CCuda3dMaxPoolingDesc : public C3dMaxPoolingDesc {
	CCuda3dMaxPoolingDescInternal Internal{};
};

struct CCuda3dMeanPoolingDescInternal {
	CCudaBlobDesc Source{};
	CCudaBlobDesc Result{};
	int FilterHeight = 0;
	int FilterWidth = 0;
	int FilterDepth = 0;
	int StrideHeight = 0;
	int StrideWidth = 0;
	int StrideDepth = 0;
};

struct CCuda3dMeanPoolingDesc : public C3dMeanPoolingDesc {
	CCuda3dMeanPoolingDescInternal Internal{};
};

struct CCudaMaxOverTimePoolingDescInternal {
	CCudaBlobDesc Source{};
	CCudaBlobDesc Result{};
	int FilterLen = 0;
	int StrideLen = 0;
};

struct CCudaMaxOverTimePoolingDesc : public CMaxOverTimePoolingDesc {
	CCudaMaxOverTimePoolingDescInternal Internal{};
};

} // namespace NeoML

#endif // NEOML_USE_CUDA
