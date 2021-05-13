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

#include "Tensor.h"

// Forward declaration(s)
namespace onnx {
class TensorProto;
} // namespace onnx

namespace NeoOnnx {

// Graph initializer
class CGraphInitializer {
public:
	explicit CGraphInitializer( const onnx::TensorProto& initializer );

	CGraphInitializer( const CGraphInitializer& other ) = delete;
	CGraphInitializer& operator= ( const CGraphInitializer& other ) = delete;

	// Graph initializer name
	const CString& Name() const { return name; }

	// Returns initializer value as data tensor
	CPtr<const CDataTensor> GetDataTensor( IMathEngine& mathEngine );

private:
	// graph initializer name
	CString name;
	// graph initializer info from onnx
	const onnx::TensorProto& initializer;
};

} // namespace NeoOnnx
