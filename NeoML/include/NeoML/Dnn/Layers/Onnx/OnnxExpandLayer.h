/* Copyright © 2017-2023 ABBYY Production LLC

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

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

// Layer which emulates Onnx Expand operator
// It broadcasts blob from the first input to shape from the second input
class NEOML_API COnnxExpandLayer : public CBaseLayer {
	NEOML_DNN_LAYER( COnnxExpandLayer )
public:
	explicit COnnxExpandLayer( IMathEngine& mathEngine ) : CBaseLayer( mathEngine, "OnnxExpandLayer", false) {}

	// Onnx tensor layout
	// Its size determines the rank of the tensor
	// TensorLayout()[i] contains the blob dimension which contains i'th dimension of Onnx tensor
	const CFastArray<TBlobDim, 8>& TensorLayout() const { return tensorLayout; }
	CFastArray<TBlobDim, 8>& TensorLayout() { return tensorLayout; }

	void Serialize( CArchive& archive );

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override { NeoAssert( false ); }

private:
	CFastArray<TBlobDim, 8> tensorLayout;
};

} // namespace NeoML

