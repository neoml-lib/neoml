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
#include <NeoML/Dnn/Layers/TransposedConvLayer.h>

namespace NeoML {

// Layer which emulates Onnx ConvTranspose layer
class NEOML_API COnnxConvTransposeLayer : public CTransposedConvLayer {
	NEOML_DNN_LAYER( COnnxConvTransposeLayer )
public:
	explicit COnnxConvTransposeLayer( IMathEngine& mathEngine ) : CTransposedConvLayer( mathEngine ),
		useExternalPadding( false ) {}

	CString& AutoPad() { return autoPad; }
	const CString& AutoPad() const { return autoPad; }

	// Onnx pads attribute
	CFastArray<int, 8>& Pads() { return pads; }
	const CFastArray<int, 8>& Pads() const { return pads; }

	// Onnx output_pads attribute
	CFastArray<int, 8>& OutputPadding() { return outputPadding; }
	const CFastArray<int, 8>& OutputPadding() const { return outputPadding; }

	// Onnx outputShape attribute
	CFastArray<int, 8>& OutputShape() { return outputShape; }
	const CFastArray<int, 8>& OutputShape() const { return outputShape; }

	void Serialize( CArchive& archive );

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override { NeoAssert( false ); }
	void LearnOnce() override { NeoAssert( false ); }

private:
	// Onnx attributes of ConvTranspose node
	CString autoPad;
	CFastArray<int, 8> pads;
	CFastArray<int, 8> outputPadding;
	CFastArray<int, 8> outputShape;

	// the result padding based on attributes and the size of input
	CFastArray<int, 8> totalPadding;
	// true if padding can't be done by NeoML convolution
	// otherwise the padding must be done manually without
	bool useExternalPadding;
	CBlobDesc neomlConvOutputDesc;

	void calcTotalPadding();
	CBlobDesc getPaddedDesc( const CBlobDesc& inputDesc );
};

} // namespace NeoML
