/* Copyright © 2017-2022 ABBYY Production LLC

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
#include <NeoML/Dnn/Layers/Onnx/OnnxLayerBase.h>

namespace NeoML {

// Layer which emulates basic Onnx arithmetic operators operator over shape tensors
// Expects all inputs to be blobs or shape-blobs simultaneously
// Returns shape-blob or blob based on inputs
class NEOML_API COnnxEltwiseLayer : public COnnxLayerBase {
	NEOML_DNN_LAYER( COnnxEltwiseLayer )
public:
	enum class TOperation : char {
		// Arithmetics, any number of inputs, return data type is the same as input datatype
		Add, // +
		Sub, // -
		Mul, // *
		Div, // /

		// Comparison, 2 inputs, return data type is always integere
		Less,
		Greater,
		Equal,
		LessOrEqual,
		GreaterOrEqual,

		// Masking, 3 inputs, first input is integer, second and third are of the same data type
		// return type is equal to type of second and third inputs
		Where,

		Count
	};

	explicit COnnxEltwiseLayer( IMathEngine& mathEngine ) :
		COnnxLayerBase( mathEngine, "OnnxEltwiseLayer" ), operation( TOperation::Count ) {}

	TOperation GetOperation() const { return operation; }
	void SetOperation( TOperation newOperation ) { operation = newOperation; }

	void Serialize( CArchive& archive ) override;

protected:
	void CalculateShapes() override;
	void RunOnce() override;

private:
	TOperation operation;
};

} // namespace NeoML
