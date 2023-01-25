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
#include <NeoML/Dnn/Layers/Onnx/OnnxLayerBase.h>

namespace NeoML {

// Layer which emulates Onnx Concat operator
class NEOML_API COnnxConcatLayer : public COnnxLayerBase {
	NEOML_DNN_LAYER( COnnxConcatLayer )
public:
	explicit COnnxConcatLayer( IMathEngine& mathEngine ) : COnnxLayerBase( mathEngine, "OnnxConcatLayer" ),
		concatDim( BD_BatchLength ) {}

	// Dimension along which concat must be performed
	void SetConcatDim( TBlobDim dim ) { concatDim = dim; }
	TBlobDim GetConcatDim() const { return concatDim; }

	void Serialize( CArchive& archive );

protected:
	void CalculateShapes() override;
	void RunOnce() override;

private:
	TBlobDim concatDim;

	bool inputHasElements( int inputIndex ) const;
	void calcOutput( const CObjectArray<CDnnBlob>& inputs, const CPtr<CDnnBlob>& output );
};

} // namespace NeoML
