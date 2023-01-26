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

// Layer which emulates Onnx Split operator
// The dimension along which split is performed must be set via SetDim.
// It may have 1 or 2 inputs
//    Data - blob or shape-blob of any data type and size
//    [optional] Sizes - integer shape-blob with number of elements equal to the number of outputs
//                 If missing, the blob will be split evenly into OutputCount() parts
// Each output will contain blob or shape-blob of the same data type as Data
// If Sizes input is present, splitDim of i'th output is equal to the Sizes[i]
// Otherwise splitDim of i'th output is equal to Data.DimSize(splitDim) / OutputCount
// Other dimensions of output are equal to the corresponding dimensions of Data input
class NEOML_API COnnxSplitLayer : public COnnxLayerBase {
	NEOML_DNN_LAYER( COnnxSplitLayer )
public:
	explicit COnnxSplitLayer( IMathEngine& mathEngine ) : COnnxLayerBase( mathEngine, "OnnxSplitLayer" ),
		splitDim( BD_Count ) {}

	void Serialize( CArchive& archive );

	TBlobDim GetSplitDim() const { return splitDim; }
	void SetSplitDim( TBlobDim newDim ) { splitDim = newDim; }

protected:
	void CalculateShapes() override;
	void RunOnce() override;

private:
	TBlobDim splitDim;
};

} // namespace NeoML
