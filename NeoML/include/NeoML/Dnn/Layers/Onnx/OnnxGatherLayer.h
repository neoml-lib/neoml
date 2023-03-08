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

// Layer which emulates Onnx Gather operator
// Expect 2 blobs or 2 shape-blobs as input
//    1. Data to select from. Any data type. Data.DimSize(GatherDim) is a number of object to gather from
//       All the dimensions before GatherDim must be 1
//       All the dimensions after GatherDim are interpreted as dimensions of gathered objects
//       Values must be in [-Data.DimSize(GatherDim); Data.DimSize(GatherDim) - 1] interval
//    2. Indices. Integer data type. All the dimensions after GatherDim must be 1.
//       All the dimensions before GatherDim (including GatherDim itself) is the number of indices
// Has 1 output blob or shape-blob of the same type as Data input.
// All dimensions before GatherDim (including GatherDim itself) are equal to corresponding dimensions
// of Indices input.
// All dimensions after GatherDim are equal to corresponding dimensions of Data input.
class NEOML_API COnnxGatherLayer : public COnnxLayerBase {
	NEOML_DNN_LAYER( COnnxGatherLayer )
public:
	explicit COnnxGatherLayer( IMathEngine& mathEngine ) : COnnxLayerBase( mathEngine, "OnnxGatherLayer" ),
		gatherDim( BD_BatchLength ) {}

	void SetGatherDim( TBlobDim dim ) { gatherDim = dim; }
	TBlobDim GetGatherDim() const { return gatherDim; }

	void Serialize( CArchive& archive ) override;

protected:
	void CalculateShapes() override;
	void RunOnce() override;

private:
	TBlobDim gatherDim;
};

} // namespace NeoML
