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
#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

class COnnxResizeLayer;

// Base class for Onnx layers

// One of the biggest incompatibilities between NeoML and Onnx is the fact that in NeoML shape inference
// and data inference are separated. As a result, operators like 'Range' whose output size depends on the
// values from inputs become almost impossible.

// This class uses special shape-blobs to work around this problem. These are CDnnBlob calculated during
// shape inference (CDnnBaseLayer::Reshape). Because of the size of these blobs (just a few elements in
// most of the cases) they are always allocated on single-threaded CPU MathEngine.

// This class finalizes CBaseLayer::Reshape and provides virtual COnnxLayerBase::CalculateShapes

// Input shape-blobs are given in CObjectArray<CDnnBlob> inputShapeBlobs
// Its size is equal to the number of inputs
// If i'th input is not a shape-blob then inputShapeBlobs[i] == nullptr

// If layer returns a shape-blob at outputIndex then it must allocate and calculate it during
// CalculateShapes() and store it at outputShapeBlobs[outputIndex]

// If layer returns an usual blob at outputIndex then it must:
//    1. during CalculateShapes() leave outputShapeBlobs[outputIndex] as-is (it will be nullptr)
//    2. during CalculateShapes() fill outputDescs[outputIndex] with expected shape
//    3. override RunOnce() and fill outputBlobs[outputIndex] with the data during it
//       inputShapeBlobs will be available during RunOnce()

class NEOML_API COnnxLayerBase : public CBaseLayer {
public:
	void Serialize( CArchive& archive ) override;

protected:
	COnnxLayerBase( IMathEngine& mathEngine, const char* name ) :
		CBaseLayer( mathEngine, name, false ) {}

	// Shape blobs from input layers
	// inputShapeBlobs[i] == nullptr means that i'th input doesn't contain shape-blob
	CObjectArray<CDnnBlob> inputShapeBlobs;
	// Shape blobs of this layer
	CObjectArray<CDnnBlob> outputShapeBlobs;

	// This method must contain the calculation of outputShapeBlobs or outputDescs
	// See large comment above for more info
	virtual void CalculateShapes() = 0;

	void Reshape() final;
	
	// RunOnce must be overridden if layer wants to return some outputs as usual blobs (not shape-blobs)
	// See large comment above for more info
	void RunOnce() override {}

	void BackwardOnce() final { NeoAssert( false ); }

private:
	friend class COnnxResizeLayer;
};

} // namespace NeoML
