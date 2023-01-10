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

// Base class for reshapers: special layers whose purpose is to compute special shape tensors during Reshape phase
class NEOML_API CBaseReshaper : public CBaseLayer {
public:
	const CObjectArray<CDnnBlob>& GetOutputShapeBlobs() const { return outputShapeBlobs; }

	void Serialize( CArchive& archive ) override;

protected:
	CBaseReshaper( IMathEngine& mathEngine, const char* name ) :
		CBaseLayer( mathEngine, name, false ) {}

	// Shape blobs from input layers
	// nullptr if input layer is not a reshaper
	CObjectArray<CDnnBlob> inputShapeBlobs;
	// Shape blobs of this layer
	CObjectArray<CDnnBlob> outputShapeBlobs;

	// This method must contain the calculation of OutputShape based on inputShapeTensors (called from CBaseLayer::Reshape)
	// When called the following fields are set to following:
	//  inputShapeTensors is filled with the shape tensors from corresponding inputs
	//  OutputShape filled with CBaseLayer::GetOutputCount() unintialized shape tensors
	virtual void CalculateShapes() = 0;

	bool HasShapeInputs() const;

private:
	void Reshape() final;
	void RunOnce() override {}
	void BackwardOnce() final { NeoAssert( false ); }
};

} // namespace NeoML
