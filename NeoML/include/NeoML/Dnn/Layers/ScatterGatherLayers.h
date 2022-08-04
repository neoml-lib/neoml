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

// This layer puts the updates into the data based on the indices
//
// Let IndexDims be the number of coordinates used to locate the place where the update should be located
// The layer expects 3 inputs:
//
// 1. Original data of any data type. Contains the set of objects:
//    - First IndexDims dimensions represent the number of objects (ObjectCount)
//    - Rest (BD_Count - IndexDims) dimensions represent the size of the object (ObjectSize)
// 2. Indices. Integer blob which indicate where corresponding updates must be applied to the original data.
//    - BD_Channels must be equal to IndexDims.
//    - The product of the other dimensions must be equal to UpdateCount.
// 3. Updates of the same data type as original data. Contains the set of updates:
//    - Must contain (UpdateCount x ObjectSize) elements (the shape doesn't matter)
//
// This layer performs the following operation:
//
// for( int updateIndex = 0; updateIndex < UpdateCount; ++updateIndex )
//     data[indices[updateIndex]] = updates[updateIndex]
//
// Where indices[...] is an integer vector of length IndexDims which contains coordinates
// in the first IndexDims dimensions of the data blob
//
// The only output contains updated data of the same size and type as original data.
class NEOML_API CScatterNDLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CScatterNDLayer )
public:
	explicit CScatterNDLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	enum TInput {
		I_Data,
		I_Indices,
		I_Updates
	};

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
};

} // namespace NeoML
