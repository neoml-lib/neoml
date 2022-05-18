/* Copyright © 2017-2021 ABBYY Production LLC

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

// This layer performs a special convolution used in BERT architecture
//
// This operation extracts the convolution regions as if 1-dimensional kernel of size (kernel size)
// would move along (sequence length) padded by ((kernel size) - 1 / 2) zeros from both sides.
// Then it applies different kernel values for every position along (sequence length), (batch size) and (num heads).
// The only dimension shared betweed different kernels is (head size).
// The kernel values are provided by an additional input.
//
// First input: convolution data
//     - BD_BatchLentgh is equal to (sequence length)
//     - BD_BatchWidth is equal to (batch size)
//     - BD_Channels is equal to (attention heads) * (head size)
//     - others are equal to 1
//
// Second input: convolution kernels
//     - BD_BatchLength is equal to (sequence length)
//     - BD_BatchWidth is equal to (batch size) * (attention heads)
//     - BD_Height is equal to (kernel size)
//     - others are equal to 1
//
// Output:
//     - BD_BatchLength is equal to (sequence length)
//     - BD_BatchWidth is equal to (batch size) * (attention heads)
//     - BD_Height is equal to (head size)
//     - others are equal to 1
class NEOML_API CBertConvLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CBertConvLayer )
public:
	explicit CBertConvLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	int BlobsForBackward() const override { return TInputBlobs; }
};

} // namespace NeoML
