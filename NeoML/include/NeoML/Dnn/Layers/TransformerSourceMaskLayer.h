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

#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

// Tha layer that, based on a batch of widths and the shape of the matrix Q, builds a mask in multihead 
// attention layer. It's needed to ignore padding in batches with sequences of different lengths.

//  Inputs:
//  #0 - batch of widths (1 x BatchWidth x 1 x 1 x 1 x 1 x 1)
//  #1 - matrix Q (1 x BatchWidth x ListSize_Q x 1 x 1 x 1 x Channels_Q)
//
// Result has size (1 x BatchWidth x headCount x 1 x ListSize_Q x 1 x ListSize_Q)
class NEOML_API CTransformerSourceMaskLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CTransformerSourceMaskLayer )
public:
	explicit CTransformerSourceMaskLayer( IMathEngine& mathEngine );

	// The number of heads in multihead attention
	int GetHeadCount() const { return headCount; }
	void SetHeadCount( int _headCount ) { headCount = _headCount; }

	virtual void Serialize( CArchive& archive ) override;

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	// The amount of heads. This parameter is necessary so that the shape of the mask
	// coincides with the shape of the Q * K_t matrix in multihead attention
	int headCount;

	// Layer inputs
	enum TInputs {
		I_Widths = 0,
		I_Q = 1
	};
};

NEOML_API CLayerWrapper<CTransformerSourceMaskLayer> TransformerSourceMask(
	int headCount );

}  // namespace NeoML
