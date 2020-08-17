/* Copyright © 2017-2020 ABBYY Production LLC

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

#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

// Positional embeddings - some addition for every element in sequences.
// Takes as input sequences of length BD_ListSize
// (1, BatchWidth, ListSize, H, W, D, C )
class NEOML_API CPositionalEmbeddingLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CPositionalEmbeddingLayer )
public:
	explicit CPositionalEmbeddingLayer( IMathEngine& mathEngine );

	// Embedding type
	enum TPositionalEmbeddingType {
		// Learnable, addition-only Y = X + embedding
		PET_LearnableAddition = 0,
		// Non-learnable (used in transformers). https://arxiv.org/abs/1807.03819
		// Additional restriction: D = H = W = 1.
		PET_Transformers,
		
		PET_EnumCount
	};

	// Embedding type
	// By default is equal to PET_LearnableAddition
	TPositionalEmbeddingType GetType() const { return type; }
	void SetType( TPositionalEmbeddingType newType ) { type = newType; }

	// Additive positional embeddings
	// Trained if GetType() is equal to PET_LearnableAddition
	CPtr<CDnnBlob> GetAddends() const;
	void SetAddends( CDnnBlob* newAddends, bool copy );

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	void LearnOnce() override;
	void Serialize( CArchive& archive ) override;

private:
	// Embedding type
	TPositionalEmbeddingType type;

	// Positional embeddings (if type == PET_Transformers)
	CPtr<CDnnBlob> positionalEmbeddings;

	void checkDimensions();
	void initializeLearnableAddition();
	void fillPositionalEmbedding( CDnnBlob* blob );
};

} // namespace NeoML
