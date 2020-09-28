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

#pragma once

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

// CMultichannelLookupLayer stores vector embeddings for the objects and for each input number i returns the ith embedding.
// Each channel has its own embedding table. If there are fewer embedding tables than channels, 
// the extra channels are passed to the output with no change.
class NEOML_API CMultichannelLookupLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CMultichannelLookupLayer )
public:
	explicit CMultichannelLookupLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// The dimensions of the embeddings
	void SetDimensions(const CArray<CLookupDimension>&);
	const CArray<CLookupDimension>& GetDimensions() const { return dimensions; }

	// Gets the blob with the embeddings (written in the rows) 
	const CDnnBlob* GetEmbeddings(int i) const;
	// Sets the i'th embedding table
	// Copies embeddings from data
	void SetEmbeddings( const CPtr<CDnnBlob>& data, int i );
	// If copy is false, the function doesn't create additional copy but data will be changed during training
	void SetEmbeddings( CPtr<CDnnBlob>& data, int i, bool copy );

	// Indicates that external training should be used
	// The default value is false, which means "internal" training is performed: no regularization, etc.
	bool IsUseFrameworkLearning() const { return useFrameworkLearning; }
	void SetUseFrameworkLearning(bool _useFrameworkLearning);

	// Initializes the layer data. Called automatically on Reshape, 
	// however, you may call it in other situations as well (i.e. on Word2VecStep). 
	// Set the input parameter to 0 to clear the embeddings.
	void Initialize(CDnnInitializer* init = 0);

	// Performs a step of "negative sampling" algorithm (see https://arxiv.org/abs/1411.2738)
	// Parameters:
	// batchSize - the number of "words" used on one step (loss averaged over the words)
	// word2vec - the trainable matrix word2vec, of wordCount X vectorLen size
	// context2vec - the trainable matrix context2vec, of wordCount X vectorLen size
	// positiveSamples - positive context samples, of batchSize X positiveCount size
	// positiveCount - the number of positive samples PER WORD
	// negativeSamples - negative context samples, of batchSize X negativeCount size
	// negativeCount - the number of negative samples PER WORD
	// positiveWeights - the positive samples weights, of batchSize X positiveCount size. May be 0
	// negativeWeights - the negative samples weights, of batchSize X negativeWeights size. May be 0
	// words - the words, of batchSize size
	// learningRate - the matrices' learning rate
	// loss - the variable receiving the average error. May be 0
	static void Word2VecStep( IMathEngine& mathEngine, int batchSize,
		CMultichannelLookupLayer& word2vec, CMultichannelLookupLayer& context2vec,
		const CConstIntHandle& positiveSamples, int positiveCount,
		const CConstIntHandle& negativeSamples, int negativeCount,
		const CConstFloatHandle& positiveWeights, const CConstFloatHandle& negativeWeights,
		const CConstIntHandle& words, const CConstFloatHandle& learningRate, const CFloatHandle& loss );

protected:
	// CBaseLayer methods
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	void LearnOnce() override;

private:
	// The size of stored vectors
	CArray<CLookupDimension> dimensions;

	// Indicates that "external" training should be used
	bool useFrameworkLearning;

	CObjectArray<CDnnBlob> ownParams; // "internal" training parameters
	CObjectArray<CDnnBlob>& getParams() { return useFrameworkLearning ? paramBlobs : ownParams; }
	const CObjectArray<CDnnBlob>& getParams() const { return useFrameworkLearning ? paramBlobs : ownParams; }
};

} // namespace NeoML
