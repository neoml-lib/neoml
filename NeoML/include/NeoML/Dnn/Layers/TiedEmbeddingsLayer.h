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

////////////////////////////////////////////////////////////////////////////////////////////////////

// Tied embeddings layer.  https://arxiv.org/pdf/1608.05859.pdf
// Uses matrix from CMultichannelLookupLayer.
class NEOML_API CTiedEmbeddingsLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CTiedEmbeddingsLayer )
public:
	explicit CTiedEmbeddingsLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;
	
	// Embeddings layer name from which we take the matrix.
	const char* GetEmbeddingsLayerName() const { return embeddingsLayerName; }
	void SetEmbeddingsLayerName( const char* name ) { embeddingsLayerName = name; }
	// Channel index in embeddings layer.
	int GetChannelIndex() const { return channelIndex; }
	void SetChannelIndex( int val );

protected:
	virtual void Reshape() override;
	virtual void RunOnce() override;
	virtual void BackwardOnce() override;
	virtual void LearnOnce() override;

private:
	// Embedding layer name from which we take the matrix.
	CString embeddingsLayerName;
	// Channel index in embedding layer.
	int channelIndex;

	const CDnnBlob* getEmbeddingsTable() const;
};

// Tied embeddings.
NEOML_API CLayerWrapper<CTiedEmbeddingsLayer> TiedEmbeddings( const char* name, int channel );

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace NeoML
