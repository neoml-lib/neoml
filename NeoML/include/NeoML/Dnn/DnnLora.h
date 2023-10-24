/* Copyright © 2023 ABBYY

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

// Forward declaration
class CDistributedTraining;

// Implementation of Low-Ranked Adaption (LoRA)
// https://arxiv.org/pdf/2106.09685v2.pdf

struct NEOML_API CLoraParams {
	int Rank; // Size of vector in-between A and B matrices of LoRA
	float Alpha; // Coefficient, the output will be multiplied by Alpha / Rank
	float Dropout; // Dropout applied to input before matrix multiplications

	explicit CLoraParams( int rank = 1, float alpha = 1.f, float dropout = 0.f )
		: Rank( rank ), Alpha( alpha ), Dropout( dropout ) {}

	void Serialize( CArchive& archive );
};

// Mechanism which allows to add/remove/merge LoRA into nets
// It works with CDnnLayerGraph which allows you to modify CDnn or specific composites (e.g. CTransformerEncoderLayer)
class NEOML_API CLoraBuilder {
public:
	CLoraBuilder();
	// Special constructor which sets list of composites allowed to be modified by this builder
	// See BuildForAll* methods for more information
	explicit CLoraBuilder( const CArray<CString>& _compositeClases );

	// Adds LoRA weights to a specific layer
	// Fully-connected
	void BuildFcWrapper( CDnnLayerGraph& graph, const char* fcName, const CLoraParams& params ) const;

	// Builds LoRA weights for every layer of specific type inside graph and its subgraphs (composite layers)
	// In some cases it may lead to troubles because some composite derivatives contain logic of their own
	// And these layers may break if some of their internal layers will be replaced with LoRA wrappers
	// Which is why not every derivative of CCompositeLayer is supported as a subgraph
	//
	// By default supported derivatives are:
	//    1. CCompositeLayer
	//    2. CTemplateLayer
	//    3. CRecurrentLayer
	//    4. CMultiheadAttentionLayer
	//    5. CTransformerEncoderLayer
	//
	// If this list doesn't fit your task you can replace it with your own via constructor
	// Always replaces CFullyConnectedLayer which are directly inside of graph
	// Returns the total number of fully-connected layers replaced by this call
	// Fully-connected
	int BuildAllFcWrappers( CDnnLayerGraph& graph, const CLoraParams& params ) const;

	// Disables training of all layers in the net except LoRA wrappers
	// Affects only trainable non-composite layers which enabled training
	int DisableNonLoraTraining( CDnnLayerGraph& graph ) const;

	// Replaces specific LoRA wrapper without merging LoRA weights (restores original layer weights)
	// Fully-connected
	void DiscardFcWrapper( CDnnLayerGraph& graph, const char* fcName ) const
		{ replaceFcWrapper( graph, fcName, false ); }

	// Replaces all LoRA wrappers of specific type in graph without merging LoRA weights
	// Fully-connected
	int DiscardAllFcWrappers( CDnnLayerGraph& graph ) const { return replaceAllFcWrappers( graph, false ); }

	// Replaces specific LoRA wrapper with merging LoRA weights
	// Fully-connected
	void MergeFcWrapper( CDnnLayerGraph& graph, const char* fcName ) const { replaceFcWrapper( graph, fcName, true ); }

	// Replaces all LoRA wrappers of specific type in graph with merging LoRa weights
	// Fully-connected
	int MergeAllFcWrappers( CDnnLayerGraph& graph ) const { return replaceAllFcWrappers( graph, true ); }

private:
	CArray<CString> compositeClasses;

	void replaceFcWrapper( CDnnLayerGraph& graph, const char* fcName, bool mergeWeights ) const;
	int replaceAllFcWrappers( CDnnLayerGraph& graph, bool mergeWeights ) const;
};

// A special mechanism which allows to serialize only LoRA weights of CDnn
class NEOML_API CLoraSerializer {
public:
	// Returns the number of LoRA layers whose weights were stored/load
	// Weights can be loaded into net with both wrappers or original layers
	// In second case LoRA wrappers will be built on the fly
	int Serialize( CDnn& dnn, CArchive& archive ) const;
	// The same as above but for distributed training
	int Serialize( CDistributedTraining& distributed, CArchive& archive ) const;

	// LoRA checkpoint is serialized LoRA weights + solver(s)
	int SerializeCheckpoint( CDnn& dnn, CArchive& archive ) const;
	int SerializeCheckpoint( CDistributedTraining& distributed, CArchive& archive ) const;
};

} // namespace NeoML
