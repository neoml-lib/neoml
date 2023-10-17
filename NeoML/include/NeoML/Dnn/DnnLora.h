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
#include <NeoML/Dnn/DnnDistributed.h>

namespace NeoML {

// Implementation of Low-Ranked Adaption (LoRA)
// https://arxiv.org/pdf/2106.09685v2.pdf

struct NEOML_API CLoraParams {
	int Rank;
	float Alpha;
	float Dropout;

	explicit CLoraParams( int rank = 1, float alpha = 1.f, float dropout = 0.f )
		: Rank( rank ), Alpha( alpha ), Dropout( dropout ) {}

	void Serialize( CArchive& archive );
};

// This builder is used for adding LoRA to specific layers in graph
// or recursively adds LoRA to all layers in graph and its subgraphs (composite layers)
class NEOML_API CLoraBuilder {
public:
	CLoraBuilder();
	explicit CLoraBuilder( const CArray<CString>& _compositeClases );

	// Builds LoRA weights for a specific fully-connected layer in graph
	void BuildForFc( CDnnLayerGraph& graph, const char* fcName, const CLoraParams& params ) const;

	// Builds LoRA weights for every fully-connected layer in graph and its composite layers
	// In some cases it may lead to troubles because some composite layers contain the logic of their own
	// And these layers may break if some of their internal layers will be replaced with LoRA wrappers
	// Which is why not every derivative of CCompositeLayer is supported by LoRA
	//
	// By default supported derivatives are:
	//    1. CCompositeLayer
	//    2. CTemplateLayer
	//    3. CRecurrentLayer
	//    4. CMultiheadAttentionLayer
	//    5. CTransformerEncoderLayer
	//
	// If this list doesn't fit your task you can replace it with your own via non-default c-tor
	// Always replaces CFullyConnectedLayer which are directly inside of grpah
	// Returns the total number of fully-connected layers replaced by this call
	int BuildForAllFcs( CDnnLayerGraph& graph, const CLoraParams& params ) const;

private:
	CArray<CString> compositeClasses;
};

// A special mechanism which allows to serialize only LoRA weights of CDnn
class NEOML_API CLoraSerializer {
public:
	// Returns the number of LoRA layers whose weights were stored
	int Serialize( CDnn& dnn, CArchive& archive );
	
	// LoRA checkpoint is serialized LoRA weights + solver (same as CDnn)
	int SerializeCheckpoint( CDnn& dnn, CArchive& archive );

	// TODO: distributed ???

private:
	CLoraBuilder loraBuilder; // used for replacing fc's with loraFc if needed
};

//class NEOML_API CLoraMerger {
//public:
//	static void MergeLoraFc( CDnnLayerGraph& graph, const char* fcName );
//	static void DiscardLoraFc( CDnnLayerGraph& graph, const char* fcName );
//	// static void MergeAllFcs( CDnnLayerGraph& graph );
//};

} // namespace NeoML
