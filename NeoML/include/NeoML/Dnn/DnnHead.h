/* Copyright © 2024 ABBYY

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

#include <initializer_list>
#include <NeoML/NeoML.h>
#include <NeoML/Dnn/Dnn.h>
#include <NeoML/Dnn/Layers/CompositeLayer.h>

namespace NeoML {

template <typename T>
class CLayerWrapper;
class CDnnHeadAdapterLayer;
class CGraph;

namespace optimization {
int OptimizeDnnHeadAdapters( NeoML::CGraph& );
}

class CDnnHead : public IObject {
public:
	CDnnHead() = default;

	template <typename... Ts>
	CDnnHead(CRandom& random, IMathEngine& mathEngine, CLayerWrapper<Ts>... linearWrappers)
	{
		CDnn* head(new CDnn(random, mathEngine));

		CPtr<CCompositeSourceLayer> source = new CCompositeSourceLayer(head->GetMathEngine());
		source->SetName("source");
		head->AddLayer(*source);
		CBaseLayer* inputLayer = source;

		// chain connect wrapped layers
		using TExpanding = CBaseLayer * [];
		TExpanding{ inputLayer = linearWrappers(inputLayer)... };

		CPtr<CCompositeSinkLayer> headSink = new CCompositeSinkLayer(head->GetMathEngine());
		headSink->SetName("sink");
		head->AddLayer(*headSink);
		headSink->Connect(0, *(inputLayer));
		dnn = head;
	}

private:
	~CDnnHead() override
	{
		if( dnn != nullptr ) {
			delete dnn;
			dnn = nullptr;
		}
	}

	void increment()
	{
		if( ++headCounter == connections.Size() ) {
			headCounter = 0;
			firstAdapterNum = -1;
		}
	}

	CDnn* dnn = nullptr;

	// Stores all adapter using this head
	CObjectArray<CDnnHeadAdapterLayer> connections;
	// Layers for which input/output blobs are stored for Backward/Learn
	CArray<CBaseLayer*> inputLayers;
	CArray<CBaseLayer*> outputLayers;
	// Pointers to source/sink layers of inner network
	CCompositeSourceLayer* sourceLayer = nullptr;
	CCompositeSinkLayer* sinkLayer = nullptr;
	// Which of the blobs will be used during backward
	int blobsForBackward = 0;
	// Which of the blobs will be used during learn
	int blobsForLearn = 0;

	int headCounter = 0;
	int firstAdapterNum = -1;

	friend class CDnnHeadAdapterLayer;
	friend int optimization::OptimizeDnnHeadAdapters( CGraph& );
};

} // namespace NeoML
