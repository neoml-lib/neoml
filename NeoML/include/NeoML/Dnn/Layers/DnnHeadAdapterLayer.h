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

#include <memory>
#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Dnn.h>
#include <NeoML/Dnn/DnnHead.h>

namespace NeoML {

// CDnnHeadAdapterLayer passes data blobs between multiple external layers and a shared internal DNN (head)
// Unlike CompositeLayer, it allows to connect several external layers to same head
class NEOML_API CDnnHeadAdapterLayer final : public CBaseLayer {
	NEOML_DNN_LAYER( CDnnHeadAdapterLayer )
public:
	explicit CDnnHeadAdapterLayer( IMathEngine& mathEngine, const char* name = nullptr )
		: CBaseLayer( mathEngine, name == nullptr ? "CDnnHeadAdapterLayer" : name, /*isLearnable*/true )
	{}

	void Serialize( CArchive& archive ) override;

	// Internal shared Dnn between DnnHeadAdapters
	void SetDnnHead( CPtr<CDnnHead> head );

	// Get Dnn head
	const CDnnHead* GetDnnHead() const { return head; };

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	void LearnOnce() override;
	// It does not allocate outputBlobs in CBaseLayer in runOnce, because they are not used for inference.
	// The outputBlob for CDnnHeadAdapterLayer are sinkLayer->GetBlob() of its internalDnn.
	void AllocateOutputBlobs() override {}
	int BlobsForBackward() const override { return head->blobsForBackward; }
	int BlobsForLearn() const override { return head->blobsForLearn; }

private:
	// Pointer to HeadLayer with inner dnn
	CPtr<CDnnHead> head = nullptr;
	// Save first adapter name to connect to necessary head in serialization
	CString firstAdapter;
	// Stores the number of the layer connected to the internal network
	int num = -1;
	// Temporarily used to store layers during serialization
	CObjectArray<CBaseLayer> layers;
	// Stores the input/output blobs from last Inference
	CObjectArray<CDnnBlob> innerInputBlobs;
	CObjectArray<CDnnBlob> innerOutputBlobs;

	void OnDnnChanged( CDnn* ) override;
	void processBackwardOrLearn();
	void configureAdapter();
	void configureFromHead();
	void saveBlobs();
	void loadBlobs();
	void configureForBackwardAndLearn();
};

inline NEOML_API CLayerWrapper<CDnnHeadAdapterLayer> DnnHeadAdapter( CDnnHead* head )
{
	return CLayerWrapper<CDnnHeadAdapterLayer>( "DnnHeadAdapter", [=]( CDnnHeadAdapterLayer* result ) {
		result->SetDnnHead( head );
	} );
}

} // namespace NeoML
