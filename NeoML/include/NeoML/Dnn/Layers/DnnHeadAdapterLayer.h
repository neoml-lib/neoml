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
#include <NeoML/Dnn/Layers/SourceLayer.h>
#include <NeoML/Dnn/Layers/SinkLayer.h>

namespace NeoML {

// 
// TODO: ??? add description
//
class NEOML_API CDnnHeadAdapterLayer final : public CBaseLayer {
	NEOML_DNN_LAYER( CDnnHeadAdapterLayer )
public:
	explicit CDnnHeadAdapterLayer( IMathEngine& mathEngine, const char* name = nullptr )
		: CBaseLayer( mathEngine, name == nullptr ? "CDnnHeadAdapterLayer" : name, /*isLearnable*/true )
	{}

	void Serialize( CArchive& archive ) override
	{
		CBaseLayer::Serialize( archive );
		// TODO: ???
	}

	// Internal shared Dnn between DnnHeadAdapters
	const CDnn* GetDnnHead() const { return head.get(); }
	void SetDnnHead( std::shared_ptr<CDnn> _head ) { head = _head; }

protected:
	void Reshape() override
	{
		NeoAssert( inputDescs.Size() == 1 );
		// TODO: ??? see CCompositeLayer
	}
	void RunOnce() override
	{
		NeoAssert( inputBlobs.Size() == 1 );
		CheckCast<CSourceLayer>( head->GetLayer( "source" ) )->SetBlob( inputBlobs[0] );
		head->RunOnce();
		outputBlobs[0] = CheckCast<CSinkLayer>( head->GetLayer( "sink" ) )->GetBlob();
	}
	void BackwardOnce() override
	{
		// TODO: ??? see CCompositeLayer
	}
	void LearnOnce() override
	{
		// TODO: ??? see CCompositeLayer
	}
	void AllocateOutputBlobs() override {} // no allocate outputBlobs
	// no allocate inputDiffBlobs ???

private:
	std::shared_ptr<CDnn> head;
};

inline NEOML_API CLayerWrapper<CDnnHeadAdapterLayer> DnnHeadAdapter( std::shared_ptr<CDnn> head )
{
	return CLayerWrapper<CDnnHeadAdapterLayer>( "DnnHeadAdapter", [=]( CDnnHeadAdapterLayer* result ) {
		result->SetDnnHead( head );
	} );
}

} // namespace NeoML
