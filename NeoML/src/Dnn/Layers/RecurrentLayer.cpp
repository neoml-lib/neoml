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

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/Dnn.h>
#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoML/Dnn/Layers/RecurrentLayer.h>

namespace NeoML {

CRecurrentLayer::CRecurrentLayer( IMathEngine& mathEngine, const char* name ) :
	CCompositeLayer( mathEngine, name == nullptr ? "CCnnRecurrentLayer" : name ),
	isReverseSequense(false),
	repeatCount(1)
{
}

CRecurrentLayer::~CRecurrentLayer()
{
}

void CRecurrentLayer::AddBackLink(CBackLinkLayer& backLink)
{
	AddLayer(backLink);
	backLinks.Add(&backLink);

	if(GetInternalDnn() != 0) {
		GetInternalDnn()->AddLayer(*backLink.CaptureSink());
	}
}

void CRecurrentLayer::DeleteBackLink(const char* name)
{
	CPtr<CBaseLayer> layer = GetLayer(name);
	CBackLinkLayer* backLink = dynamic_cast<CBackLinkLayer*>(layer.Ptr());
	NeoAssert(backLink != 0);

	if(GetInternalDnn() != 0) {
		GetInternalDnn()->DeleteLayer(*backLink->CaptureSink());
	}
	DeleteLayer(*backLink);

	for(int i = 0; i < backLinks.Size(); ++i) {
		if(backLinks[i] == backLink) {
			backLinks.DeleteAt(i);
			break;
		}
	}
}

void CRecurrentLayer::DeleteBackLink(CBackLinkLayer& backLink)
{
	DeleteBackLink(backLink.GetName());
}

void CRecurrentLayer::GetBackLinkList(CArray<const char*>& backLinkList) const
{
	backLinkList.SetSize(backLinks.Size());
	for(int i = 0; i < backLinkList.Size(); ++i) {
		backLinkList[i] = backLinks[i]->GetName();
	}
}

void CRecurrentLayer::DeleteAllBackLinks()
{
	CArray<const char*> backLinkList;
	GetBackLinkList(backLinkList);

	for(int i = 0; i < backLinkList.Size(); ++i) {
		DeleteBackLink(backLinkList[i]);
	}
}

void CRecurrentLayer::DeleteAllLayersAndBackLinks()
{
	DeleteAllLayers();
	DeleteAllBackLinks();
}

void CRecurrentLayer::OnDnnChanged( CDnn* old )
{
	CCompositeLayer::OnDnnChanged( old );
	if(GetInternalDnn() != 0) {
		for(int i = 0; i < backLinks.Size(); ++i) {
	// The back links have already been added
	//		GetInternalDnn()->AddLayer(*backLinks[i]);
			GetInternalDnn()->AddLayer(*backLinks[i]->CaptureSink());
		}
	}
}

void CRecurrentLayer::GetState(CObjectArray<CDnnBlob>& state) const
{
	state.SetSize(backLinks.Size());
	for(int i = 0; i < backLinks.Size(); ++i) {
		state[i] = backLinks[i]->GetState();
	}
}

void CRecurrentLayer::SetState(const CObjectArray<CDnnBlob>& state)
{
	NeoAssert(state.Size() == backLinks.Size());
	for(int i = 0; i < backLinks.Size(); ++i) {
		backLinks[i]->SetState(state[i]);
	}
}


void CRecurrentLayer::SetReverseSequence(bool _isReverseSequense)
{
	if(isReverseSequense != _isReverseSequense) {
		ForceReshape();
	}
	isReverseSequense = _isReverseSequense;
}

void CRecurrentLayer::SetRepeatCount(int count)
{
	NeoAssert(count > 0);
	if(count != repeatCount) {
		ForceReshape();
	}
	repeatCount = count;
}

void CRecurrentLayer::getSequenceParams(int& batchWidth, int& sequenceLength)
{
	// The outermost recurrent layer runs in recurrent mode, 
	// the inner layers run step-by-step managed by the outer one
	bool recurrentMode = !GetDnn()->IsRecurrentMode();
	batchWidth = inputDescs[0].BatchWidth();
	sequenceLength = recurrentMode ? inputDescs[0].BatchLength() : GetDnn()->GetMaxSequenceLength();
	sequenceLength *= repeatCount;
}

void CRecurrentLayer::SetInternalDnnParams()
{
	// A recurrent layer should have at least one input
	CheckInputs();
	// Call the parent layer's method
	CCompositeLayer::SetInternalDnnParams();
	int batchWidth;
	int sequenceLength;
	getSequenceParams(batchWidth, sequenceLength);
	if(!GetDnn()->IsRecurrentMode()) {
		GetInternalDnn()->setProcessingParams(true, sequenceLength, isReverseSequense, GetDnn()->IsBackwardPerformed());
	} else {
		CheckArchitecture( repeatCount == 1,
			GetName(), "repeat count should be 1 for internal recurrent layers" );
	}
	// Set the parameters for the back link
	for(int i = 0; i < backLinks.Size(); ++i) {
		backLinks[i]->SetBackwardForced(IsBackwardNeeded() || IsLearningNeeded());
		backLinks[i]->SetDimSize(BD_BatchWidth, batchWidth);
		backLinks[i]->SetDimSize(BD_BatchLength, sequenceLength);
	}
}

// Runs the forward pass of the internal network (overloaded in children)
void CRecurrentLayer::RunInternalDnn()
{
	CheckArchitecture( outputBlobs[0]->GetOwner()->GetBatchLength() == inputBlobs[0]->GetOwner()->GetBatchLength() * repeatCount, 
		GetName(), "incorrect batch length of outputBlobs[0]" );
	CDnn* internalDnn = GetInternalDnn();
	internalDnn->isReuseMemoryMode = GetDnn()->isReuseMemoryMode;
	if(!GetDnn()->IsRecurrentMode()) {
		// Run the internal network in recurrent mode
		if(internalDnn->IsReverseSequense()) {
			for(int sPos = internalDnn->GetMaxSequenceLength() - 1; sPos >= 0; sPos--) {
				internalDnn->runOnce(sPos);
			}
		} else {
			for(int sPos = 0; sPos < internalDnn->GetMaxSequenceLength(); sPos++) {
				internalDnn->runOnce(sPos);
			}
		}
	} else {
		// Step-by-step mode
		CCompositeLayer::RunInternalDnn();
	}
}

// Runs the backward pass of the internal network (overloaded in children)
void CRecurrentLayer::RunInternalDnnBackward()
{
	CDnn* internalDnn = GetInternalDnn();
	// Start the backward pass of the internal network in recurrent mode
	if( !GetDnn()->IsRecurrentMode() ) {
		if( internalDnn->IsReverseSequense() ) {
			for(int sPos = 0; sPos < internalDnn->GetMaxSequenceLength(); sPos++) {
				internalDnn->backwardRunAndLearnOnce(sPos);
			}
		} else {
			for(int sPos = internalDnn->GetMaxSequenceLength() - 1; sPos >= 0; sPos--) {
				internalDnn->backwardRunAndLearnOnce(sPos);
			}
		}
	} else {
		// Step-by-step mode
		CCompositeLayer::RunInternalDnnBackward();
	}
}

void CRecurrentLayer::serializationHook(CArchive& archive)
{
	if( archive.IsStoring() ) {
		archive << backLinks.Size();
		for( int i = 0; i < backLinks.Size(); i++ ) {
			CPtr<CBaseLayer> layer = backLinks[i].Ptr();
			SerializeLayer( archive, MathEngine(), layer );
		}
		archive << isReverseSequense;
		archive << repeatCount;
	} else if( archive.IsLoading() ) {
		backLinks.DeleteAll();
		CObjectArray<CBackLinkLayer> tmpBackLinks;
		int size = 0;
		archive >> size;
		tmpBackLinks.SetSize( size );
		for( int i = 0; i < tmpBackLinks.Size(); i++ ) {
			CPtr<CBaseLayer> layer;
			SerializeLayer( archive, MathEngine(), layer );
			tmpBackLinks[i] = dynamic_cast<CBackLinkLayer*>( layer.Ptr() );
		}
		for(int i = 0; i < tmpBackLinks.Size(); ++i) {
			// The layers have already been loaded, we only need to specify which of them are used as back links
			backLinks.Add(CheckCast<CBackLinkLayer>(GetLayer(tmpBackLinks[i]->GetName())));
		}
		archive >> isReverseSequense;
		archive >> repeatCount;
	} else {
		NeoAssert( false );
	}
}

static const int RecurrentLayerVersion = 2000;

void CRecurrentLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( RecurrentLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CCompositeLayer::Serialize( archive );
}

} // namespace NeoML
