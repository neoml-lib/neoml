/* Copyright © 2017-2024 ABBYY

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-------------------------------------------------------------------------------------------------------------- */

#include <common.h>
#pragma hdrstop

#include <NeoML/NeoML.h>
#include <NeoML/Dnn/Dnn.h>
#include <NeoML/Dnn/Layers/DnnHeadAdapterLayer.h>
#include <NeoML/NeoMLDefs.h>

namespace NeoML {

static const int DnnHeadAdapterLayerVersion = 0;

void CDnnHeadAdapterLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( DnnHeadAdapterLayerVersion );
	CBaseLayer::Serialize(archive);

	bool existHead = ( head != nullptr );
	archive.Serialize( existHead );
	if( !existHead ) {
		return;
	}

	if(archive.IsStoring()) {
		archive << head->headCounter;
		if(head->headCounter > 0) {
			CString name(head->connections[0]->GetName());
			archive << name;
		} else {
			NeoAssert( head->dnn != nullptr );
			archive << head->dnn->layers.Size();
			for (int i = 0; i < head->dnn->layers.Size(); i++) {
				SerializeLayer(archive, MathEngine(), head->dnn->layers[i]);
			}
		}

		head->increment();
	}
	else if(archive.IsLoading()) {
		int num;
		archive >> num;
		if(num > 0) {
			archive >> firstAdapter;
		} else {
			int layerSize;
			archive >> layerSize;
			layers.SetSize(layerSize);
			for (int i = 0; i < layerSize; i++) {
				SerializeLayer(archive, MathEngine(), layers[i]);
			}
		}
	}
	else {
		NeoAssert(false);
	}
}

void CDnnHeadAdapterLayer::OnDnnChanged(CDnn*)
{
	// If first adapter - create head dnn and initialize layers
	// Else sets the internal DNN head using the first connected adapter layer after serialization
	if(head == nullptr) {
		if(!firstAdapter.IsEmpty()) {
			SetDnnHead(static_cast<CDnnHeadAdapterLayer*>((GetDnn()->GetLayer(firstAdapter).Ptr()))->head);
		} else if (!layers.IsEmpty()) {
			if (GetDnn() != 0) {
				CDnn* internalDnn = FINE_DEBUG_NEW CDnn(GetDnn()->Random(), GetDnn()->GetMathEngine());

				for (int i = 0; i < layers.Size(); ++i) {
					internalDnn->AddLayer(*layers[i]);
				}
				head = new CDnnHead();
				head->dnn = internalDnn;
				SetDnnHead(head);
				layers.DeleteAll();
			}
		}
	}
}

void CDnnHeadAdapterLayer::Reshape()
{
	if(head->headCounter > 0) {
		configureFromHead();
		return;
	}

	configureAdapter();
}

void CDnnHeadAdapterLayer::RunOnce()
{
	NeoAssert(inputBlobs.Size() == 1);
	NeoAssert(head->dnn != 0);

	head->sourceLayer->SetBlob(inputBlobs[0]);
	head->dnn->isReuseMemoryMode = GetDnn()->isReuseMemoryMode;
	head->dnn->runOnce(GetDnn()->GetCurrentSequencePos());
	outputBlobs[0] = head->sinkLayer->GetInputBlob()->GetCopy();

	// save blobs required for next backward/learn
	if(IsBackwardNeeded() || IsLearningEnabled()) {
		saveBlobs();
	}
}

void CDnnHeadAdapterLayer::processBackwardOrLearn()
{
	NeoAssert(head->dnn->isBackwardPerformed == GetDnn()->isBackwardPerformed);

	if(IsBackwardNeeded()) {
		head->sourceLayer->SetDiffBlob(inputDiffBlobs[0]);
	}

	head->sinkLayer->SetDiffBlob(outputDiffBlobs[0]);

	// loading blobs for backward/learn from last RunOnce
	loadBlobs();

	head->dnn->backwardRunAndLearnOnce(GetDnn()->GetCurrentSequencePos());

	if( head->headCounter == head->connections.Size() - 1 ) {
		for( const CBaseLayer* layer : head->dnn->layers ) {
			if( layer->IsLearningPerformed() ) {
				int& layerCount = GetDnn()->GetSolver()->layerToParamDiffBlobsSum.GetOrCreateValue( layer->GetPath() ).Count;
				layerCount = layerCount - head->connections.Size() + 1;
			}
		}
	}

	head->increment();
	innerInputBlobs.DeleteAll();
	innerInputBlobs.DeleteAll();
}

void CDnnHeadAdapterLayer::BackwardOnce()
{
	processBackwardOrLearn();
}

void CDnnHeadAdapterLayer::LearnOnce()
{
	if(!IsBackwardPerformed()) {
		processBackwardOrLearn();
	}
}

void CDnnHeadAdapterLayer::SetDnnHead(const CPtr<CDnnHead>& _head)
{
	head = _head;
	num = head->connections.Size();
	head->connections.Add(this);
	ForceReshape();
}

void CDnnHeadAdapterLayer::configureAdapter()
{
	NeoAssert(head->dnn != 0);
	head->sinkLayer = static_cast<CCompositeSinkLayer*>(head->dnn->GetLayer("sink").Ptr());
	head->sourceLayer = static_cast<CCompositeSourceLayer*>(head->dnn->GetLayer("source").Ptr());
	if(head->sourceLayer->GetBackwardForced() != IsBackwardNeeded()) {
		head->sourceLayer->SetBackwardForced(IsBackwardNeeded());
	}
	head->sourceLayer->SetBlobDesc(inputDescs[0]);
	// If the backward pass requirements have changed, call reshape
	bool forcedReshape = head->dnn->IsBackwardPerformed() != GetDnn()->IsBackwardPerformed();

	// Set the internal network parameters from the external network parameters
	head->dnn->setProcessingParams(GetDnn()->IsRecurrentMode(), GetDnn()->GetMaxSequenceLength(),
		GetDnn()->IsReverseSequense(), GetDnn()->IsBackwardPerformed());
	head->dnn->RequestReshape(forcedReshape);
	head->dnn->SetInitializer(GetDnn()->GetInitializer());

	head->dnn->SetSolver(GetDnn()->GetSolver());
	head->dnn->reshape();
	configureForBackwardAndLearn();

	outputDescs[0] = head->sinkLayer->inputDescs[0];
	head->firstAdapterNum = num;
	head->increment();
}

void CDnnHeadAdapterLayer::configureFromHead()
{
	if(head->connections[head->firstAdapterNum]->IsLearningEnabled()) {
		EnableLearning();
	}
	else {
		DisableLearning();
	}

	outputDescs[0] = head->sinkLayer->inputDescs[0];
	head->increment();
}

void CDnnHeadAdapterLayer::saveBlobs()
{
	for(int i = 0; i < head->inputLayers.Size(); ++i) {
		innerInputBlobs.Add(head->inputLayers[i]->inputBlobs[0]->GetCopy());
	}

	for(int i = 0; i < head->outputLayers.Size(); ++i) {
		innerOutputBlobs.Add(head->outputLayers[i]->outputBlobs[0]->GetCopy());
	}
}

void CDnnHeadAdapterLayer::loadBlobs()
{
	for(int i = 0; i < head->inputLayers.Size(); ++i) {
		head->inputLayers[i]->inputBlobs[0] = innerInputBlobs[i];
	}

	for(int i = 0; i < head->outputLayers.Size(); ++i) {
		head->outputLayers[i]->outputBlobs[0] = innerOutputBlobs[i];
	}
}

void CDnnHeadAdapterLayer::configureForBackwardAndLearn()
{
	head->blobsForBackward = 0;
	head->blobsForLearn = 0;
	const bool hasBackward = IsBackwardPerformed();
	const bool hasLearn = IsLearningPerformed();

	bool needLearn = false;
	for(int i = 0; i < head->dnn->layers.Size(); ++i) {
		needLearn |= head->dnn->layers[i]->IsLearningPerformed();
		auto layer = dynamic_cast<CDropoutLayer*>(head->dnn->layers[i].Ptr());
		if(layer != nullptr) {
			layer->SetHeadCounter(head->connections.Size());
		}
	}

	if(needLearn) {
		EnableLearning();
	} else {
		DisableLearning();
	}

	if(IsLearningEnabled()) {
		head->dnn->EnableLearning();
	} else {
		head->dnn->DisableLearning();
	}

	if(!hasBackward && !hasLearn) {
		return;
	}

	for(int layerIndex = 0; layerIndex < head->dnn->layers.Size(); ++layerIndex) {
		const CBaseLayer& layer = *head->dnn->layers[layerIndex];
		if(layer.IsBackwardPerformed() && (layer.BlobsForBackward() & TInputBlobs)) {
			head->inputLayers.Add(head->dnn->layers[layerIndex]);
		} else if(layer.IsLearningPerformed() && (layer.BlobsForLearn() & TInputBlobs)) {
			head->inputLayers.Add(head->dnn->layers[layerIndex]);
		}

		if(layer.IsBackwardPerformed() && (layer.BlobsForBackward() & TOutputBlobs)) {
			head->outputLayers.Add(head->dnn->layers[layerIndex]);
		} else if(layer.IsLearningPerformed() && (layer.BlobsForLearn() & TOutputBlobs)) {
			head->outputLayers.Add(head->dnn->layers[layerIndex]);
		}

		if((!hasBackward || head->blobsForBackward != 0) && (!hasLearn || head->blobsForLearn != 0)) {
			break;
		}

		for(int inputIndex = 0; inputIndex < layer.GetInputCount(); ++inputIndex) {
			if(dynamic_cast<const CCompositeSourceLayer*>(layer.GetInputLayer(inputIndex)) != nullptr) {
				if(hasBackward && layer.IsBackwardPerformed() && (layer.BlobsForBackward() & TInputBlobs) != 0) {
					head->blobsForBackward |= TInputBlobs;
				}
				if(hasLearn && layer.IsLearningPerformed() && (layer.BlobsForLearn() & TInputBlobs) != 0) {
					head->blobsForLearn |= TInputBlobs;
				}
				break;
			}
		}
	}
}

}
