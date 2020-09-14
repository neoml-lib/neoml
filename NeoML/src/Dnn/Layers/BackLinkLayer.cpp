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

#include <NeoML/Dnn/Layers/BackLinkLayer.h>
#include <NeoML/Dnn/Layers/CompositeLayer.h>

namespace NeoML {

static const char* BackLinkSinkNamePostfix = "@Sink";

CBackLinkLayer::CBackLinkLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnBackLink", false ),
	blobDesc( CT_Float )
{
	captureSink = FINE_DEBUG_NEW CCaptureSinkLayer( mathEngine );
	CString captureSinkName = GetName() + CString( BackLinkSinkNamePostfix );
	captureSink->SetName(captureSinkName);
	isProcessingFirstPosition = true;
}

void CBackLinkLayer::SetName(const char* _name)
{
	CBaseLayer::SetName(_name);
	CString captureSinkName = GetName() + CString( BackLinkSinkNamePostfix );
	captureSink->SetName(captureSinkName);
}

void CBackLinkLayer::SetDimSize(TBlobDim d, int size)
{
	if(blobDesc.DimSize(d) == size) {
		return;
	}
	blobDesc.SetDimSize(d, size);
	ForceReshape();
}

void CBackLinkLayer::Connect(int inputNumber, const char* input, int outputNumber)
{
	NeoAssert(inputNumber == 0 || inputNumber == 1);
	if( inputNumber == 0 ) {
		captureSink->Connect( inputNumber, input, outputNumber );
	} else {
		CBaseLayer::Connect( 0, input, outputNumber );
	}
}

void CBackLinkLayer::Reshape()
{
	NeoAssert(GetDnn()->GetMaxSequenceLength() == blobDesc.DimSize(BD_BatchLength));
	outputDescs[0] = blobDesc;
	isProcessingFirstPosition = true;
}

void CBackLinkLayer::RestartSequence()
{
	captureSink->ClearBlob();
	isProcessingFirstPosition = true;
}

void CBackLinkLayer::RunOnce()
{
	// On beginning a new batch, automatically restart sequence (for a reverse sequence, the history will be lost):
	if(GetDnn()->IsReverseSequense() && GetDnn()->IsFirstSequencePos()) {
		RestartSequence();
	}
	// Before learning, initialize backpropagation with diffs (for a NON-reverse sequence, the diff history will be lost):
	if(GetDnn()->IsLastSequencePos() && IsBackwardPerformed()) {
		captureSink->ClearDiffBlob();
	}
	CheckArchitecture(outputBlobs[0]->HasEqualDimensions( captureSink->GetBlob()),
		GetName(), "input and output blobs have different dimensions" );
	if( inputBlobs.IsEmpty() ) {
		outputBlobs[0]->CopyFrom(captureSink->GetBlob());
	} else {
		if( inputBlobs[0]->GetParent() != 0 ) {
			// Teacher forcing mode
			NeoAssert(inputBlobs[0]->GetParentPos() == GetDnn()->GetCurrentSequencePos());
			outputBlobs[0]->CopyFrom(inputBlobs[0]);
		} else if( isProcessingFirstPosition ) {
			// Initializes backpropagation first step or teacher forcing mode with a 1-length sequence
			outputBlobs[0]->CopyFrom(inputBlobs[0]);
		} else {
			// Backpropagation
			outputBlobs[0]->CopyFrom(captureSink->GetBlob());
		}
	}
	isProcessingFirstPosition = false;
}

void CBackLinkLayer::BackwardOnce()
{
	captureSink->CopyDiffBlob(outputDiffBlobs[0]);
	if( !inputDiffBlobs.IsEmpty() && GetDnn()->IsFirstSequencePos() ) {
		inputDiffBlobs[0]->CopyFrom( outputDiffBlobs[0] );
	}
}

const CPtr<CDnnBlob>& CBackLinkLayer::GetState() const 
{ 
	return captureSink->GetBlob();
}

void CBackLinkLayer::SetState(const CPtr<CDnnBlob>& state) 
{ 
	captureSink->SetBlob(state);
}

static const int BackLinkLayerVersion = 2000;

void CBackLinkLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( BackLinkLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	if( archive.IsStoring() ) {
		archive << static_cast<int>( 0 );
		for(TBlobDim d = TBlobDim(0); d < BD_Count; ++d) {
			archive << blobDesc.DimSize(d);
		}
		CPtr<CBaseLayer> baseLayer = captureSink.Ptr();
		SerializeLayer( archive, MathEngine(), baseLayer );
	} else if( archive.IsLoading() ) {
		int pack;
		archive >> pack;
		blobDesc = CBlobDesc(CT_Float);
		for( TBlobDim d = TBlobDim( 0 ); d < BD_Count; ++d ) {
			int size;
			archive >> size;
			blobDesc.SetDimSize( d, size );
		}
		CPtr<CBaseLayer> baseLayer;
		SerializeLayer( archive, MathEngine(), baseLayer );
		captureSink = dynamic_cast<CCaptureSinkLayer*>( baseLayer.Ptr() );
	} else {
		NeoAssert( false );
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void CCaptureSinkLayer::Reshape()
{
	CBlobDesc inputWindow = inputDescs[0];
	inputWindow.SetDimSize( BD_BatchLength, 1 );

	if(blob == 0 || !blob->GetDesc().HasEqualDimensions(inputWindow)) {
		blob = CDnnBlob::CreateBlob( MathEngine(), inputWindow.GetDataType(), inputWindow );
		blob->Clear();
	}
	if(diffBlob == 0 || !diffBlob->GetDesc().HasEqualDimensions(inputWindow)) {
		diffBlob = CDnnBlob::CreateBlob( MathEngine(), inputWindow.GetDataType(), inputWindow );
		diffBlob->Clear();
	}
}

void CCaptureSinkLayer::RunOnce()
{
	blob->CopyFrom(inputBlobs[0]);
}

void CCaptureSinkLayer::ClearBlob()
{
	if(blob != 0) {
		blob->Clear();
	}
}

void CCaptureSinkLayer::CopyDiffBlob( CDnnBlob* _diffBlob ) 
{
	diffBlob->CopyFrom(_diffBlob);
}

void CCaptureSinkLayer::ClearDiffBlob()
{
	if(diffBlob != 0) {
		diffBlob->Clear();
	}
}

void CCaptureSinkLayer::BackwardOnce()
{
	inputDiffBlobs[0] = diffBlob;
}

static const int CaptureSinkLayerVersion = 2000;

void CCaptureSinkLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( CaptureSinkLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );
	
	int temp = 0;
	if( archive.IsStoring() ) {
		archive << temp;
	} else if( archive.IsLoading() ) {
		archive >> temp;
	} else {
		NeoAssert( false );
	}
}

} // namespace NeoML
