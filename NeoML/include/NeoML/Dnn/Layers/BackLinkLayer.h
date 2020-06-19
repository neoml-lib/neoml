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

class CCaptureSinkLayer;

// Implements a backward link for recurrent networks
class NEOML_API CBackLinkLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CBackLinkLayer )
public:
	explicit CBackLinkLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	void SetName(const char* _name) override;
	void Connect(int inputNumber, const char* input, int outputNumber) override;
	using CBaseLayer::Connect;

	// Gets and sets blob dimensions for the backward link
	void SetDimSize(TBlobDim d, int size);
	int GetDimSize(TBlobDim d) const { return blobDesc.DimSize(d); }

	// The layer serving to retrieve the data and pass it out of the layer
	CCaptureSinkLayer* CaptureSink() { return captureSink; }
	const CCaptureSinkLayer* CaptureSink() const { return captureSink; }
	// Begin processing a new sequence
	void RestartSequence() override;

	// Saves or loads the link state
	const CPtr<CDnnBlob>& GetState() const;
	void SetState(const CPtr<CDnnBlob>& state);

protected:
	// CBaseLayer methods
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	void OnDnnChanged( CDnn* ) override { RestartSequence(); }

private:
	// The description of the backward link blob
	CBlobDesc blobDesc;

	// The layer serving to retrieve the data and pass it out of the layer
	CPtr<CCaptureSinkLayer> captureSink;

	// Indicates if the first step in the sequence is currently under processing
	bool isProcessingFirstPosition;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////

// CDnnCaptureSink implements a layer to retrieve the data for CDnnBackLink
class NEOML_API CCaptureSinkLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CCaptureSinkLayer )
public:
	explicit CCaptureSinkLayer( IMathEngine& mathEngine ) : CBaseLayer( mathEngine, "CCnnCaptureSink", false ) {}

	void Serialize( CArchive& archive ) override;

	// Gets the reference to the blob
	const CPtr<CDnnBlob>& GetBlob() const { return blob; }
	// Sets the blob that will store the output data
	void SetBlob( CDnnBlob* _blob ) { blob = _blob; }
	void ClearBlob();

	// Sets the blob with corrections
	void CopyDiffBlob( CDnnBlob* diffBlob );
	void ClearDiffBlob();

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	CPtr<CDnnBlob> blob;
	CPtr<CDnnBlob> diffBlob;
};

} // namespace NeoML
