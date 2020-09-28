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

#include <NeoML/Dnn/Layers/CompositeLayer.h>

namespace NeoML {

// The layer implements a recurrent cell
// The number of iterations and the sequence length of the result are equal to BatchLength * repeatCount
class NEOML_API CRecurrentLayer : public CCompositeLayer {
	NEOML_DNN_LAYER( CRecurrentLayer )
public:
	explicit CRecurrentLayer( IMathEngine& mathEngine, const char* name = nullptr );

	~CRecurrentLayer();

	void Serialize( CArchive& archive ) override;

	// Working with backward links
	void AddBackLink(CBackLinkLayer& backLink);
	void DeleteBackLink(const char* name);
	void DeleteBackLink(CBackLinkLayer& backLink);
	void GetBackLinkList(CArray<const char*>& backLinkList) const;
	void DeleteAllBackLinks();
	void DeleteAllLayersAndBackLinks();

	// Retrieves or sets the recurrent layer state
	void GetState(CObjectArray<CDnnBlob>& state) const;
	void SetState(const CObjectArray<CDnnBlob>& state);
	// Indicates that the sequence is processed in reverse order
	bool IsReverseSequense() const { return isReverseSequense; }
	void SetReverseSequence(bool _isReverseSequense);
	// The number of times the same input sequence is processed
	// The full number of iterations and the sequence length of the result = Input[0]->BatchLength() * repeatCount
	int GetRepeatCount() const { return repeatCount; }
	void SetRepeatCount(int count);

protected:
	void OnDnnChanged( CDnn* old ) override;
	void RunInternalDnn() override;
	void RunInternalDnnBackward() override;
	void SetInternalDnnParams() override;

private:
	// The backward links
	CObjectArray<CBackLinkLayer> backLinks;

	// Indicates if reverse order processing is on
	bool isReverseSequense;
	// The number of repetitions of the same input sequence
	int repeatCount;

	void getSequenceParams(int& batchWidth, int& sequenceLength);
	void serializationHook(CArchive& archive) override;
};

} // namespace NeoML
