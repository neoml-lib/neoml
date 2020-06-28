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

// The layer that repeats a sequence
class NEOML_API CRepeatSequenceLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CRepeatSequenceLayer )
public:
	explicit CRepeatSequenceLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// The number of repetitions
	// The resulting sequence will have the length = InputBatchLength * repeatCount
	int GetRepeatCount() const { return repeatCount; }
	void SetRepeatCount(int count);

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	int repeatCount;
};

} // namespace NeoML
