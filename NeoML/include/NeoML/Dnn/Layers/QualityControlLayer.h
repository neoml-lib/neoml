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

// A base class for layers used to estimate network quality between runs.
// The layer has two inputs: #0 - the network result, #1 - the correct result
class NEOML_API CQualityControlLayer : public CBaseLayer {
public:
	// Controlling statistics calculation
	virtual void SetReset( const bool value ) { needReset = value; }
	virtual bool IsResetNeeded() const { return needReset; }

	void Serialize( CArchive& archive ) override;

protected:
	CQualityControlLayer( IMathEngine& mathEngine, const char* name );
	~CQualityControlLayer() {}

	void Reshape() override;
	void BackwardOnce() override;

	// User-implemented
	virtual void OnReset() = 0;
	virtual void RunOnceAfterReset() = 0;

private:
	// Indicates if Reset should be called after every iteration
	bool needReset;

	// RunOnce may not be overloaded further. Use the RunOnceAfterReset() method
	void RunOnce() override;
};

} // namespace NeoML
