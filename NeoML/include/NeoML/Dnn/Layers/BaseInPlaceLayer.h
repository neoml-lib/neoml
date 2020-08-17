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

// CBaseInPlaceLayer is the base class for an in-place processing layer
class NEOML_API CBaseInPlaceLayer : public CBaseLayer {
protected:
	CBaseInPlaceLayer(IMathEngine& mathEngine, const char* name, bool isLearnable = false) : CBaseLayer(mathEngine, name, isLearnable), isInPlace( false ) {};

	// Called once reshape is complete
	virtual void OnReshaped() {}
	void AllocateOutputBlobs() override;

	void Serialize( CArchive& archive ) override;

private:
	// The Reshape method may not be overloaded
	void Reshape() final;

	// Indicates if the layer performs in-place processing (after the Reshape method call)
	bool isInPlace;
};

} // namespace NeoML
