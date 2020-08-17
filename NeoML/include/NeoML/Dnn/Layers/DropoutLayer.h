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
#include <NeoML/Dnn/Layers/BaseInPlaceLayer.h>

namespace NeoML {

// CDropoutLayer implements a layer that randomly zeroes out some of its input
class NEOML_API CDropoutLayer : public CBaseInPlaceLayer {
	NEOML_DNN_LAYER( CDropoutLayer )
public:
	explicit CDropoutLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// The dropout rate, that is, the probability of setting an input element to zero
	void SetDropoutRate( float value );
	float GetDropoutRate() const { return dropoutRate; }

	// Indicates if spatial dropout mode (setting the whole contents of a channel to zero) should be used
	// By default, spatial dropout is not used
	bool IsSpatial() const { return isSpatial; }
	void SetSpatial( bool value );

	// Indicates if batchwise dropout mode (using the same mask for all batch elements) should be used
	// Not used by default
	bool IsBatchwise() const { return isBatchwise; }
	void SetBatchwise( bool value );

protected:
	virtual ~CDropoutLayer() { destroyDropoutDesc(); }

	// CBaseLayer methods
	void RunOnce() override;
	void BackwardOnce() override;
	void OnReshaped() override;

private:
	CDropoutDesc* desc; // the dropout description
	float dropoutRate; // the dropout rate
	bool isSpatial; // the spatial mode (channel-wise)
	bool isBatchwise; // the batchwise mode

	void initDropoutDesc();
	void destroyDropoutDesc();
};

} // namespace NeoML
