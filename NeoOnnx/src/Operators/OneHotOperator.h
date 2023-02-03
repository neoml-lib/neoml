/* Copyright © 2017-2022 ABBYY Production LLC

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

#include "../LayerOperator.h"

namespace NeoOnnx {

// OneHot operator
class COneHotOperator : public CLayerOperator {
public:
	COneHotOperator( const onnx::NodeProto& oneHot, int opsetVersion );

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& input, CDnn& dnn, CTensorArray& outputs ) const override;

private:
	// Operator inputs
	enum TInput {
		I_Indices = 0, // Indices in which on-values must be set
		I_Depth = 1, // The depth of the output blob
		I_Values = 2 // off-value and on-value
	};

	void checkValuesSupport( const CTensorBase& values ) const;
	CPtr<const CTensorBase> prepareIndices( const CTensorBase& indicesInput ) const;
	int getAxis( int indicesDimCount ) const;
};

} // namespace NeoOnnx
