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

// Expand operator
class CExpandOperator : public CLayerOperator {
public:
	CExpandOperator( const onnx::NodeProto& expand, int opsetVersion );

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& input, CDnn& dnn, CTensorArray& outputs ) const override;

private:
	void getScales( const CTensorArray& inputs, CFastArray<int, 8>& scales ) const;
};

} // namespace NeoOnnx

