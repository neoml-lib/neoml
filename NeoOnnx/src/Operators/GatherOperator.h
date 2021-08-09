/* Copyright © 2017-2021 ABBYY Production LLC

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

// Gather operator
class CGatherOperator : public CLayerOperator {
public:
	CGatherOperator( const onnx::NodeProto& gather, int opsetVersion );

protected:
	// COperator methods
	void ProcessTensors( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;

	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;

private:
	// axis value from attributes
	int axisAttr;

	void processDataTensors( const CDataTensor& data, const CDataTensor& indices, CTensorArray& outputs ) const;
	void addImageToPixelLayer( const CUserTensor& data, const CUserTensor& indices, CDnn& dnn, CTensorArray& outputs ) const;
};

} // namespace NeoOnnx
