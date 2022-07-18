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

// ConvTranspose
class CConvTransposeOperator : public CLayerOperator {
public:
	CConvTransposeOperator( const onnx::NodeProto& convTranspose, int opsetVersion );

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;

private:
	void getStrides( const CTensorArray& inputs, CFastArray<int, 8>& strides ) const;
	void getDilations( const CTensorArray& inputs, CFastArray<int, 8>& dilations ) const;
	void getOutputPadding( const CTensorShape& convShape, CFastArray<int, 8>& outputPadding ) const;
	void getPads( const CTensorShape& convShape, const CFastArray<int, 8>& outputPadding,
		CFastArray<int, 8>& pads ) const;
	void getTotalPadding( const CTensorShape& convShape, CFastArray<int, 8>& totalPadding ) const;
};

} // namespace NeoOnnx
