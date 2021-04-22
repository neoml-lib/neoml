/* Copyright Â© 2017-2020 ABBYY Production LLC

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

#include "../Node.h"

namespace NeoOnnx {

// Base class for non-global Pool operator nodes
class CPoolNodeBase : public CLayerOpNode {
public:
	// Operation type
	enum TPoolType {
		PT_Max, // Max pooling
		PT_Mean, // Average pooling

		PT_Count
	};

	CPoolNodeBase( TPoolType poolType, const onnx::NodeProto& pool, int opsetVersion );

	// CNode methods
	void AddLayers( const CObjectArray<const CTensorBase>& inputs,
		CDnn& dnn, CObjectArray<const CTensorBase>& outputs ) override;

	// COpNode methods
	void UserInputMask( CUserInputMask& mask ) const override { mask.Add( true ); }

private:
	TPoolType poolType; // pooling type
	const CString autoPad; // padding mode
	CFastArray<int, 8> kernelShape; // shape of pool kernel
	CFastArray<int, 8> strides; // kernel strides
	CFastArray<int, 8> pads; // convolution paddings
};

// MaxPool operator node
class CMaxPoolNode : public CPoolNodeBase {
public:
	CMaxPoolNode( const onnx::NodeProto& maxPool, int opsetVersion ) :
		CPoolNodeBase( PT_Max, maxPool, opsetVersion ) {}
};

// AveragePool operator node
class CAveragePoolNode : public CPoolNodeBase {
public:
	CAveragePoolNode( const onnx::NodeProto& averagePool, int opsetVersion ) :
		CPoolNodeBase( PT_Mean, averagePool, opsetVersion ) {}
};

} // namespace NeoOnnx