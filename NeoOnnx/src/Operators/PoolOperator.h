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

#include "../LayerOperator.h"

namespace NeoOnnx {

// Base class for non-global Pool operators
class CPoolOperatorBase : public CLayerOperator {
protected:
	// Operation type
	enum TPoolType {
		// Max pooling
		PT_Max,
		// Average pooling
		PT_Mean,

		PT_Count
	};

	CPoolOperatorBase( TPoolType poolType, const onnx::NodeProto& pool, int opsetVersion );

	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const final;

private:
	// Pooling type
	const TPoolType poolType;
	// Padding mode
	CString autoPad;
	// Indicates whether pad pixels should be included in output calculation (average pool only)
	bool includePad;
	// Shape of pool kernel
	CFastArray<int, 8> kernelShape;

	void getStrides( const CTensorArray& inputs, CFastArray<int, 8>& strides ) const;
	void getPads( const CTensorArray& inputs, CFastArray<int, 8>& pads ) const;

	static CPtr<CPoolingLayer> createPoolingLayer( CPoolOperatorBase::TPoolType poolType, IMathEngine& mathEngine );
};

// MaxPool operator
class CMaxPoolOperator : public CPoolOperatorBase {
public:
	CMaxPoolOperator( const onnx::NodeProto& maxPool, int opsetVersion ) :
		CPoolOperatorBase( PT_Max, maxPool, opsetVersion ) {}
};

// AveragePool operator
class CAveragePoolOperator : public CPoolOperatorBase {
public:
	CAveragePoolOperator( const onnx::NodeProto& averagePool, int opsetVersion ) :
		CPoolOperatorBase( PT_Mean, averagePool, opsetVersion ) {}
};

} // namespace NeoOnnx

