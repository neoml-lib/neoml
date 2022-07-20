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

#include "../LayerOperator.h"

namespace NeoOnnx {

// Base class for non-global Pool operators
class CPoolOperatorBase : public CLayerOperator {
protected:
	CPoolOperatorBase( const onnx::NodeProto& pool, int opsetVersion );

	// Gets padding sizes
	void GetPads( const CTensorArray& inputs, CFastArray<int, 8>& pads ) const;

	// AddLayers implementation for the given padding value and pooling layer
	// The derivatives should call this method from their AddLayers
	void AddLayersImpl( const CTensorArray& inputs, float padValue,
		CPoolingLayer& layer, CDnn& dnn, CTensorArray& outputs ) const;

private:
	// Padding mode
	CString autoPad;
	// Shape of pool kernel
	CFastArray<int, 8> kernelShape;

	void getStrides( const CTensorArray& inputs, CFastArray<int, 8>& strides ) const;
};

// MaxPool operator
class CMaxPoolOperator : public CPoolOperatorBase {
public:
	CMaxPoolOperator( const onnx::NodeProto& maxPool, int opsetVersion ) :
		CPoolOperatorBase( maxPool, opsetVersion ) {}

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;
};

// AveragePool operator
class CAveragePoolOperator : public CPoolOperatorBase {
public:
	CAveragePoolOperator( const onnx::NodeProto& averagePool, int opsetVersion );

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;

private:
	// Indicates whether pad pixels should be included in output calculation
	bool includePad;
};

} // namespace NeoOnnx

