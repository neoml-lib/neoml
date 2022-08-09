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
#include "EltwiseOperator.h"

namespace NeoOnnx {

// Not operator
class CNotOperator : public CLayerOperator {
public:
	CNotOperator( const onnx::NodeProto& notNode, int opsetVersion );

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;
};

// --------------------------------------------------------------------------------------------------------------------

// Less operator
// (and also Greater, LessOrEqual and GreaterOrEqual)
class CLessOperator : public CEltwiseBinaryOperatorBase {
public:
	CLessOperator( const onnx::NodeProto& less, int opsetVersion ) : CEltwiseBinaryOperatorBase( less, opsetVersion ) {}

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;
};

// --------------------------------------------------------------------------------------------------------------------

// Equal operator
class CEqualOperator : public CEltwiseBinaryOperatorBase {
public:
	CEqualOperator( const onnx::NodeProto& equal, int opsetVersion ) : CEltwiseBinaryOperatorBase( equal, opsetVersion ) {}

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;
};

// --------------------------------------------------------------------------------------------------------------------

// Where operator
class CWhereOperator : public CEltwiseOperatorBase {
public:
	CWhereOperator( const onnx::NodeProto& whereNode, int opsetVersion );

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;
};

} // namespace NeoOnnx
