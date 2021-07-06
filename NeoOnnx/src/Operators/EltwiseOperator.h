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

#include "TensorUtils.h"

namespace NeoOnnx {

// Base class for operators which perform eltwise operations
class CEltwiseOperatorBase : public CLayerOperator {
protected:
	CEltwiseOperatorBase( const onnx::NodeProto& eltwise, int opsetVersion, int argsNum = -1 );

	// AddLayers implementation for the given broadcast and layer
	// The derivatives should call this method from their AddLayers
	void AddLayers( const CBroadcast& broadcast, const CTensorArray& inputs,
		CBaseLayer& eltwiseLayer, CDnn& dnn, CTensorArray& outputs ) const;

private:
	// Expected number of arguments (-1 if any number is supported)
	const int argsNum;
};

//---------------------------------------------------------------------------------------------------------------------

// Eltwise operators with 2 inputs

// Base class
class CEltwiseBinaryOperatorBase : public CEltwiseOperatorBase {
protected:
	CEltwiseBinaryOperatorBase( const onnx::NodeProto& eltwise, int opsetVersion ) :
		CEltwiseOperatorBase( eltwise, opsetVersion, 2 ) {}

	// Returns broadcast
	// The broadcast logic is similar for all of the eltwise binary operators in onnx
	CBroadcast Broadcast() const;
};

// Add operator
class CAddOperator : public CEltwiseBinaryOperatorBase {
public:
	CAddOperator( const onnx::NodeProto& add, int opsetVersion ) : CEltwiseBinaryOperatorBase( add, opsetVersion ) {}

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;
};

// Sub operator
class CSubOperator : public CEltwiseBinaryOperatorBase {
public:
	CSubOperator( const onnx::NodeProto& sub, int opsetVersion ) : CEltwiseBinaryOperatorBase( sub, opsetVersion ) {}

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;
};

// Mul operator
class CMulOperator : public CEltwiseBinaryOperatorBase {
public:
	CMulOperator( const onnx::NodeProto& mul, int opsetVersion ) : CEltwiseBinaryOperatorBase( mul, opsetVersion ) {}

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;
};

// Div operator
class CDivOperator : public CEltwiseBinaryOperatorBase {
public:
	CDivOperator( const onnx::NodeProto& div, int opsetVersion ) : CEltwiseBinaryOperatorBase( div, opsetVersion ) {}

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;
};

//---------------------------------------------------------------------------------------------------------------------

// Eltwise operators with any number of inputs

// Sum operator
class CSumOperator : public CEltwiseOperatorBase {
public:
	CSumOperator( const onnx::NodeProto& sum, int opsetVersion ) : CEltwiseOperatorBase( sum, opsetVersion ) {}

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;
};

} // namespace NeoOnnx

