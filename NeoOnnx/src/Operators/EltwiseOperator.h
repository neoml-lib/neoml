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

#include "../Operator.h"

namespace NeoOnnx {

// Base class for operators which perform eltwise operations
class CEltwiseOperatorBase : public CLayerOperator {
public:
	// Supported eltwise operations
	enum TOperation {
		O_Add, // Addition
		O_Sub, // Subtraction
		O_Mul, // Multiplication
		O_Div, // Division

		O_Count
	};

	// Tensor broadcast types
	enum TBroadcastType {
		BT_None, // Broadcast not supported
		BT_Onnx, // Onnx custom broadcast, used in some versions
		BT_Numpy, // Numpy-style broadcast, used in later versions of ONNX

		BT_Count
	};

	struct CBroadcastInfo {
		TBroadcastType Type;
		int Axis;

		explicit CBroadcastInfo( TBroadcastType type, int axis = NotFound ) :
			Type( type ), Axis( axis ) {}
	};

	// CLayerOperator methods
	void AddLayers( const CObjectArray<const CTensorBase>& inputs,
		CDnn& dnn, CObjectArray<const CTensorBase>& outputs ) override;

	// COperator methods
	// We can guarantee the support only when second input is CDataTensor (other division is impossible)
	void UserInputMask( CUserInputMask& mask ) const override { mask |= 0; }

	// In some versions different operators supported different broadcast types
	// E.g. 'Add' operators in opset v1 supports onnx-broadcast but 'Sum' operators doesn't support broadcast at all
	// That's why each derivative should determine by itself which broadcast type is supported
	virtual CBroadcastInfo BroadcastInfo() const = 0;

protected:
	CEltwiseOperatorBase( const onnx::NodeProto& eltwise, int opsetVersion, TOperation operation, int argsNum = NotFound );

private:
	TOperation operation; // Operation performed by this operator
	int argsNum; // Expected number of arguments (-1 if any number is supported)

	bool broadcastShape( const CTensorShape& first, const CTensorShape& second,
		const CBroadcastInfo& broadcast, CTensorShape& result ) const;
	CPtr<const CTensorBase> broadcast( const CTensorBase& input, const CBroadcastInfo& broadcast, const CTensorShape& outputShape ) const;
	CPtr<const CDataTensor> broadcast( const CDataTensor& input, const CBroadcastInfo& broadcast, const CTensorShape& outputShape ) const;
	CPtr<const CUserTensor> broadcast( const CUserTensor& input, const CBroadcastInfo& broadcast, const CTensorShape& outputShape ) const;
	CPtr<const CTensorBase> prepareSecondInput( const CObjectArray<const CTensorBase>& inputs ) const;
	CPtr<const CUserTensor> padTensorShape( const CUserTensor& input, int dimCount, int axis ) const;
};

// Eltwise operators with 2 inputs

// Base class
class CEltwiseBinaryOperatorBase : public CEltwiseOperatorBase {
public:
	CEltwiseBinaryOperatorBase( const onnx::NodeProto& eltwise, int opsetVersion, TOperation operation ) :
		CEltwiseOperatorBase( eltwise, opsetVersion, operation, 2 ) {}

	// CEltwiseOperatorBase methods
	CBroadcastInfo BroadcastInfo() const override;
};

// Add operator
class CAddOperator : public CEltwiseBinaryOperatorBase {
public:
	CAddOperator( const onnx::NodeProto& add, int opsetVersion ) :
		CEltwiseBinaryOperatorBase( add, opsetVersion, O_Add ) {}
};

// Sub operator
class CSubOperator : public CEltwiseBinaryOperatorBase {
public:
	CSubOperator( const onnx::NodeProto& sub, int opsetVersion ) :
		CEltwiseBinaryOperatorBase( sub, opsetVersion, O_Sub ) {}
};

// Mul operator
class CMulOperator : public CEltwiseBinaryOperatorBase {
public:
	CMulOperator( const onnx::NodeProto& mul, int opsetVersion ) :
		CEltwiseBinaryOperatorBase( mul, opsetVersion, O_Mul ) {}
};

// Div operator
class CDivOperator : public CEltwiseBinaryOperatorBase {
public:
	CDivOperator( const onnx::NodeProto& div, int opsetVersion ) :
		CEltwiseBinaryOperatorBase( div, opsetVersion, O_Div ) {}
};

// Eltwise operators with any number of inputs

// Sum operator
class CSumOperator : public CEltwiseOperatorBase {
public:
	CSumOperator( const onnx::NodeProto& sum, int opsetVersion ) :
		CEltwiseOperatorBase( sum, opsetVersion, O_Add ) {}

	// CEltwiseOperatorBase methods
	CBroadcastInfo BroadcastInfo() const override;
};

} // namespace NeoOnnx
