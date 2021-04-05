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

#include "../Node.h"

namespace NeoOnnx {

// Base class for nodes which perform eltwise operations
class CEltwiseNodeBase : public CLayerOpNode {
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

	// CNode methods
	void AddLayers( const CObjectArray<const CTensorBase>& inputs,
		CObjectArray<const CTensorBase>& outputs, CDnn& dnn ) override;

	// COpNode methods
	// We can guarantee the support only when second input is CDataTensor (other division is impossible)
	void UserInputMask( CUserInputMask& mask ) const override
		{ mask.Add( true ); mask.Add( false, InputCount() - 1 ); }

	// In some versions different nodes supported different broadcast types
	// E.g. 'Add' node in opset v1 supports onnx-broadcast but 'Sum' node doesn't support broadcast at all
	// That's why each derivative should determine by itself which broadcast type is supported
	virtual CBroadcastInfo BroadcastInfo() const = 0;

protected:
	CEltwiseNodeBase( const onnx::NodeProto& eltwise, int opsetVersion, TOperation operation, int argsNum = NotFound );

private:
	TOperation operation; // Operation performed by this node
	int argsNum; // Expected number of arguments (-1 if any number is supported)

	bool broadcastShape( const CTensorShape& first, const CTensorShape& second,
		const CBroadcastInfo& broadcast, CTensorShape& result ) const;
	CPtr<const CTensorBase> broadcast( const CTensorBase& input, const CBroadcastInfo& broadcast, const CTensorShape& outputShape ) const;
	CPtr<const CDataTensor> broadcast( const CDataTensor& input, const CBroadcastInfo& broadcast, const CTensorShape& outputShape ) const;
	CPtr<const CUserTensor> broadcast( const CUserTensor& input, const CBroadcastInfo& broadcast, const CTensorShape& outputShape ) const;
	CPtr<const CTensorBase> prepareSecondInput( const CObjectArray<const CTensorBase>& inputs ) const;
	CPtr<const CUserTensor> padTensorShape( const CUserTensor& input, int dimCount, int axis ) const;
};

// Eltwise operator nodes with 2 inputs

// Base class
class CEltwiseBinaryOpNodeBase : public CEltwiseNodeBase {
public:
	CEltwiseBinaryOpNodeBase( const onnx::NodeProto& eltwise, int opsetVersion, TOperation operation ) :
		CEltwiseNodeBase( eltwise, opsetVersion, operation, 2 ) {}

	// CEltwiseNodeBase methods
	CBroadcastInfo BroadcastInfo() const override;
};

// Add operator graph node
class CAddNode : public CEltwiseBinaryOpNodeBase {
public:
	CAddNode( const onnx::NodeProto& add, int opsetVersion ) :
		CEltwiseBinaryOpNodeBase( add, opsetVersion, O_Add ) {}
};

// Sub operator graph node
class CSubNode : public CEltwiseBinaryOpNodeBase {
public:
	CSubNode( const onnx::NodeProto& sub, int opsetVersion ) :
		CEltwiseBinaryOpNodeBase( sub, opsetVersion, O_Sub ) {}
};

// Mul operator graph node
class CMulNode : public CEltwiseBinaryOpNodeBase {
public:
	CMulNode( const onnx::NodeProto& mul, int opsetVersion ) :
		CEltwiseBinaryOpNodeBase( mul, opsetVersion, O_Mul ) {}
};

// Div operator graph node
class CDivNode : public CEltwiseBinaryOpNodeBase {
public:
	CDivNode( const onnx::NodeProto& div, int opsetVersion ) :
		CEltwiseBinaryOpNodeBase( div, opsetVersion, O_Div ) {}
};

// Eltwise operator nodes with any number of inputs

// Sum operator graph node
class CSumNode : public CEltwiseNodeBase {
public:
	CSumNode( const onnx::NodeProto& sum, int opsetVersion ) :
		CEltwiseNodeBase( sum, opsetVersion, O_Add ) {}

	// CEltwiseNodeBase methods
	CBroadcastInfo BroadcastInfo() const override;
};

} // namespace NeoOnnx
