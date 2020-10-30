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
class CEltwiseNodeBase : public COpNode {
public:
	// Supported eltwise operations
	enum TOperation {
		O_Add, // Addition
		O_Sub, // Subtraction
		O_Mul, // Multiplication
		O_Div, // Division

		O_Count
	};

	// CNode methods' realizations
	void CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine ) override;
	void LabelTensorDims( const CTensorCache& tensors, CDimCache& dims ) override;
	void AddLayers( const CGraph& graph, const CTensorCache& tensors, const CDimCache& dims,
		CNeoMLLinkCache& neoMLLinks, CDnn& dnn ) override;

protected:
	CEltwiseNodeBase( int nodeIndex, const onnx::NodeProto& eltwise, int opsetVersion, TOperation operation, int argsNum = -1 );

private:
	TOperation operation; // Operation performed by this node
	int argsNum; // Expected number of arguments (-1 if any number is supported)
	const int axis; // Broadcast axis
	mutable int userInputCached; // Index of the input with data provided by user

	int userInput( const CTensorCache& tensors ) const;
	CPtr<CDnnBlob> broadcast( const CTensor& input, const CTensorShape& outputShape, int axis, bool negative, bool inverted ) const;
	CPtr<CDnnBlob> broadcast( const CTensor& input, const CTensorShape& outputShape, const CTensorDim& outputDim, int axis,
		bool negative, bool inverted ) const;
	CPtr<CDnnBlob> precalcOutput( const CTensorCache& tensors, const CTensorShape& outputShape, IMathEngine& mathEngine ) const;
};

// Eltwise operator nodes with 2 inputs

// Add operator graph node
class CAddNode : public CEltwiseNodeBase {
public:
	CAddNode( int nodeIndex, const onnx::NodeProto& add, int opsetVersion ) :
		CEltwiseNodeBase( nodeIndex, add, opsetVersion, O_Add, 2 ) {}
};

// Sub operator graph node
class CSubNode : public CEltwiseNodeBase {
public:
	CSubNode( int nodeIndex, const onnx::NodeProto& sub, int opsetVersion ) :
		CEltwiseNodeBase( nodeIndex, sub, opsetVersion, O_Sub, 2 ) {}
};

// Mul operator graph node
class CMulNode : public CEltwiseNodeBase {
public:
	CMulNode( int nodeIndex, const onnx::NodeProto& mul, int opsetVersion ) :
		CEltwiseNodeBase( nodeIndex, mul, opsetVersion, O_Mul, 2 ) {}
};

// Div operator graph node
class CDivNode : public CEltwiseNodeBase {
public:
	CDivNode( int nodeIndex, const onnx::NodeProto& div, int opsetVersion ) :
		CEltwiseNodeBase( nodeIndex, div, opsetVersion, O_Div, 2 ) {}
};

// Eltwise operator nodes with any number of inputs

// Sum operator graph node
class CSumNode : public CEltwiseNodeBase {
public:
	CSumNode( int nodeIndex, const onnx::NodeProto& sum, int opsetVersion ) :
		CEltwiseNodeBase( nodeIndex, sum, opsetVersion, O_Add ) {}
};

} // namespace NeoOnnx
