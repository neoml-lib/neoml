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

// Base class for global pool nodes
class CGlobalPoolNodeBase : public CLayerOpNode {
public:
	// Pooling types
	enum TPoolType {
		PT_Max, // Max pooling
		PT_Min, // Min pooling
		PT_Mean, // Mean pooling

		PT_Count
	};

	// Interface
	// Default implementations are used for GlobalMax and GlobalAverage operators
	// and are overridden in other operators

	// Should pooled dims be kept of size 1 or removed from tensor shape
	virtual bool KeepDims() const { return true; }

	// Get indices of dimensions to be pooled (in Onnx order)
	// Result shouldn't contain negative indices and must be in sorted order
	virtual void PoolAxes( const CTensorShape& inputShape, CFastArray<int, 8>& axes ) const;

	// Implementations

	// CNode methods
	void AddLayers( const CObjectArray<const CTensorBase>& inputs,
		CDnn& dnn, CObjectArray<const CTensorBase>& outputs ) override;

	// COpNode methods
	void UserInputMask( CUserInputMask& mask ) const override { mask.Add( true ); }

protected:
	CGlobalPoolNodeBase( TPoolType poolType, const onnx::NodeProto& onnxNode, int opsetVersion );

private:
	TPoolType poolType; // pool type

	CPtr<const CUserTensor> prepareInput( const CUserTensor& input, const CFastArray<int, 8>& axes, CDnn& dnn ) const;
	CPtr<const CUserTensor> convertInputLayout( const CUserTensor& input, const CFastArray<int, 8>& axes ) const;

	CPtr<const CUserTensor> addPoolingLayer( const CUserTensor& preparedInput, const CFastArray<int, 8>& axes, CDnn& dnn ) const;
	void calcOutputShape( const CTensorShape& inputShape, const CFastArray<int, 8>& axes, CTensorShape& outputShape ) const;
	CTensorLayout calcOutputLayout( const CTensorLayout& inputLayout, const CFastArray<int, 8>& axes ) const;

	CPtr<const CUserTensor> addPostProcessing( const CUserTensor& layerOutput, CDnn& dnn ) const;
};

// --------------------------------------------------------------------------------------------------------------------
// Global pool operators

// GlobalMaxPool operator
class CGlobalMaxPoolNode : public CGlobalPoolNodeBase {
public:
	CGlobalMaxPoolNode( const onnx::NodeProto& onnxNode, int opsetVersion ) :
		CGlobalPoolNodeBase( PT_Max, onnxNode, opsetVersion )
	{
	}
};

// GlobalAveragePool operator
class CGlobalAveragePoolNode : public CGlobalPoolNodeBase {
public:
	CGlobalAveragePoolNode( const onnx::NodeProto& onnxNode, int opsetVersion ) :
		CGlobalPoolNodeBase( PT_Mean, onnxNode, opsetVersion )
	{
	}
};

// --------------------------------------------------------------------------------------------------------------------
// Reduce operators which can be emulated via glob pooling

// Base class
class CReducePoolNodeBase : public CGlobalPoolNodeBase {
public:
	// CGlobalPoolNodeBase methods
	bool KeepDims() const override;
	void PoolAxes( const CTensorShape& inputShape, CFastArray<int, 8>& axes ) const override;

protected:
	CReducePoolNodeBase( TPoolType poolType, const onnx::NodeProto& onnxNode, int opsetVersion ) :
		CGlobalPoolNodeBase( poolType, onnxNode, opsetVersion )
	{
	}
};

// ReduceMax operator
class CReduceMaxNode : public CReducePoolNodeBase {
public:
	CReduceMaxNode( const onnx::NodeProto& onnxNode, int opsetVersion ) :
		CReducePoolNodeBase( CGlobalPoolNodeBase::PT_Max, onnxNode, opsetVersion )
	{
	}
};

// ReduceMean operator
class CReduceMeanNode : public CReducePoolNodeBase {
public:
	CReduceMeanNode( const onnx::NodeProto& onnxNode, int opsetVersion ) :
		CReducePoolNodeBase( CGlobalPoolNodeBase::PT_Mean, onnxNode, opsetVersion )
	{
	}
};

// ReduceMin operator
class CReduceMinNode : public CReducePoolNodeBase {
public:
	CReduceMinNode( const onnx::NodeProto& onnxNode, int opsetVersion ) :
		CReducePoolNodeBase( CGlobalPoolNodeBase::PT_Min, onnxNode, opsetVersion )
	{
	}
};

} // namespace NeoOnnx
