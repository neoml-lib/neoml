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

// Base class for global pool operators
class CGlobalPoolOperatorBase : public CLayerOperator {
public:
	// Pooling types
	enum TPoolType {
		PT_Max, // Max pooling
		PT_Min, // Min pooling
		PT_Mean, // Mean pooling
		PT_Sum, // Sum pooling
		PT_L2, // L2 pooling

		PT_Count
	};

	// Interface
	// Default implementations are used for Global* operators
	// In other operators these methods are overridden

	// Should pooled dims be kept of size 1 or removed from tensor shape
	virtual bool KeepDims() const { return true; }

	// Get indices of dimensions to be pooled (in Onnx order)
	// Result shouldn't contain negative indices and must be in sorted order
	virtual void PoolAxes( const CTensorShape& inputShape, CFastArray<int, 8>& axes ) const;

protected:
	CGlobalPoolOperatorBase( TPoolType poolType, const onnx::NodeProto& onnxNode, int opsetVersion );

	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const final;

private:
	// Pool type
	TPoolType poolType;

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
class CGlobalMaxPoolOperator : public CGlobalPoolOperatorBase {
public:
	CGlobalMaxPoolOperator( const onnx::NodeProto& onnxNode, int opsetVersion ) :
		CGlobalPoolOperatorBase( PT_Max, onnxNode, opsetVersion )
	{
	}
};

// GlobalAveragePool operator
class CGlobalAveragePoolOperator : public CGlobalPoolOperatorBase {
public:
	CGlobalAveragePoolOperator( const onnx::NodeProto& onnxNode, int opsetVersion ) :
		CGlobalPoolOperatorBase( PT_Mean, onnxNode, opsetVersion )
	{
	}
};

// --------------------------------------------------------------------------------------------------------------------
// Reduce operators which can be emulated via glob pooling

// Base class
class CReducePoolOperatorBase : public CGlobalPoolOperatorBase {
public:
	// CGlobalPoolOperatorBase methods
	bool KeepDims() const override;
	void PoolAxes( const CTensorShape& inputShape, CFastArray<int, 8>& axes ) const override;

protected:
	CReducePoolOperatorBase( TPoolType poolType, const onnx::NodeProto& onnxNode, int opsetVersion ) :
		CGlobalPoolOperatorBase( poolType, onnxNode, opsetVersion )
	{
	}
};

// ReduceMax operator
class CReduceMaxOperator : public CReducePoolOperatorBase {
public:
	CReduceMaxOperator( const onnx::NodeProto& onnxNode, int opsetVersion ) :
		CReducePoolOperatorBase( CGlobalPoolOperatorBase::PT_Max, onnxNode, opsetVersion )
	{
	}
};

// ReduceMean operator
class CReduceMeanOperator : public CReducePoolOperatorBase {
public:
	CReduceMeanOperator( const onnx::NodeProto& onnxNode, int opsetVersion ) :
		CReducePoolOperatorBase( CGlobalPoolOperatorBase::PT_Mean, onnxNode, opsetVersion )
	{
	}
};

// ReduceMin operator
class CReduceMinOperator : public CReducePoolOperatorBase {
public:
	CReduceMinOperator( const onnx::NodeProto& onnxNode, int opsetVersion ) :
		CReducePoolOperatorBase( CGlobalPoolOperatorBase::PT_Min, onnxNode, opsetVersion )
	{
	}
};

// ReduceSum operator
class CReduceSumOperator : public CReducePoolOperatorBase {
public:
	CReduceSumOperator( const onnx::NodeProto& onnxNode, int opsetVersion ) :
		CReducePoolOperatorBase( CGlobalPoolOperatorBase::PT_Sum, onnxNode, opsetVersion )
	{
	}
};

// ReduceL2 operator
class CReduceL2Operator : public CReducePoolOperatorBase {
public:
	CReduceL2Operator( const onnx::NodeProto& onnxNode, int opsetVersion ) :
		CReducePoolOperatorBase( CGlobalPoolOperatorBase::PT_L2, onnxNode, opsetVersion )
	{
	}
};

} // namespace NeoOnnx
