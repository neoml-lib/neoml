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

#include "TensorLayout.h"

namespace NeoOnnx {

// Tensor shape
typedef CFastArray<int, 8> CTensorShape;

struct CLayerOutput
{
	CLayerOutput() : Layer( nullptr ), OutputIndex( NotFound ) {}
	CLayerOutput( CBaseLayer* layer, int outputIndex ) :
		Layer( layer ), OutputIndex( outputIndex ) {}

	CBaseLayer* Layer;
	int OutputIndex;
};

// Base class for tensor in onnx graph
class CTensorBase : public virtual IObject {
public:
	// Tensor's shape. The shape always describes Onnx axes.
	const CTensorShape& Shape() const { return shape; }

	// Tensor's layout. Contains info about how tensors is represented in memory.
	const CTensorLayout& Layout() const { return layout; }

	// Returns true if tensor's data doesn't depend on user data
	// Used for optimization (avoid unnecessary dynammic_cast)
	virtual bool IsCalculated() const = 0;
	
protected:
	CTensorBase( const CTensorShape& _shape, const CTensorLayout& _layout ) :
		layout( _layout ) { _shape.CopyTo( shape ); }
	CTensorBase( const CTensorBase& other ) = delete;
	CTensorBase& operator=( const CTensorBase& other ) = delete;
	virtual ~CTensorBase() = default;

private:
	// Tensor's shape. Always on Onnx order.
	CTensorShape shape;

	// Information about how tensor is represented in memory
	CTensorLayout layout;
};

// All tensors during Onnx processing can be divided into 2 groups:
//
// 1. The tensors whose data depend on the user input. These tensors can't be calculated during conversion.
// In that case tensor is the result of the work of a layer.
//
// 2. The tensors whose data doesn't depend on user input.
// These tensors' data can (and should) be calculated during generation.
// Usually these tensors contain trained weights of the model.

// Tensor with data depending on user input
class CUserTensor : public CTensorBase {
public:
	CUserTensor( const CTensorShape& shape, const CTensorLayout& layout, const CLayerOutput& output ) :
		CTensorBase( shape, layout ), layerOutput( output ) {}

	// CTensorBase methods implementation
	bool IsCalculated() const override { return false; }

	// Information about corresponding layer and its' output index
	const CLayerOutput& LayerOutput() const { return layerOutput; }
	CBaseLayer* Layer() const { return layerOutput.Layer; }
	int OutputIndex() const { return layerOutput.OutputIndex; }

private:
	// Information about corresponding layer and its' output index
	CLayerOutput layerOutput;
};

// Tensor with data independent of user input
class CDataTensor : public CTensorBase {
public:
	CDataTensor( const CTensorShape& shape, const CTensorLayout& layout, const CDnnBlob& _data ) :
		CTensorBase( shape, layout ), data( &_data ) {}

	// CTensorBase methods implementation
	bool IsCalculated() const override { return true; }

	// Blob with data
	// Data ordering depends on CTensorBase::GetLayout
	const CDnnBlob* Data() const { return data.Ptr(); }

private:
	// Blob with data
	CPtr<const CDnnBlob> data;
};

} // namespace NeoOnnx
