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

// NeoML layer's output
struct CLayerOutput {
	CLayerOutput() : Layer( nullptr ), OutputIndex( NotFound ) {}
	CLayerOutput( CBaseLayer* layer, int outputIndex ) : Layer( layer ), OutputIndex( outputIndex ) {}

	// NeoML layer
	CBaseLayer* Layer;
	// Layer output index (cause layers may have multiple outputs)
	int OutputIndex;
};

// Base class for tensor in onnx graph
class CTensorBase : public virtual IObject {
public:
	// Number of tensors dimensions
	int DimCount() const { return shape.Size(); }

	// Tensor's shape
	// The shape always describes Onnx axes
	const CTensorShape& Shape() const { return shape; }

	// Tensor's layout
	// Contains info about how the tensor is represented in memory
	const CTensorLayout& Layout() const { return layout; }

	// Returns true if tensor's data doesn't depend on user data and was calculated during import
	// Used for optimization (avoid unnecessary dynamic_cast)
	bool IsCalculated() const { return isCalculated; }
	
protected:
	CTensorBase( const CTensorShape& _shape, const CTensorLayout& _layout, bool _isCalculated );
	CTensorBase( const CTensorBase& other ) = delete;
	CTensorBase& operator=( const CTensorBase& other ) = delete;
	virtual ~CTensorBase() = default;

private:
	// Tensor's shape. Always on Onnx order
	CTensorShape shape;

	// Information about how tensor is represented in memory
	const CTensorLayout layout;

	// Indicates whether tensor's data was calculated during import or not
	const bool isCalculated;

	bool checkTensorLayout() const;
};

inline CTensorBase::CTensorBase( const CTensorShape& _shape, const CTensorLayout& _layout, bool _isCalculated ) :
	layout( _layout ),
	isCalculated( _isCalculated )
{
	_shape.CopyTo( shape );
	NeoPresume( checkTensorLayout() );
}

// Checks that layout is consistent with tensor shape (for debug)
// Returns false if inconsistency was found
inline bool CTensorBase::checkTensorLayout() const
{
	const CTensorLayout& layout = Layout();

	if( layout.Size() != Shape().Size() ) {
		return false;
	}

	// Check that every dimension is valid and used only once
	int mask = 0;
	for( int dimIndex = 0; dimIndex < layout.Size(); ++dimIndex ) {
		if( layout[dimIndex] < BD_BatchLength || layout[dimIndex] > BD_Count
			|| ( mask & ( 1 << layout[dimIndex] ) ) != 0 )
		{
			return false;
		}
		mask |= ( 1 << layout[dimIndex] );
	}

	return true;
}

//---------------------------------------------------------------------------------------------------------------------

// All tensors during Onnx processing can be divided into 2 groups:
//
// 1. The tensors whose data depend on the user input. These tensors can't be calculated during import.
// In that case the tensor is an output of a layer in dnn.
//
// 2. The tensors whose data doesn't depend on user input.
// These tensors' data will be calculated during import.
// Usually these tensors contain trained weights of the model.

// Tensor whose data depends on user input
class CUserTensor : public CTensorBase {
public:
	CUserTensor( const CTensorShape& shape, const CTensorLayout& layout, const CLayerOutput& output );

	// Information about corresponding layer and its' output index
	const CLayerOutput& LayerOutput() const { return layerOutput; }
	CBaseLayer* Layer() const { return layerOutput.Layer; }
	int OutputIndex() const { return layerOutput.OutputIndex; }

private:
	// Information about corresponding layer and its' output index
	const CLayerOutput layerOutput;
};

inline CUserTensor::CUserTensor( const CTensorShape& shape, const CTensorLayout& layout, const CLayerOutput& output ) :
	CTensorBase( shape, layout, false ),
	layerOutput( output )
{
	NeoPresume( output.Layer != nullptr );
	NeoPresume( output.Layer->GetDnn() != nullptr );
	NeoPresume( output.OutputIndex >= 0 );
}

//---------------------------------------------------------------------------------------------------------------------

// Tensor with data independent of user input
class CDataTensor : public CTensorBase {
public:
	explicit CDataTensor( IMathEngine& mathEngine );
	CDataTensor( const CTensorShape& shape, const CTensorLayout& layout, const CDnnBlob& data );

	// Blob with data
	// Data ordering depends on CTensorBase::GetLayout
	const CDnnBlob* Data() const { return data.Ptr(); }

private:
	// Blob with data
	const CPtr<const CDnnBlob> data;

	bool checkTensorLayout() const;
};

inline CDataTensor::CDataTensor( IMathEngine& mathEngine ) :
	CTensorBase( CTensorShape(), CTensorLayout(), true ),
	data( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) )
{
	NeoPresume( checkTensorLayout() );
}

inline CDataTensor::CDataTensor( const CTensorShape& shape, const CTensorLayout& layout, const CDnnBlob& _data ) :
	CTensorBase( shape, layout, true ),
	data( &_data )
{
	NeoPresume( checkTensorLayout() );
}

// Checks that layout is consistent with tensor shape (for debug)
// Returns false if inconsistency was found
inline bool CDataTensor::checkTensorLayout() const
{
	// Checking that shape, layout and CDnnBlob are matching
	for( TBlobDim i = BD_BatchLength; i < BD_Count; ++i ) {
		const int index = Layout().Find( i );
		if( index == NotFound && data->DimSize( i ) != 1 ) {
			return false;
		} else if( index != NotFound && Shape()[index] != data->DimSize( i ) ) {
			return false;
		}
	}

	return true;
}

//---------------------------------------------------------------------------------------------------------------------

typedef CObjectArray<const CTensorBase> CTensorArray;

} // namespace NeoOnnx

