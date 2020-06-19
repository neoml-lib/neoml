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

#include "common.h"
#pragma hdrstop

#include "Tensor.h"
#include "NodeAttributes.h"
#include "TensorUtils.h"

#include <onnx.pb.h>

#include <utility>

namespace NeoOnnx {

CTensor::CTensor( TTensorType _type, const CTensorShape& _shape, CDnnBlob* _data ) :
	type( _type ),
	data( _data )
{
	_shape.CopyTo( shape );
	static_assert( TT_Count == 2, "TT_Count != 2" );
	switch( type ) {
		case TT_ConstantTensor:
			NeoAssert( data != nullptr );
			NeoAssert( shape.Size() > 0 );
			break;
		case TT_DataTensor:
			NeoAssert( data == nullptr );
			NeoAssert( shape.Size() > 0 );
			break;
		default:
			NeoAssert( false );
	}
}

CTensor::CTensor( const CTensor& other ) :
	type( other.type ),
	data( other.data )
{
	other.shape.CopyTo( shape );
}

const CTensorShape& CTensor::GetShape() const
{
	NeoAssert( type == TT_DataTensor || type == TT_ConstantTensor );
	return shape;
}

const CDnnBlob* CTensor::GetData() const
{
	NeoAssert( type == TT_ConstantTensor );
	return data;
}

CDnnBlob* CTensor::GetData()
{
	NeoAssert( type == TT_ConstantTensor );
	return data;
}

bool CTensor::SetTensorDim( const CTensorDim& supposedDim )
{
	if( tensorDim.IsEmpty() ) {
		if( supposedDim.Size() == shape.Size() ) {
			// It's the first request for a match.
			// And the number of dimensions is matching with the shape.
			supposedDim.CopyTo( tensorDim );
			return true;
		}
		// Dimensions number mismatch...
		return false;
	}

	if( supposedDim.Size() != tensorDim.Size() ) {
		// Dimensions number mismatch...
		return false;
	}

	for( int dimIndex = 0; dimIndex < supposedDim.Size(); ++dimIndex ) {
		if( tensorDim[dimIndex] != supposedDim[dimIndex] ) {
			// Supposed dimensions don't match with the previously set one.
			return false;
		}
	}

	// Number of dimensions and their values match.
	return true;
}

} // namespace NeoOnnx
