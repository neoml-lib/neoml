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

#include "onnx.pb.h"

#include <limits>

#include <NeoML/NeoML.h>
#include "NeoOnnxCheck.h"
#include "Tensor.h"

namespace NeoOnnx {

// Auxiliary data loading functions

// Gets NeoML blob type from onnx tensor's data type
TBlobType GetBlobType( const onnx::TensorProto_DataType& onnxDataType );

// Loads data from raw bytes as an array of TSrc and stores it as an array of TDst (via static_cast)
template<class TSrc, class TDst>
inline void LoadFromRawData( const std::string& rawSrc, TDst* dest )
{
	const TSrc* src = reinterpret_cast<const TSrc*>( rawSrc.data() );
	for( size_t i = 0; i < rawSrc.size() / sizeof( TSrc ); ++i ) {
		TSrc value = src[i];
		if( value >= static_cast<TSrc>( (std::numeric_limits<TDst>::max)() ) ) {
			dest[i] = (std::numeric_limits<TDst>::max)();
		} else if( value <= static_cast<TSrc>( (std::numeric_limits<TDst>::lowest)() ) ) {
			dest[i] = (std::numeric_limits<TDst>::lowest)();
		} else {
			dest[i] = static_cast<TDst>( value );
		}
	}
}

// Loads NeoML's blob data (of type T) from onnx::TensorProto
template<class T>
inline void LoadBlobData( const onnx::TensorProto& src, CDnnBlob& dest )
{
	dest.Clear();
	T* buffer = dest.GetBuffer<T>( 0, dest.GetDataSize() );
	const bool isRaw = src.has_raw_data();
	switch( src.data_type() ) {
		case onnx::TensorProto::FLOAT:
			if( isRaw ) {
				LoadFromRawData<float, T>( src.raw_data(), buffer );
			} else {
				for( int valueIndex = 0; valueIndex < src.float_data_size(); ++valueIndex ) {
					buffer[valueIndex] = static_cast<T>( src.float_data( valueIndex ) );
				}
			}
			break;
		case onnx::TensorProto::DOUBLE:
			// Here downcast may happen (double -> float)
			if( isRaw ) {
				LoadFromRawData<double, T>( src.raw_data(), buffer );
			} else {
				for( int valueIndex = 0; valueIndex < src.double_data_size(); ++valueIndex ) {
					buffer[valueIndex] = static_cast<T>( src.double_data( valueIndex ) );
				}
			}
			break;
		case onnx::TensorProto::BOOL:
		case onnx::TensorProto::INT8:
		case onnx::TensorProto::UINT8:
		case onnx::TensorProto::INT16:
		case onnx::TensorProto::UINT16:
		case onnx::TensorProto::INT32:
			if( isRaw ) {
				LoadFromRawData<int, T>( src.raw_data(), buffer );
			} else {
				for( int valueIndex = 0; valueIndex < src.int32_data_size(); ++valueIndex ) {
					buffer[valueIndex] = static_cast<T>( src.int32_data( valueIndex ) );
				}
			}
			break;
		// NeoML works only with 32-bit signed types
		// But sometimes data values are quite small (like some indices)
		// that's why we can try to load them into NeoML data type
		case onnx::TensorProto::UINT32:
		case onnx::TensorProto::UINT64:
			if( isRaw ) {
				LoadFromRawData<uint64_t, T>( src.raw_data(), buffer );
			} else {
				for( int valueIndex = 0; valueIndex < src.uint64_data_size(); ++valueIndex ) {
					uint64_t value = src.uint64_data( valueIndex );
					if( value >= static_cast<uint64_t>( std::numeric_limits<T>::max() ) ) {
						buffer[valueIndex] = std::numeric_limits<T>::max();
					} else {
						buffer[valueIndex] = static_cast<T>( value );
					}
				}
			}
			break;
		case onnx::TensorProto::INT64:
			if( isRaw ) {
				LoadFromRawData<int64_t, T>( src.raw_data(), buffer );
			} else {
				for( int valueIndex = 0; valueIndex < src.int64_data_size(); ++valueIndex ) {
					int64_t value = src.int64_data( valueIndex );
					if( value >= static_cast<int64_t>( (std::numeric_limits<T>::max)() ) ) {
						buffer[valueIndex] = (std::numeric_limits<T>::max)();
					} else if( value <= static_cast<int64_t>( (std::numeric_limits<T>::lowest)() ) ) {
						buffer[valueIndex] = (std::numeric_limits<T>::lowest)();
					} else {
						buffer[valueIndex] = static_cast<T>( value );
					}
				}
			}
			break;
		case onnx::TensorProto::FLOAT16:
		case onnx::TensorProto::BFLOAT16:
		case onnx::TensorProto::COMPLEX64:
		case onnx::TensorProto::COMPLEX128:
		case onnx::TensorProto::UNDEFINED:
		default:
			CheckNeoOnnxSupport( false, "tensor type" );
	}
	dest.ReleaseBuffer( buffer, true );
}

//---------------------------------------------------------------------------------------------------------------------
// Auxiliary tensor layout functions

// Converts tensor into given layout
CPtr<const CTensorBase> ConvertTensor( const CTensorBase& inputTensor, const CTensorLayout& destLayout );

// Removes dimensions of size 1
// Doesn't add any layers or perform any conversions
CPtr<const CTensorBase> RemoveTensorDims( const CTensorBase& input, const CFastArray<int, 8>& dims );

//---------------------------------------------------------------------------------------------------------------------
// Auxiliary tensor padding functions

// Calculates padding size if autoPad is SAME_*
void CalculatePadding( const CString& autoPad, const CTensorShape& kernelShape, CFastArray<int, 8>& pads );

// Pads tensor with user-dependent data
// Last pads.Size() / 2 dimensions of tensors will be padded (it's compatible with both Conv and Pad onnx operators)
// First pads.Size() / 2 numbers determine padding size at the front of the axis
// Last pads.Size() / 2 numbers determine padding size at the back of the axis
CPtr<const CUserTensor> PadUserTensor( const CUserTensor& input, const CFastArray<int, 8>& pads, float padValue );

//---------------------------------------------------------------------------------------------------------------------
// Auxiliary tensor broadcast functions

// Tensor broadcast types
enum TBroadcastType {
	// Broadcast not supported
	BT_None,
	// Onnx custom broadcast, used in some older versions
	BT_Onnx,
	// Numpy-style broadcast, used in later versions of ONNX
	BT_Numpy,

	BT_Count
};

// Broadcast info
struct CBroadcast {
	// Broadcast type
	TBroadcastType Type;
	// Broadcasted axis index
	int Axis;

	explicit CBroadcast( TBroadcastType type, int axis = NotFound ) :
		Type( type ), Axis( axis ) {}
};

// Calculates shape of the result of the broadcast operation
// If shapes can be broadcasted writes broadcasted shape to the result and returns true
// Returns false if shapes can't be broadcasted (in this case result will be empty)
bool BroadcastTensorShape( const CTensorShape& first, const CTensorShape& second, const CBroadcast& broadcast, CTensorShape& result );

// Broadcasts the given tensor to the given outputShape according to given broadcast
CPtr<const CTensorBase> BroadcastTensor( const CTensorBase& input, const CBroadcast& broadcast, const CTensorShape& outputShape );

} // namespace NeoOnnx
