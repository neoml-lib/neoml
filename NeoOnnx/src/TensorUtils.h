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

#include <NeoML/NeoML.h>

#include "onnx.pb.h"

namespace NeoOnnx {

// Gets NeoML blob type from ONNX tensor's data type.
inline TBlobType GetBlobType( const onnx::TensorProto_DataType& onnxDataType )
{
	switch( onnxDataType ) {
		case onnx::TensorProto::FLOAT:
		case onnx::TensorProto::DOUBLE:
			return CT_Float;
		case onnx::TensorProto::BOOL:
		case onnx::TensorProto::INT8:
		case onnx::TensorProto::UINT8:
		case onnx::TensorProto::INT16:
		case onnx::TensorProto::UINT16:
		case onnx::TensorProto::INT32:
		case onnx::TensorProto::UINT32:
		case onnx::TensorProto::INT64:
		case onnx::TensorProto::UINT64:
			return CT_Int;
		case onnx::TensorProto::FLOAT16:
		case onnx::TensorProto::BFLOAT16:
		case onnx::TensorProto::COMPLEX64:
		case onnx::TensorProto::COMPLEX128:
		case onnx::TensorProto::UNDEFINED:
		default:
			NeoAssert( false );
	}
	return CT_Invalid;
}

template<class TSrc, class TDst>
inline void LoadFromRawData( const std::string& rawSrc, TDst* dest )
{
	const TSrc* src = reinterpret_cast<const TSrc*>( rawSrc.data() );
	for( size_t i = 0; i < rawSrc.size() / sizeof( TSrc ); ++i ) {
		dest[i] = static_cast<TDst>( src[i] );
	}
}

// Loads NeoML's blob data (of type T) from onnx::TensorProto.
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
					buffer[valueIndex] = static_cast< T >( src.float_data( valueIndex ) );
				}
			}
			break;
		case onnx::TensorProto::DOUBLE:
			if( isRaw ) {
				LoadFromRawData<int, T>( src.raw_data(), buffer );
			} else {
				for( int valueIndex = 0; valueIndex < src.double_data_size(); ++valueIndex ) {
					buffer[valueIndex] = static_cast< T >( src.double_data( valueIndex ) );
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
					buffer[valueIndex] = static_cast< T >( src.int32_data( valueIndex ) );
				}
			}
			break;
		case onnx::TensorProto::UINT32:
		case onnx::TensorProto::UINT64:
			if( isRaw ) {
				LoadFromRawData<uint64_t, T>( src.raw_data(), buffer );
			} else {
				for( int valueIndex = 0; valueIndex < src.uint64_data_size(); ++valueIndex ) {
					buffer[valueIndex] = static_cast< T >( src.uint64_data( valueIndex ) );
				}
			}
			break;
		case onnx::TensorProto::INT64:
			if( isRaw ) {
				LoadFromRawData<int64_t, T>( src.raw_data(), buffer );
			} else {
				for( int valueIndex = 0; valueIndex < src.int64_data_size(); ++valueIndex ) {
					buffer[valueIndex] = static_cast< T >( src.int64_data( valueIndex ) );
				}
			}
			break;
		case onnx::TensorProto::FLOAT16:
		case onnx::TensorProto::BFLOAT16:
		case onnx::TensorProto::COMPLEX64:
		case onnx::TensorProto::COMPLEX128:
		case onnx::TensorProto::UNDEFINED:
		default:
			NeoAssert( false );
	}
	dest.ReleaseBuffer( buffer, true );
}

} // namespace NeoOnnx
