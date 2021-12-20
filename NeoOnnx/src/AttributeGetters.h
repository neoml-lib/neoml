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

#include "TensorUtils.h"

namespace NeoOnnx {

// This file contains getters for different types of onnx attributes

template<class T>
inline void GetAttributeValue( const onnx::AttributeProto& attribute, T& /* value */, const COperator& op )
{
	CheckNeoOnnxSupport( false, CString( "'" ) + attribute.name().c_str() + "' attribute's type", op );
}

template<>
inline void GetAttributeValue<int>( const onnx::AttributeProto& attribute, int& value, const COperator& op )
{
	CheckOnnxProtocol( attribute.type() == onnx::AttributeProto_AttributeType_INT && attribute.has_i(),
		( attribute.name() + " attribute is not an int" ).c_str(), op );
	value = static_cast<int>( attribute.i() );
}

template<>
inline void GetAttributeValue<float>( const onnx::AttributeProto& attribute, float& value, const COperator& op )
{
	CheckOnnxProtocol( attribute.type() == onnx::AttributeProto_AttributeType_FLOAT && attribute.has_f(),
		( attribute.name() + " attribute is not a float" ).c_str(), op );
	value = static_cast<float>( attribute.f() );
}

template<>
inline void GetAttributeValue<CString>( const onnx::AttributeProto& attribute, CString& value, const COperator& op )
{
	CheckOnnxProtocol( attribute.type() == onnx::AttributeProto_AttributeType_STRING && attribute.has_s(),
		( attribute.name() + " attribute is not a string" ).c_str(), op );
	value = attribute.s().c_str();
}

template<>
inline void GetAttributeValue<CArray<int>>( const onnx::AttributeProto& attribute, CArray<int>& value, const COperator& op )
{
	CheckOnnxProtocol( attribute.type() == onnx::AttributeProto_AttributeType_INTS,
		( attribute.name() + " attribute is not an array of ints" ).c_str(), op );
	value.Empty();
	value.SetBufferSize( attribute.ints_size() );
	for( int64_t element : attribute.ints() ) {
		value.Add( static_cast<int>( clamp( element, static_cast<int64_t>( INT_MIN ), static_cast<int64_t>( INT_MAX ) ) ) );
	}
}

template<>
inline void GetAttributeValue<CArray<int64_t>>( const onnx::AttributeProto& attribute, CArray<int64_t>& value, const COperator& op )
{
	CheckOnnxProtocol( attribute.type() == onnx::AttributeProto_AttributeType_INTS,
		( attribute.name() + " attribute is not an array of ints" ).c_str(), op );
	value.Empty();
	value.SetBufferSize( attribute.ints_size() );
	for( int64_t element : attribute.ints() ) {
		value.Add( element );
	}
}

template<>
inline void GetAttributeValue<CFastArray<int, 8>>( const onnx::AttributeProto& attribute, CFastArray<int, 8>& value, const COperator& op )
{
	CheckOnnxProtocol( attribute.type() == onnx::AttributeProto_AttributeType_INTS,
		( attribute.name() + " attribute is not an array of ints" ).c_str(), op );
	for( int64_t element : attribute.ints() ) {
		value.Add( static_cast<int>( clamp( element, static_cast<int64_t>( INT_MIN ), static_cast<int64_t>( INT_MAX ) ) ) );
	}
}

template<>
inline void GetAttributeValue<CPtr<CDataTensor>>( const onnx::AttributeProto& attribute, CPtr<CDataTensor>& value, const COperator& op )
{
	CheckOnnxProtocol( attribute.type() == onnx::AttributeProto_AttributeType_TENSOR && attribute.has_t(),
		( attribute.name() + " attribute is not a tensor" ).c_str(), op );

	TBlobType resultDataType = GetBlobType( static_cast<onnx::TensorProto_DataType>( attribute.t().data_type() ) );
	CTensorLayout resultLayout( attribute.t().dims().size() );
	CBlobDesc desc( resultDataType );
	CTensorShape resultShape;
	for( int i = 0; i < attribute.t().dims().size(); ++i ) {
		desc.SetDimSize( resultLayout[i], static_cast<int>( attribute.t().dims( i ) ) );
		resultShape.Add( static_cast<int>( attribute.t().dims( i ) ) );
	}
	CPtr<CDnnBlob> resultBlob = CDnnBlob::CreateBlob( value->Data()->GetMathEngine(), resultDataType, desc );

	if( resultDataType == CT_Float ) {
		LoadBlobData<float>( attribute.t(), *resultBlob );
	} else {
		LoadBlobData<int>( attribute.t(), *resultBlob );
	}
	value = new CDataTensor( resultShape, resultLayout, *resultBlob );
}

} // namespace NeoOnnx

