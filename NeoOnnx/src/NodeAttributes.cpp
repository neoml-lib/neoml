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

#include "NodeAttributes.h"
#include "TensorUtils.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CNodeAttributes::CNodeAttributes( const onnx::NodeProto& node )
{
	for( const onnx::AttributeProto& attribute : node.attribute() ) {
		attributes.Add( attribute.name().c_str(), &attribute );
	}
}

template<class T>
static void extractValue( const onnx::AttributeProto& attribute, T& /*value*/ )
{
	CheckOnnxProtocol( false, CString( "attribute " ) + attribute.name().c_str() + " is of unsupported data type" );
}

template<>
void extractValue<int>( const onnx::AttributeProto& attribute, int& value )
{
	CheckOnnxProtocol( attribute.has_i(), CString( "attribute " ) + attribute.name().c_str() + " is not an int" );
	value = static_cast<int>( attribute.i() );
}

template<>
void extractValue<float>( const onnx::AttributeProto& attribute, float& value )
{
	CheckOnnxProtocol( attribute.has_f(), CString( "attribute " ) + attribute.name().c_str() + " is not a float" );
	value = static_cast<float>( attribute.f() );
}

template<>
void extractValue<CString>( const onnx::AttributeProto& attribute, CString& value )
{
	CheckOnnxProtocol( attribute.has_s(), CString( "attribute " ) + attribute.name().c_str() + " is not a string" );
	value = attribute.s().c_str();
}

template<>
void extractValue<CArray<int>>( const onnx::AttributeProto& attribute, CArray<int>& value )
{
	value.Empty();
	value.SetBufferSize( attribute.ints_size() );
	for( int64_t element : attribute.ints() ) {
		value.Add( static_cast<int>( element ) );
	}
}

template<>
void extractValue<CArray<int64_t>>( const onnx::AttributeProto& attribute, CArray<int64_t>& value )
{
	value.Empty();
	value.SetBufferSize( attribute.ints_size() );
	for( int64_t element : attribute.ints() ) {
		value.Add( element );
	}
}

template<>
void extractValue<CFastArray<int, 8>>( const onnx::AttributeProto& attribute, CFastArray<int, 8>& value )
{
	for( int64_t element : attribute.ints() ) {
		value.Add( static_cast<int>( element ) );
	}
}

template<>
void extractValue<CPtr<CDnnBlob>>( const onnx::AttributeProto& attribute, CPtr<CDnnBlob>& value )
{
	CheckOnnxProtocol( attribute.has_t(), CString( "attribute " ) + attribute.name().c_str() + " is not a tensor" );
	TBlobType resultDataType = GetBlobType( static_cast<onnx::TensorProto_DataType>( attribute.t().data_type() ) );
	value = CDnnBlob::CreateVector( value->GetMathEngine(), resultDataType, 1 );
	if( resultDataType == CT_Float ) {
		LoadBlobData<float>( attribute.t(), *value );
	} else {
		LoadBlobData<int>( attribute.t(), *value );
	}
}

template<class T>
static bool getValue( const CString& name, const CMap<CString, const onnx::AttributeProto*>& attributes, T& value )
{
	const int attrPos = attributes.GetFirstPosition( name );
	if( attrPos == NotFound ) {
		return false;
	}

	const onnx::AttributeProto* attributeValue = attributes.GetValue( attrPos );
	CheckNeoOnnxInternal( attributeValue != nullptr, CString( "attribute " ) + name + " has nullptr" );

	extractValue<T>( *attributeValue, value );
	return true;
}

int CNodeAttributes::GetOptionalInt( const CString& name, int defaultValue ) const
{
	int result = defaultValue;
	getValue( name, attributes, result );
	return result;
}

float CNodeAttributes::GetOptionalFloat( const CString& name, float defaultValue ) const
{
	float result = defaultValue;
	getValue( name, attributes, result );
	return result;
}

void CNodeAttributes::GetOptionalIntArray( const CString& name, CArray<int>& value ) const
{
	getValue( name, attributes, value );
}

void CNodeAttributes::GetOptionalIntArray( const CString& name, CFastArray<int, 8>& value ) const
{
	getValue( name, attributes, value );
}

CString CNodeAttributes::GetOptionalString( const CString& name, const CString& defaultValue ) const
{
	CString result = defaultValue;
	getValue( name, attributes, result );
	return result;
}

CPtr<CDnnBlob> CNodeAttributes::GetOptionalTensor( const CString& name, CDnnBlob& defaultValue ) const
{
	CPtr<CDnnBlob> result = &defaultValue;
	getValue( name, attributes, result );
	return result;
}

int CNodeAttributes::GetRequiredInt( const CString& name ) const
{
	int value = 0;
	CheckOnnxProtocol( getValue( name, attributes, value ), "required attribute is missing: " + name );
	return value;
}

float CNodeAttributes::GetRequiredFloat( const CString& name ) const
{
	float result = 0.f;
	CheckOnnxProtocol( getValue( name, attributes, result ), "required attribute is missing: " + name );
	return result;
}

void CNodeAttributes::GetRequiredIntArray( const CString& name, CArray<int>& value ) const
{
	CheckOnnxProtocol( getValue( name, attributes, value ), "required attribute is missing: " + name );
}

void CNodeAttributes::GetRequiredIntArray( const CString& name, CFastArray<int, 8>& value ) const
{
	CheckOnnxProtocol( getValue( name, attributes, value ), "required attribute is missing: " + name );
}

void CNodeAttributes::GetRequiredInt64Array( const CString& name, CArray<int64_t>& value ) const
{
	CheckOnnxProtocol( getValue( name, attributes, value ), "required attribute is missing: " + name );
}

CString CNodeAttributes::GetRequiredString( const CString& name ) const
{
	CString result;
	CheckOnnxProtocol( getValue( name, attributes, result ), "required attribute is missing: " + name );
	return result;
}

CPtr<CDnnBlob> CNodeAttributes::GetRequiredTensor( const CString& name, IMathEngine& mathEngine ) const
{
	CPtr<CDnnBlob> result = CDnnBlob::CreateVector( mathEngine, CT_Float, 1 );
	CheckOnnxProtocol( getValue( name, attributes, result ), "required attribute is missing: " + name );
	return result;
}

} // namespace NeoOnnx
