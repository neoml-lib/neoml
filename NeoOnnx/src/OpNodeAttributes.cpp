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

#include "OpNodeAttributes.h"
#include "TensorUtils.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

COpNodeAttributes::COpNodeAttributes( const onnx::NodeProto& onnxNode, const COpNode& _node ) :
	node( _node )
{
	for( const onnx::AttributeProto& attribute : onnxNode.attribute() ) {
		attributes.Add( attribute.name().c_str(), &attribute );
	}
}

// Extracts value of type T from attribute
// CheckOnnxProtocol( false ) if attribute doesn't contain value of required type
template<class T>
static void extractValue( const onnx::AttributeProto& attribute, T& /*value*/, const COpNode& node )
{
	CheckOnnxProtocol( false, CString( "attribute " ) + attribute.name().c_str() + " has unsupported data type", node );
}

template<>
void extractValue<int>( const onnx::AttributeProto& attribute, int& value, const COpNode& node )
{
	CheckOnnxProtocol( attribute.has_i(), CString( "attribute " ) + attribute.name().c_str() + " is not an int", node );
	value = static_cast<int>( attribute.i() );
}

template<>
void extractValue<float>( const onnx::AttributeProto& attribute, float& value, const COpNode& node )
{
	CheckOnnxProtocol( attribute.has_f(), CString( "attribute " ) + attribute.name().c_str() + " is not a float", node );
	value = static_cast<float>( attribute.f() );
}

template<>
void extractValue<CString>( const onnx::AttributeProto& attribute, CString& value, const COpNode& node )
{
	CheckOnnxProtocol( attribute.has_s(), CString( "attribute " ) + attribute.name().c_str() + " is not a string", node );
	value = attribute.s().c_str();
}

template<>
void extractValue<CArray<int>>( const onnx::AttributeProto& attribute, CArray<int>& value,
	const COpNode& /* node */ )
{
	value.Empty();
	value.SetBufferSize( attribute.ints_size() );
	for( int64_t element : attribute.ints() ) {
		value.Add( static_cast<int>( element ) );
	}
}

template<>
void extractValue<CArray<int64_t>>( const onnx::AttributeProto& attribute, CArray<int64_t>& value,
	const COpNode& /* node */ )
{
	value.Empty();
	value.SetBufferSize( attribute.ints_size() );
	for( int64_t element : attribute.ints() ) {
		value.Add( element );
	}
}

template<>
void extractValue<CFastArray<int, 8>>( const onnx::AttributeProto& attribute, CFastArray<int, 8>& value,
	const COpNode& /* node */ )
{
	for( int64_t element : attribute.ints() ) {
		if( element >= static_cast<int64_t>( INT_MAX ) ) {
			value.Add( INT_MAX );
		} else if( element <= static_cast<int64_t>( INT_MIN ) ) {
			value.Add( INT_MIN );
		} else {
			value.Add( static_cast< int >( element ) );
		}
	}
}

template<>
void extractValue<CPtr<CDataTensor>>( const onnx::AttributeProto& attribute, CPtr<CDataTensor>& value,
	const COpNode& node )
{
	CheckOnnxProtocol( attribute.has_t(), CString( "attribute " ) + attribute.name().c_str() + " is not a tensor", node );
	
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

// Gets value of type T from the attribute 'name'
// Returns true and fills the value if attribute is present
// Returns false if attribute is missing
template<class T>
static bool getValue( const CString& name, const CMap<CString, const onnx::AttributeProto*>& attributes, T& value,
	const COpNode& node )
{
	const int attrPos = attributes.GetFirstPosition( name );
	if( attrPos == NotFound ) {
		return false;
	}

	const onnx::AttributeProto* attributeValue = attributes.GetValue( attrPos );
	CheckNeoOnnxInternal( attributeValue != nullptr, CString( "attribute " ) + name + " is nullptr", node );

	extractValue<T>( *attributeValue, value, node );
	return true;
}

int COpNodeAttributes::GetOptionalInt( const CString& name, int defaultValue ) const
{
	int result = defaultValue;
	getValue( name, attributes, result, node );
	return result;
}

float COpNodeAttributes::GetOptionalFloat( const CString& name, float defaultValue ) const
{
	float result = defaultValue;
	getValue( name, attributes, result, node );
	return result;
}

void COpNodeAttributes::GetOptionalIntArray( const CString& name, CArray<int>& value ) const
{
	getValue( name, attributes, value, node );
}

void COpNodeAttributes::GetOptionalIntArray( const CString& name, CFastArray<int, 8>& value ) const
{
	getValue( name, attributes, value, node );
}

CString COpNodeAttributes::GetOptionalString( const CString& name, const CString& defaultValue ) const
{
	CString result = defaultValue;
	getValue( name, attributes, result, node );
	return result;
}

CPtr<CDataTensor> COpNodeAttributes::GetOptionalTensor( const CString& name, CDataTensor* defaultValue, IMathEngine& mathEngine ) const
{
	CPtr<CDnnBlob> resultBlob = CDnnBlob::CreateVector( mathEngine, CT_Float, 1 );
	CPtr<CDataTensor> result = new CDataTensor( { 1 }, CTensorLayout( 1 ), *resultBlob );
	if( getValue( name, attributes, result, node ) ) {
		return result;
	}
	return defaultValue;
}

int COpNodeAttributes::GetRequiredInt( const CString& name ) const
{
	int value = 0;
	CheckOnnxProtocol( getValue( name, attributes, value, node ), "required attribute is missing: " + name, node );
	return value;
}

float COpNodeAttributes::GetRequiredFloat( const CString& name ) const
{
	float result = 0.f;
	CheckOnnxProtocol( getValue( name, attributes, result, node ), "required attribute is missing: " + name, node );
	return result;
}

void COpNodeAttributes::GetRequiredIntArray( const CString& name, CArray<int>& value ) const
{
	CheckOnnxProtocol( getValue( name, attributes, value, node ), "required attribute is missing: " + name, node );
}

void COpNodeAttributes::GetRequiredIntArray( const CString& name, CFastArray<int, 8>& value ) const
{
	CheckOnnxProtocol( getValue( name, attributes, value, node ), "required attribute is missing: " + name, node );
}

void COpNodeAttributes::GetRequiredInt64Array( const CString& name, CArray<int64_t>& value ) const
{
	CheckOnnxProtocol( getValue( name, attributes, value, node ), "required attribute is missing: " + name, node );
}

CString COpNodeAttributes::GetRequiredString( const CString& name ) const
{
	CString result;
	CheckOnnxProtocol( getValue( name, attributes, result, node ), "required attribute is missing: " + name, node );
	return result;
}

CPtr<CDataTensor> COpNodeAttributes::GetRequiredTensor( const CString& name, IMathEngine& mathEngine ) const
{
	CPtr<CDnnBlob> resultBlob = CDnnBlob::CreateVector( mathEngine, CT_Float, 1 );
	CPtr<CDataTensor> result = new CDataTensor( { 1 }, CTensorLayout( 1 ), *resultBlob );
	CheckOnnxProtocol( getValue( name, attributes, result, node ), "required attribute is missing: " + name, node );
	return result;
}

} // namespace NeoOnnx
