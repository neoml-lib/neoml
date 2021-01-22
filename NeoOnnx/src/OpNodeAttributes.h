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

// Forward declaration(s)
namespace onnx {
class NodeProto;
class AttributeProto;
} // namespace onnx

namespace NeoOnnx {

// Onnx operator node attributes
class COpNodeAttributes {
public:
	explicit COpNodeAttributes( const onnx::NodeProto& node );

	// Getters for optional attributes
	// Returns default values (or empty arrays) if attribute is missing
	int GetOptionalInt( const CString& name, int defaultValue ) const;
	float GetOptionalFloat( const CString& name, float defaultValue ) const;
	void GetOptionalIntArray( const CString& name, CArray<int>& value ) const;
	void GetOptionalIntArray( const CString& name, CFastArray<int, 8>& value ) const;
	CString GetOptionalString( const CString& name, const CString& defaultValue ) const;
	CPtr<CDnnBlob> GetOptionalTensor( const CString& name, CDnnBlob& defaultValue ) const;

	// Getters for required attributes
	// CheckOnnxProtocol( false ) if required attribute is missing
	int GetRequiredInt( const CString& name ) const;
	float GetRequiredFloat( const CString& name ) const;
	void GetRequiredIntArray( const CString& name, CArray<int>& value ) const;
	void GetRequiredIntArray( const CString& name, CFastArray<int, 8>& value ) const;
	void GetRequiredInt64Array( const CString& name, CArray<int64_t>& value ) const;
	CString GetRequiredString( const CString& name ) const;
	CPtr<CDnnBlob> GetRequiredTensor( const CString& name, IMathEngine& mathEngine ) const;

private:
	const onnx::NodeProto& onnxNode; // reference to the node for error handling
	CMap<CString, const onnx::AttributeProto*> attributes; // mapping between attribute keys and values
};

} // namespace NeoOnnx
