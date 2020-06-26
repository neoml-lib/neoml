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

#include <exception>
#include <string>

#include "onnx.pb.h"

namespace NeoOnnx {

#ifdef NEOML_USE_FINEOBJ

static CError COnnxError( L"%0" );

// Throws std::logic_error with text 'what' if 'expr' is false
inline void NeoOnnxCheck( bool expr, const CString& what )
{
	check( expr, COnnxError, what.CreateUnicodeString() );
}

#else

// Throws std::logic_error with text 'what' if 'expr' is false
inline void NeoOnnxCheck( bool expr, const CString& what )
{
	if( !( expr ) ) {
		FineDebugBreak();
		throw std::logic_error( what );
	}
}

#endif

// Adds node info to message 'what'
inline CString GetMessageWithNodeInfo( const CString& what, const onnx::NodeProto& node )
{
	return CString( what ) + " at node " + node.op_type().c_str() + "(" + node.output( 0 ).c_str() + ")";
}

// Throws std::logic_error if 'expr' is false
// Used for checking if graph and nodes stick to the ONNX protocol
inline void CheckOnnxProtocol( bool expr, const CString& what )
{
	NeoOnnxCheck( expr, CString( "ONNX protocol violation: " ) + what );
}

inline void CheckOnnxProtocol( bool expr, const CString& what, const onnx::NodeProto& node )
{
	CheckOnnxProtocol( expr, GetMessageWithNodeInfo( what, node )  );
}

// Throws std::logic_error if 'expr' is false
// Used for checking if graph and nodes have something which is not supported by NeoOnnx
inline void CheckNeoOnnxSupport( bool expr, const CString& what ) 
{
	NeoOnnxCheck( expr, CString( "Not supported by NeoOnnx: " ) + what );
}

inline void CheckNeoOnnxSupport( bool expr, const CString& what, const onnx::NodeProto& node )
{
	CheckNeoOnnxSupport( expr, GetMessageWithNodeInfo( what, node ) );
}

// Throws std::logic_error if 'expr' is false
// Used for checking if something goes wrong during NeoOnnx work
inline void CheckNeoOnnxInternal( bool expr, const CString& what )
{
	NeoOnnxCheck( expr, CString( "NeoOnnx internal error: " ) + what );
}

inline void CheckNeoOnnxInternal( bool expr, const CString& what, const onnx::NodeProto& node )
{
	CheckNeoOnnxInternal( expr, GetMessageWithNodeInfo( what, node ) );
}

} // namespace NeoOnnx
