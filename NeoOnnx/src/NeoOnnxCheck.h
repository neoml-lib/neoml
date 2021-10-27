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

namespace NeoOnnx {

class COperator;

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
#if defined(_DEBUG) && defined(_WIN32)
		if( ::IsDebuggerPresent() ) {
			FineDebugBreak();
		}
#endif
		throw std::logic_error( what );
	}
}

#endif

// Adds operator info to message 'what'
CString GetMessageWithOperatorInfo( const CString& what, const COperator& op );

// Throws std::logic_error if 'expr' is false
// Checks if onnx protocol has been violated
inline void CheckOnnxProtocol( bool expr, const CString& what )
{
	if( !( expr ) ) {
		NeoOnnxCheck( false, CString( "onnx protocol violation: " ) + what );
	}
}

inline void CheckOnnxProtocol( bool expr, const CString& what, const COperator& op )
{
	if( !( expr ) ) {
		CheckOnnxProtocol( false, GetMessageWithOperatorInfo( what, op ) );
	}
}

// Throws std::logic_error if 'expr' is false
// Used when expr can't be emulated by NeoOnnx despite being valid case of onnx
// e.g. 8+ dimensional tensors which aren't supported by NeoML
inline void CheckNeoOnnxSupport( bool expr, const CString& what ) 
{
	if( !( expr ) ) {
		NeoOnnxCheck( false, CString( "Not supported by NeoOnnx: " ) + what );
	}
}

inline void CheckNeoOnnxSupport( bool expr, const CString& what, const COperator& op )
{
	if( !( expr ) ) {
		CheckNeoOnnxSupport( false, GetMessageWithOperatorInfo( what, op ) );
	}
}

} // namespace NeoOnnx
