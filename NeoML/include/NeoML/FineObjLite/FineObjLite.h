/* Copyright © 2017-2024 ABBYY

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

#include "FineObjLiteDefs.h"
#include "StringFOL.h"
#include "ErrorsFOL.h"
#include "MathFOL.h"
#include "AllocFOL.h"
#include "BaseFileFOL.h"
#include "ArchiveFOL.h"
#include "MemoryFileFOL.h"
#include "internal/FastArrayFOL.h"
#include "ObjectFOL.h"
#include "ArrayFOL.h"
#include "ArrayIteratorFOL.h"
#include "internal/PointerArrayFOL.h"
#include "internal/HashTableAllocatorFOL.h"
#include "internal/HashTableFOL.h"
#include "internal/MapFOL.h"
#include "TextStreamFOL.h"
#include "internal/PriorityQueueFOL.h"
#include "AscendingFOL.h"
#include "DescendingFOL.h"
#include "internal/DynamicBitSetFOL.h"
#include "internal/IntervalFOL.h"
#include "SortFOL.h"
#include "internal/CriticalSectionFOL.h"
#include "PtrOwnerFOL.h"

namespace FObj {

using std::swap;

constexpr const char* GetDefaultDelimiter( char ) { return ", "; }

inline CString JoinStrings( const CArray<CString>& strings, const char* delimiter )
{
	if( strings.IsEmpty() ) {
		return CString{};
	}

	// reserve space
	size_t length = strlen( delimiter ) * ( strings.Size() - 1 );
	for( int i = 0; i < strings.Size(); i++ ) {
		length += strings[i].Length();
	}

	CString result;
	result.SetBufferLength( static_cast<int>( length ) );

	result += strings[0];
	for( int i = 1; i < strings.Size(); ++i ) {
		result += delimiter + strings[i];
	}
	return result;
}

inline CString JoinStrings( const CArray<CString>& strings, const CString& delimiter )
{
	return JoinStrings( strings, delimiter.data() );
}

inline CString JoinStrings( const CArray<CString>& strings )
{
	return JoinStrings( strings, GetDefaultDelimiter( char{} ) );
}

} // namespace FObj

using namespace FObj;
