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

#include <FineObjLiteDefs.h>

#include <AllocFOL.h>
#include <ArchiveFOL.h>
#include <ArrayFOL.h>
#include <ArrayIteratorFOL.h>
#include <AscendingFOL.h>
#include <BaseFileFOL.h>
#include <CriticalSectionFOL.h>
#include <DescendingFOL.h>
#include <DynamicBitSetFOL.h>
#include <ErrorsFOL.h>
#include <FastArrayFOL.h>
#include <HashTableAllocatorFOL.h>
#include <HashTableFOL.h>
#include <HashTableIteratorFOL.h>
#include <IntervalFOL.h>
#include <ObjectFOL.h>
#include <MapFOL.h>
#include <MapIteratorFOL.h>
#include <MapPositionIteratorFOL.h>
#include <MathFOL.h>
#include <MemoryFileFOL.h>
#include <PointerArrayFOL.h>
#include <PriorityQueueFOL.h>
#include <PtrOwnerFOL.h>
#include <SortFOL.h>
#include <StringFOL.h>
#include <TextStreamFOL.h>

namespace FObj {

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
	return JoinStrings( strings, delimiter.Ptr() );
}

inline CString JoinStrings( const CArray<CString>& strings )
{
	return JoinStrings( strings, GetDefaultDelimiter( char{} ) );
}

template<>
struct IsMemmoveable<CString> {
	static const bool Value = false;
};

} // namespace FObj

using namespace FObj;
