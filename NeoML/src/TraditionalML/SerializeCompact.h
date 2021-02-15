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

#include <NeoML/NeoMLDefs.h>

namespace NeoML {

/////////////////////////////////////////////////////////////////////////////////////////

// Serializes an integer depending on its value (similar to the UTF8 encoding)
template<typename TValue>
inline void SerializeCompact( CArchive& archive, TValue& value )
{
	const int UsefulBitCount = 7;
	const TValue UsefulBitMask = (static_cast<TValue>(1) << UsefulBitCount) - 1;
	const int ContinueBitMask = 1 << UsefulBitCount;

	if( archive.IsStoring() ) {
		NeoAssert( value >= 0 );
		TValue temp = value;
		do {
			const TValue div = temp >> UsefulBitCount;
			const TValue mod = temp & UsefulBitMask;
			archive << static_cast<unsigned char>( mod | (div != 0 ? ContinueBitMask : 0) );
			temp = div;
		} while( temp != 0 );
	} else if( archive.IsLoading() ) {
		TValue temp = 0;
		int shift = 0;
		unsigned char mod = 0;
		do {
			archive >> mod;
			temp = static_cast<TValue>( (mod & UsefulBitMask) << shift ) | temp;
			shift += UsefulBitCount;
		} while( (mod & ContinueBitMask) != 0 );
		value = temp;
	} else {
		NeoAssert( false );
	}
}

// Serializes `float` similar to the UTF8 encoding.
// Makes sense when integers stored as float are common.
// Packing order is MSB to LSB, i.e. opposite to integer case.
inline void SerializeCompact( CArchive& archive, float& value )
{
	union {
		float f;
		uint32_t i = 0;
	} temp;
	static_assert( sizeof temp.i == sizeof temp.f, "Unexpected float size" );

	const int TotalBitCount = CHAR_BIT * sizeof( temp.i );
	const int UsefulBitCount = 7;
	const uint32_t UsefulBitMask = (static_cast<uint32_t>(1) << UsefulBitCount) - 1;
	const int ContinueBitMask = 1 << UsefulBitCount;

	if( archive.IsStoring() ) {
		temp.f = value;
		do {
			const uint32_t div = temp.i << UsefulBitCount;
			const uint32_t mod = temp.i >> (TotalBitCount - UsefulBitCount);
			archive << static_cast<unsigned char>( mod | (div != 0 ? ContinueBitMask : 0) );
			temp.i = div;
		} while( temp.i != 0 );
	} else if( archive.IsLoading() ) {
		temp.i = 0;
		int shift = TotalBitCount - UsefulBitCount;
		unsigned char mod = 0;
		do {
			archive >> mod;
			if( shift >= 0 ) {
				temp.i = ((mod & UsefulBitMask) << shift) | temp.i;
			} else {
				temp.i = ((mod & UsefulBitMask) >> (-shift)) | temp.i;
			}
			shift -= UsefulBitCount;
		} while( (mod & ContinueBitMask) != 0 );
		value = temp.f;
	} else {
		NeoAssert( false );
	}
}

/////////////////////////////////////////////////////////////////////////////////////////

} // namespace NeoML
