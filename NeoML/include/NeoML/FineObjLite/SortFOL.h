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

namespace FObj {

// The sorting function template that depends on the array data type 
// and the comparer class with the comparison and swap functions
// The comparison function (Predicate) returns a boolean type
// and should return false if its arguments have to be swapped to get the correct sorting order
// When searching the first parameter is always the element to look for
// Predicate == true for the found element and false for the previous one
// The library has standard comparer implementations that use the < and = operators (see AscendingFOL.h, DescendingFOL.h)

// Для массивов меньшего размера эффективней использовать сортировку вставками
const int QuickSortCutoff = 8;

//----------------------------------------------------------------------------------------------------------------
// Вспомогательные функции
template<class TYPE, class COMPARE>
int divideArray( TYPE* arr, int size, COMPARE* param );

template<class TYPE, class COMPARE>
void doQuickSort( TYPE* arr, int size, COMPARE* param );

template<class TYPE, class COMPARE, class SEARCHED_TYPE>
int doFindInsertionPoint( const SEARCHED_TYPE& element, const TYPE* arr, int size, COMPARE* param );

template<class TYPE, class COMPARE>
bool isSorted( const TYPE* arr, int size, COMPARE* param );

template<class TYPE, class COMPARE>
inline void QuickSort( TYPE* arr, int size, COMPARE* param )
{
	AssertFO( std::is_empty<COMPARE>::value || param != nullptr || size <= 1 );
	doQuickSort( arr, size, param );
}

template<class TYPE, class COMPARE>
inline void QuickSort( TYPE* arr, int size )
{
	COMPARE cmp;
	doQuickSort( arr, size, &cmp );
}

template<class TYPE, class COMPARE>
inline bool IsSorted( const TYPE* arr, int size, COMPARE* compare )
{
	AssertFO( std::is_empty<COMPARE>::value || compare != nullptr || size <= 1 );
	return isSorted<TYPE, COMPARE>( arr, size, compare );
}

template<class TYPE, class COMPARE>
inline bool IsSorted( const TYPE* arr, int size )
{
	COMPARE compare;
	return isSorted<TYPE, COMPARE>( arr, size, &compare );
}

// Binary searches a sorted array
// and returns the position where the element should be inserted
// The array element type and the type of the object passed to the search function may have different types
// In this case the comparer class should have the IsEqual and Predicate methods 
// that accept a parameter of this object type (see AscendingByMember for an example)
template<class TYPE, class COMPARE, class SEARCHED_TYPE>
inline int FindInsertionPoint( const SEARCHED_TYPE& element, const TYPE* arr, int size, COMPARE* param )
{
	AssertFO( std::is_empty<COMPARE>::value || param != nullptr || size <= 1 );
	return doFindInsertionPoint( element, arr, size, param );
}

template<class TYPE, class COMPARE, class SEARCHED_TYPE>
inline int FindInsertionPoint( const SEARCHED_TYPE& element, const TYPE* arr, int size )
{
	COMPARE cmp;
	return doFindInsertionPoint( element, arr, size, &cmp );
}

// Binary searches a sorted array and returns the position where the element should be inserted
template<class TYPE>
int FindInsPoint( const TYPE& element, const TYPE* arr, int size, int ( *compareFunc )( const TYPE*, const TYPE* ) )
{
	int first = 0;
	int last = size;
	while( first < last ) {
		int mid = ( first + last ) / 2;
		if( compareFunc( AddressOfObject( element ), arr + mid ) < 0 ) {
			last = mid;
		} else {
			first = mid + 1;
		}
	}
	return first;
}

template<class TYPE, class COMPARE>
bool isSorted( const TYPE* arr, int size, COMPARE* compare )
{
	for( int i = 0; i < size - 1; i++ ) {
		if( !compare->Predicate( arr[i], arr[i + 1] ) && !compare->IsEqual( arr[i], arr[i + 1] ) ) {
			return false;
		}
	}
	return true;
}

template<class TYPE, class COMPARE>
inline void sort2( TYPE& a, TYPE& b, COMPARE* param )
{
	if( param->Predicate( b, a ) ) {
		param->Swap( a, b );
	}
}

template<class TYPE, class COMPARE>
inline void sort3( TYPE& a, TYPE& b, TYPE& c, COMPARE* param )
{
	const bool ab = param->Predicate( b, a );
	const bool bc = param->Predicate( c, b );
	if( ab && bc ) {
		param->Swap( a, c );
	} else if( ab ) {
		param->Swap( a, b );
		sort2( b, c, param );
	} else if( bc ) {
		param->Swap( b, c );
		sort2( a, b, param );
	}
}

template<class TYPE, class COMPARE>
void InsertionSort( TYPE* arr, int size, COMPARE* param )
{
	AssertFO( std::is_empty<COMPARE>::value || param != nullptr || size <= 1 );
	if( size == 2 ) {
		sort2( arr[0], arr[1], param );
	} else if( size == 3 ) {
		sort3( arr[0], arr[1], arr[2], param );
	} else {
		for( int i = size - 1; i > 0; i-- ) {
			int best = i;
			for( int j = i - 1; j >= 0; j-- ) {
				if( param->Predicate( arr[best], arr[j] ) ) {
					best = j;
				}
			}

			if( i != best ) {
				param->Swap( arr[best], arr[i] );
			}
		}
	}
}

const int QsortStackSize = CHAR_BIT * sizeof( void* );

template<class TYPE, class COMPARE>
void doQuickSort( TYPE* arr, int size, COMPARE* param )
{
	if( size < 2 ) {
		return;
	}

	TYPE* arrayStack[QsortStackSize];
	int sizeStack[QsortStackSize];
	int stackPtr = 0;

	while( true ) {
		if( size <= QuickSortCutoff ) {
			InsertionSort( arr, size, param );
		} else {
			int mean = divideArray( arr, size, param );
			int rightSize = size - mean - 1;
			if( mean < rightSize ) {
				if( rightSize >= 2 ) {
					arrayStack[stackPtr] = arr + mean + 1;
					sizeStack[stackPtr] = rightSize;
					stackPtr++;
				}
				if( mean >= 2 ) {
					size = mean;
					continue;
				}
			} else {
				if( mean >= 2 ) {
					arrayStack[stackPtr] = arr;
					sizeStack[stackPtr] = mean;
					stackPtr++;
				}
				if( rightSize >= 2 ) {
					arr += mean + 1;
					size = rightSize;
					continue;
				}
			}
		}
		if( stackPtr == 0 ) {
			break;
		}
		stackPtr--;
		arr = arrayStack[stackPtr];
		size = sizeStack[stackPtr];
	}
}

template<class TYPE, class COMPARE>
int divideArray( TYPE* arr, int size, COMPARE* param )
{
	PresumeFO( size >= 3 );

	const int mid = size / 2;
	sort3( arr[0], arr[mid], arr[size-1], param );

	int indicatorPlace = 1;
	int currentGreaterPlace = size - 1;
	for( ; indicatorPlace < mid && param->Predicate( arr[indicatorPlace], arr[mid] ); ++indicatorPlace );
	if( indicatorPlace < mid ) {
		param->Swap( arr[indicatorPlace], arr[mid] );
		do {
			--currentGreaterPlace;
		} while( currentGreaterPlace > mid && param->Predicate( arr[indicatorPlace], arr[currentGreaterPlace] ) );
		if( currentGreaterPlace > mid ) {
			param->Swap( arr[mid], arr[currentGreaterPlace] );
		}
	}
	int currentLessPlace = indicatorPlace;
	while( true ) {
		do {
			currentLessPlace++;
		} while( currentLessPlace < currentGreaterPlace &&
			param->Predicate( arr[currentLessPlace], arr[indicatorPlace] ) );
		do {
			currentGreaterPlace--;
		} while( currentGreaterPlace >= currentLessPlace &&
			param->Predicate( arr[indicatorPlace], arr[currentGreaterPlace] ) );

		if( currentGreaterPlace < currentLessPlace ) {
			break;
		}
		PresumeFO( !param->Predicate( arr[currentLessPlace], arr[currentGreaterPlace] ) );
		param->Swap( arr[currentLessPlace], arr[currentGreaterPlace] );
	}

	if( indicatorPlace != currentGreaterPlace && param->Predicate( arr[currentGreaterPlace], arr[indicatorPlace] ) ) {
		param->Swap( arr[indicatorPlace], arr[currentGreaterPlace] );
	}
	return currentGreaterPlace;
}

template<class TYPE, class COMPARE, class SEARCHED_TYPE>
int doFindInsertionPoint( const SEARCHED_TYPE& element, const TYPE* arr, int size, COMPARE* param )
{
	int first = 0;
	int last = size;
	while( first < last ) {
		int mid = first + ( last - first ) / 2;
		PresumeFO( param->Predicate( arr[first], arr[mid] ) || param->IsEqual( arr[first], arr[mid] ) );
		PresumeFO( param->Predicate( arr[mid], arr[last - 1] ) || param->IsEqual( arr[mid], arr[last - 1] ) );
		if( param->Predicate( element, arr[mid] ) ) {
			last = mid;
			// last < size
			PresumeFO( last == 0 ||
				param->Predicate( arr[last - 1], arr[last] ) || param->IsEqual( arr[last - 1], arr[last] ) );
		} else {
			first = mid + 1;
			// first >= 0
			PresumeFO( first == size ||
				param->Predicate( arr[first - 1], arr[first] ) || param->IsEqual( arr[first - 1], arr[first] ) );
		}
	}
	return first;
}

} // namespace FObj
