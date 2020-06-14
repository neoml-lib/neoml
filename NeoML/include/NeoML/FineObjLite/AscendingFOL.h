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

namespace FObj {

// The class to sort ascending, by the elements values
template<class T>
class Ascending {
public:
	bool Predicate( const T& first, const T& second ) const { return first < second; }
	bool IsEqual( const T& first, const T& second ) const { return first == second; }
	void Swap( T& first, T& second ) const { std::swap<T>( first, second ); }
};

// The class to sort ascending by the Member field, of the TMemberType type
// Two additional variations of the Predicate and IsEqual methods are needed for searching by the TMemberType value
template<class T, class TMemberType, TMemberType T::*Member>
class AscendingByMember {
public:
	bool Predicate( const T& first, const T& second ) const { return ( first.*Member ) < ( second.*Member ); }
	bool Predicate( const TMemberType& first, const T& second ) const { return first < ( second.*Member ); }
	bool Predicate( const T& first, const TMemberType& second ) const { return ( first.*Member ) < second; }
	bool IsEqual( const T& first, const T& second ) const { return ( first.*Member ) == ( second.*Member ); }
	bool IsEqual( const TMemberType& first, const T& second ) const { return first == ( second.*Member ); }
	bool IsEqual( const T& first, const TMemberType& second ) const { return ( first.*Member ) == second; }
	void Swap( T& first, T& second ) const { std::swap<T>( first, second ); }
};

// The class to sort ascending by the Method value, of the TMethodReturnType Method() const signature
// Two additional variations of the Predicate and IsEqual methods are needed for searching by the TMethodReturnType value
template<class T, class TMethodReturnType, TMethodReturnType ( T::*Method )() const>
class AscendingByMethod {
public:
	bool Predicate( const T& first, const T& second ) const { return ( first.*Method )() < ( second.*Method )(); }
	bool Predicate( const TMethodReturnType& first, const T& second ) const { return first < ( second.*Method )(); }
	bool Predicate( const T& first, const TMethodReturnType& second ) const { return ( first.*Method )() < second; }
	bool IsEqual( const T& first, const T& second ) const { return ( first.*Method )() == ( second.*Method )(); }
	bool IsEqual( const TMethodReturnType& first, const T& second ) const { return first == ( second.*Method )(); }
	bool IsEqual( const T& first, const TMethodReturnType& second ) const { return ( first.*Method )() == second; }
	void Swap( T& first, T& second ) const { std::swap<T>( first, second ); }
};

// A special version for constant references
template<class T, class TMethodReturnType, const TMethodReturnType& ( T::*Method )() const>
class AscendingByMethod<T, const TMethodReturnType&, Method> {
public:
	bool Predicate( const T& first, const T& second ) const { return ( first.*Method )() < ( second.*Method )(); }
	bool Predicate( const TMethodReturnType& first, const T& second ) const { return first < ( second.*Method )(); }
	bool Predicate( const T& first, const TMethodReturnType& second ) const { return ( first.*Method )() < second; }
	bool IsEqual( const T& first, const T& second ) const { return ( first.*Method )() == ( second.*Method )(); }
	bool IsEqual( const TMethodReturnType& first, const T& second ) const { return first == ( second.*Method )(); }
	bool IsEqual( const T& first, const TMethodReturnType& second ) const { return ( first.*Method )() == second; }
	void Swap( T& first, T& second ) const { std::swap<T>( first, second ); }
};

// The class to sort ascending an array of pointers, by comparing the objects the pointers refer to
template<class T>
class AscendingPtr {
public:
	bool Predicate( const T* first, const T* second ) const { return *first < *second; }
	bool IsEqual( const T* first, const T* second ) const { return *first == *second; }
	void Swap( T*& first, T*& second ) const { swap( first, second ); }
	void Swap( CPtr<T>& first, CPtr<T>& second ) const { std::swap( first, second ); }
};

// The class to sort ascending an array of pointers by comparing the objects' Member field, of the TMemberType type
// Two additional variations of the Predicate and IsEqual methods are needed for searching by the TMemberType value
template<class T, class TMemberType, TMemberType T::*Member>
class AscendingPtrByMember {
public:
	bool Predicate( const T* first, const T* second ) const { return ( first->*Member ) < ( second->*Member ); }
	bool Predicate( const TMemberType& first, const T* second ) const { return first < ( second->*Member ); }
	bool Predicate( const T* first, const TMemberType& second ) const { return ( first->*Member ) < second; }
	bool IsEqual( const T* first, const T* second ) const { return ( first->*Member ) == ( second->*Member ); }
	bool IsEqual( const TMemberType& first, const T* second ) const { return first == ( second->*Member ); }
	bool IsEqual( const T* first, const TMemberType& second ) const { return ( first->*Member ) == second; }
	void Swap( T*& first, T*& second ) const { swap( first, second ); }
	void Swap( CPtr<T>& first, CPtr<T>& second ) const { std::swap( first, second ); }
};

// The class to sort ascending an array of pointers by the Method value, of the TMethodReturnType Method() const signature
// Two additional variations of the Predicate and IsEqual methods are needed for searching by the TMethodReturnType value
template<class T, class TMethodReturnType, TMethodReturnType ( T::*Method )() const>
class AscendingPtrByMethod {
public:
	bool Predicate( const T* first, const T* second ) const { return ( first->*Method )() < ( second->*Method )(); }
	bool Predicate( const TMethodReturnType& first, const T* second ) const { return first < ( second->*Method )(); }
	bool Predicate( const T* first, const TMethodReturnType& second ) const { return ( first->*Method )() < second; }
	bool IsEqual( const T* first, const T* second ) const { return ( first->*Method )() == ( second->*Method )(); }
	bool IsEqual( const TMethodReturnType& first, const T* second ) const { return first == ( second->*Method )(); }
	bool IsEqual( const T* first, const TMethodReturnType& second ) const { return ( first->*Method )() == second; }
	void Swap( T*& first, T*& second ) const { swap( first, second ); }
	void Swap( CPtr<T>& first, CPtr<T>& second ) const { std::swap( first, second ); }
};

// A special version for constant references
template<class T, class TMethodReturnType, const TMethodReturnType& ( T::*Method )() const>
class AscendingPtrByMethod<T, const TMethodReturnType&, Method> {
public:
	bool Predicate( const T* first, const T* second ) const { return ( first->*Method )() < ( second->*Method )(); }
	bool Predicate( const TMethodReturnType& first, const T* second ) const { return first < ( second->*Method )(); }
	bool Predicate( const T* first, const TMethodReturnType& second ) const { return ( first->*Method )() < second; }
	bool IsEqual( const T* first, const T* second ) const { return ( first->*Method )() == ( second->*Method )(); }
	bool IsEqual( const TMethodReturnType& first, const T* second ) const { return first == ( second->*Method )(); }
	bool IsEqual( const T* first, const TMethodReturnType& second ) const { return ( first->*Method )() == second; }
	void Swap( T*& first, T*& second ) const { swap( first, second ); }
	void Swap( CPtr<T>& first, CPtr<T>& second ) const { std::swap( first, second ); }
};

// Composite sorting
// Sorts by the first comparer. The elements that are the same are then sorted by the second comparer
template<class T, class COMPARE1, class COMPARE2>
class CompositeComparer {
public:
	bool Predicate( const T& first, const T& second ) const;
	bool IsEqual( const T& first, const T& second ) const;
	void Swap( T& first, T& second ) const { std::swap<T>( first, second ); }

private:
	COMPARE1 comparer1;
	COMPARE2 comparer2;
};

template<class T, class COMPARE1, class COMPARE2>
inline bool CompositeComparer<T, COMPARE1, COMPARE2>::Predicate( const T& first, const T& second ) const
{
	if( comparer1.IsEqual(first, second) ) {
		return comparer2.Predicate(first, second);
	}

	return comparer1.Predicate(first, second);
}

template<class T, class COMPARE1, class COMPARE2>
inline bool CompositeComparer<T, COMPARE1, COMPARE2>::IsEqual( const T& first, const T& second ) const
{
	return comparer1.IsEqual(first, second) && comparer2.IsEqual(first, second);
}

} // namespace FObj
