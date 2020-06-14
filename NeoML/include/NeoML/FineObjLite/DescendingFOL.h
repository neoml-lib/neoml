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

// The class to sort descending, by the elements values
template<class T>
class Descending {
public:
	bool Predicate( const T& first, const T& second ) const { return second < first; }
	bool IsEqual( const T& first, const T& second ) const { return first == second; }
	void Swap( T& first, T& second ) const { std::swap<T>( first, second ); }
};

// The class to sort descending by the Member field, of the TMemberType type
// Two additional variations of the Predicate and IsEqual methods are needed for searching by the TMemberType value
template<class T, class TMemberType, TMemberType T::*Member>
class DescendingByMember {
public:
	bool Predicate( const T& first, const T& second ) const { return ( second.*Member ) < ( first.*Member ); }
	bool Predicate( const TMemberType& first, const T& second ) const { return ( second.*Member ) < first; }
	bool Predicate( const T& first, const TMemberType& second ) const { return second < ( first.*Member ); }
	bool IsEqual( const T& first, const T& second ) const { return ( first.*Member ) == ( second.*Member ); }
	bool IsEqual( const TMemberType& first, const T& second ) const { return first == ( second.*Member ); }
	bool IsEqual( const T& first, const TMemberType& second ) const { return ( first.*Member ) == second; }
	void Swap( T& first, T& second ) const { std::swap( first, second ); }
};

// The class to sort descending by the Method value, of the TMethodReturnType Method() const signature
// Two additional variations of the Predicate and IsEqual methods are needed for searching by the TMethodReturnTyp value
template<class T, class TMethodReturnType, TMethodReturnType ( T::*Method )() const>
class DescendingByMethod {
public:
	bool Predicate( const T& first, const T& second ) const { return ( second.*Method )() < ( first.*Method )(); }
	bool Predicate( const TMethodReturnType& first, const T& second ) const { return ( second.*Method )() < first; }
	bool Predicate( const T& first, const TMethodReturnType& second ) const { return second < ( first.*Method )(); }
	bool IsEqual( const T& first, const T& second ) const { return ( first.*Method )() == ( second.*Method )(); }
	bool IsEqual( const TMethodReturnType& first, const T& second ) const { return first == ( second.*Method )(); }
	bool IsEqual( const T& first, const TMethodReturnType& second ) const { return ( first.*Method )() == second; }
	void Swap( T& first, T& second ) const { std::swap( first, second ); }
};

// A special version for constant references
template<class T, class TMethodReturnType, const TMethodReturnType& ( T::*Method )() const>
class DescendingByMethod<T, const TMethodReturnType&, Method> {
public:
	bool Predicate( const T& first, const T& second ) const { return ( second.*Method )() < ( first.*Method )(); }
	bool Predicate( const TMethodReturnType& first, const T& second ) const { return ( second.*Method )() < first; }
	bool Predicate( const T& first, const TMethodReturnType& second ) const { return second < ( first.*Method )(); }
	bool IsEqual( const T& first, const T& second ) const { return ( first.*Method )() == ( second.*Method )(); }
	bool IsEqual( const TMethodReturnType& first, const T& second ) const { return first == ( second.*Method )(); }
	bool IsEqual( const T& first, const TMethodReturnType& second ) const { return ( first.*Method )() == second; }
	void Swap( T& first, T& second ) const { std::swap( first, second ); }
};

// The class to sort descending an array of pointers, by comparing the objects the pointers refer to
template<class T>
class DescendingPtr {
public:
	bool Predicate( const T* first, const T* second ) const { return *second < *first; }
	bool IsEqual( const T* first, const T* second ) const { return *first == *second; }
	void Swap( T*& first, T*& second ) const { swap( first, second ); }
	void Swap( CPtr<T>& first, CPtr<T>& second ) const { std::swap( first, second ); }
};

// The class to sort descending an array of pointers by comparing the objects' Member field, of the TMemberType type
// Two additional variations of the Predicate and IsEqual methods are needed for searching by the TMemberType value
template<class T, class TMemberType, TMemberType T::*Member>
class DescendingPtrByMember {
public:
	bool Predicate( const T* first, const T* second ) const { return ( second->*Member ) < ( first->*Member ); }
	bool Predicate( const TMemberType& first, const T* second ) const { return ( second->*Member ) < first; }
	bool Predicate( const T* first, const TMemberType& second ) const { return second < ( first->*Member ); }
	bool IsEqual( const T* first, const T* second ) const { return ( first->*Member ) == ( second->*Member ); }
	bool IsEqual( const TMemberType& first, const T* second ) const { return first == ( second->*Member ); }
	bool IsEqual( const T* first, const TMemberType& second ) const { return ( first->*Member ) == second; }
	void Swap( T*& first, T*& second ) const { swap( first, second ); }
	void Swap( CPtr<T>& first, CPtr<T>& second ) const { std::swap( first, second ); }
};

// The class to sort descending an array of pointers by the Method value, of the TMethodReturnType Method() const signature
// Two additional variations of the Predicate and IsEqual methods are needed for searching by the TMethodReturnType value
template<class T, class TMethodReturnType, TMethodReturnType ( T::*Method )() const>
class DescendingPtrByMethod {
public:
	bool Predicate( const T* first, const T* second ) const { return ( second->*Method )() < ( first->*Method )(); }
	bool Predicate( const TMethodReturnType& first, const T* second ) const { return ( second->*Method )() < first; }
	bool Predicate( const T* first, const TMethodReturnType& second ) const { return second < ( first->*Method )(); }
	bool IsEqual( const T* first, const T* second ) const { return ( first->*Method )() == ( second->*Method )(); }
	bool IsEqual( const TMethodReturnType& first, const T* second ) const { return first == ( second->*Method )(); }
	bool IsEqual( const T* first, const TMethodReturnType& second ) const { return ( first->*Method )() == second; }
	void Swap( T*& first, T*& second ) const { swap( first, second ); }
	void Swap( CPtr<T>& first, CPtr<T>& second ) const { std::swap( first, second ); }
};

// A special version for constant references
template<class T, class TMethodReturnType, const TMethodReturnType& ( T::*Method )() const>
class DescendingPtrByMethod<T, const TMethodReturnType&, Method> {
public:
	bool Predicate( const T* first, const T* second ) const { return ( second->*Method )() < ( first->*Method )(); }
	bool Predicate( const TMethodReturnType& first, const T* second ) const { return ( second->*Method )() < first; }
	bool Predicate( const T* first, const TMethodReturnType& second ) const { return second < ( first->*Method )(); }
	bool IsEqual( const T* first, const T* second ) const { return ( first->*Method )() == ( second->*Method )(); }
	bool IsEqual( const TMethodReturnType& first, const T* second ) const { return first == ( second->*Method )(); }
	bool IsEqual( const T* first, const TMethodReturnType& second ) const { return ( first->*Method )() == second; }
	void Swap( T*& first, T*& second ) const { swap( first, second ); }
	void Swap( CPtr<T>& first, CPtr<T>& second ) const { std::swap( first, second ); }
};

} // namespace FObj

