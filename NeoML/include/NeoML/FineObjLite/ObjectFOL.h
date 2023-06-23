/* Copyright © 2017-2023 ABBYY

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

// The base class for all interfaces with reference counting
// They should virtually inherit from this class and use CPtr for all pointers
class NEOML_API IObject {
public:
	int RefCount() const;

	virtual void Serialize( CArchive& ) { AssertFO( false ); }

	void ReadFromArchive( CArchive& archive );
	void WriteToArchive( CArchive& archive ) const;

protected:
	IObject();
	virtual ~IObject();

private:
	mutable std::atomic_int refCounter;

	IObject( const IObject& );
	IObject& operator=( const IObject& );
	
	void addRef() const;
	void release() const;
	void detach();
	bool weakAddRef() const;

	template<class T>
	friend class CPtr;
	
	template<class T>
	friend class CCopyOnWritePtr;
};

//---------------------------------------------------------------------------------------------

// Smart pointer template for the interfaces that inherit from IObject
template<class T>
class CPtr final {
public:
	CPtr() : ptr( nullptr ) {}
	CPtr( T* other ) : ptr( other ) { if( ptr != nullptr ) { ptr->addRef(); } }
	CPtr( const CPtr<T>& other ) : CPtr( other.Ptr() ) {}
	CPtr( CPtr<T>&& other ) { Swap( other ); }
	~CPtr() { Release(); }

	void Release();

	// Binds a weak pointer to the object
	// Returns true if the pointer has been successfully bound
	// and false otherwise (because weakPtr is 0 or the object is already being destroyed)
	bool PinWeakPtr( T* weakPtr );

	int HashKey() const { return static_cast<int>( reinterpret_cast<UINT_PTR>( ptr ) ); }

	T& operator*() const;
	T* operator->() const;
	operator T*() const { return ptr; }
	T* Ptr() const { return ptr; }

	const CPtr<T>& operator=( T* other ) { return assignPtr( other ); }
	const CPtr<T>& operator=( const CPtr<T>& other ) { return assignPtr( other ); }
	const CPtr<T>& operator=( CPtr<T>&& );

	void Swap( CPtr<T>& other ) { std::swap( ptr, other.ptr ); }

private:
	T* ptr = nullptr;

	const CPtr<T>& assignPtr( T* );
	void replacePtr( T* );

	template<class U>
	friend class CPtr;
};

//------------------------------------------------------------------------------------------------------------

// Const specialization
// Common used construction of implicit cast from CPtr<T> to CPtr<const T>
// (full copy of class CPtr<T>)

template<class T>
class CPtr<const T> final {
public:
	CPtr() : ptr( nullptr ) {}
	CPtr( const T* other ) : ptr( other ) { if( ptr != nullptr ) { ptr->addRef(); } }
	CPtr( const CPtr<const T>& other ) : CPtr( other.Ptr() )  {}
	CPtr( CPtr<const T>&& other ) { Swap( other ); }
	~CPtr() { Release(); }

	CPtr( const CPtr<T>& other ) : CPtr( other.Ptr() ) {} // <-- this main ctor added in specialization.
	CPtr( CPtr<T>&& );

	void Release();

	// Binds a weak pointer to the object
	// Returns true, if the pointer has been successfully bound
	// and false otherwise (because weakPtr equals 0 or the object is already being destroyed).
	bool PinWeakPtr( const T* weakPtr );

	int HashKey() const { return static_cast<int>( reinterpret_cast<UINT_PTR>( ptr ) ); }

	const T& operator*() const;
	const T* operator->() const;
	const T* Ptr() const { return ptr; }
	operator const T*() const { return ptr; }

	const CPtr<const T>& operator=( const T* other ) { return assignPtr( other ); }
	const CPtr<const T>& operator=( const CPtr<const T>& other ) { return assignPtr( other.Ptr() ); }
	const CPtr<const T>& operator=( CPtr<const T>&& );

	const CPtr<const T>& operator=( const CPtr<T>& other ) { return assignPtr( other.Ptr() ); } // <-- this main operator= added in specialization.
	const CPtr<const T>& operator=( CPtr<T>&& other );

	void Swap( CPtr<const T>& other ) { std::swap( ptr, other.ptr ); }

private:
	const T* ptr = nullptr;

	const CPtr<const T>& assignPtr( const T* );
	void replacePtr( const T* );
};

template<class T>
inline CPtr<const T>::CPtr( CPtr<T>&& other )
{
	ptr = other.ptr;
	other.ptr = nullptr;
}

template<class T>
inline void CPtr<const T>::Release()
{
	const T* tmp = ptr;
	if( tmp != nullptr ) {
		ptr = nullptr;
		tmp->release();
	}
}

template<class T>
inline const T& CPtr<const T>::operator*() const
{
	AssertFO( ptr != nullptr );
	return *ptr;
}

template<class T>
inline const T* CPtr<const T>::operator->() const
{
	AssertFO( ptr != nullptr );
	return ptr;
}

template<class T>
inline const CPtr<const T>& CPtr<const T>::operator=( CPtr<const T>&& other )
{
	Swap( other );
	return *this;
}

template<class T>
inline const CPtr<const T>& CPtr<const T>::operator=( CPtr<T>&& other )
{
	replacePtr( other.ptr );
	other.ptr = nullptr;
	return *this;
}

template<class T>
inline bool CPtr<const T>::PinWeakPtr( const T* weakPtr )
{
	if( weakPtr == nullptr || !weakPtr->weakAddRef() ) {
		return false;
	}
	replacePtr( weakPtr );
	return true;
}

template<class T>
inline const CPtr<const T>& CPtr<const T>::assignPtr( const T* other )
{
	if( other != nullptr ) {
		other->addRef();
	}
	replacePtr( other );
	return *this;
}

template<class T>
inline void CPtr<const T>::replacePtr( const T* other )
{
	const T* tmp = ptr;
	ptr = other;
	if( tmp != nullptr ) {
		tmp->release();
	}
}

//---------------------------------------------------------------------------------------------

template<typename TDest, typename TSrc>
inline TDest* CheckCast( TSrc* ptr )
{
	TDest* ret = dynamic_cast<TDest*>( ptr );
	AssertFO( ret != 0 );
	return ret;
}

template<typename TDest, typename TSrc>
inline const TDest* CheckCast( const TSrc* ptr )
{
	const TDest* ret = dynamic_cast<const TDest*>( ptr );
	AssertFO( ret != 0 );
	return ret;
}

template<typename TDest>
inline TDest* CheckCast( IObject* ptr )
{
	return CheckCast<TDest, IObject>( ptr );
}

template<typename TDest>
inline const TDest* CheckCast( const IObject* ptr )
{
	return CheckCast<TDest, IObject>( ptr );
}

//---------------------------------------------------------------------------------------------

inline IObject::IObject() :
	refCounter( 0 )
{}

inline int IObject::RefCount() const
{
	return refCounter;
}

inline void IObject::addRef() const
{
	refCounter++;
}

inline void IObject::release() const
{
	if( refCounter.fetch_sub( 1 ) == 1 ) {
		delete this;
	}
}

inline void IObject::detach()
{
	refCounter.exchange( 0 );
}

inline bool IObject::weakAddRef() const
{
	while(1) {
		int curValue = refCounter;
		if( curValue <= 0 ) {
			return false;
		}
		if( refCounter.compare_exchange_weak( curValue, curValue + 1 ) ) {
			return true;
		}
	}
	return false;
}

//---------------------------------------------------------------------------------------------

template<class T>
inline void CPtr<T>::Release()
{
	T* tmp = ptr;
	if( tmp != nullptr ) {
		ptr = nullptr;
		tmp->release();
	}
}

template<class T>
inline T& CPtr<T>::operator*() const
{
	AssertFO( ptr != nullptr );
	return *ptr;
}

template<class T>
inline T* CPtr<T>::operator->() const
{
	AssertFO( ptr != nullptr );
	return ptr;
}

template<class T>
inline const CPtr<T>& CPtr<T>::operator=( CPtr<T>&& other )
{
	Swap( other );
	return *this;
}

template<class T>
inline bool CPtr<T>::PinWeakPtr( T* weakPtr )
{
	if( weakPtr == nullptr || !weakPtr->weakAddRef() ) {
		return false;
	}
	replacePtr( weakPtr );
	return true;
}

template<class T>
inline const CPtr<T>& CPtr<T>::assignPtr( T* other )
{
	if( other != nullptr ) {
		other->addRef();
	}
	replacePtr( other );
	return *this;
}

template<class T>
inline void CPtr<T>::replacePtr( T* other )
{
	T* tmp = ptr;
	ptr = other;
	if( tmp != nullptr ) {
		tmp->release();
	}
}

//---------------------------------------------------------------------------------------------

template<class T>
class CCopyOnWritePtr final {
public:
	CCopyOnWritePtr();
	CCopyOnWritePtr( const CCopyOnWritePtr& other );
	CCopyOnWritePtr( T* other );
	~CCopyOnWritePtr();

	const CCopyOnWritePtr& operator=( const CCopyOnWritePtr& );
	const CCopyOnWritePtr& operator=( T* );

	void Release();

	const T& operator*() const;
	const T* operator->() const;
	operator const T*() const { return ptr; }
	const T* Ptr() const { return ptr; }

	T* CopyOnWrite();

private:
	T* ptr;

	const CCopyOnWritePtr<T>& assignPtr( T* );
};

//---------------------------------------------------------------------------------------------

template<class T>
inline CCopyOnWritePtr<T>::CCopyOnWritePtr()
{
	ptr = 0;
}

template<class T>
inline CCopyOnWritePtr<T>::CCopyOnWritePtr( const CCopyOnWritePtr<T>& other )
{
	ptr = other.ptr;
	if( ptr != 0 ) {
		ptr->addRef();
	}
}

template<class T>
inline CCopyOnWritePtr<T>::CCopyOnWritePtr( T* other )
{
	ptr = other;
	if( ptr != 0 ) {
		ptr->addRef();
	}
}

template<class T>
inline CCopyOnWritePtr<T>::~CCopyOnWritePtr()
{
	Release();
}

template<class T>
inline const CCopyOnWritePtr<T>& CCopyOnWritePtr<T>::operator=( const CCopyOnWritePtr<T>& other )
{
	return assignPtr( other.ptr );
}

template<class T>
inline const CCopyOnWritePtr<T>& CCopyOnWritePtr<T>::operator=( T* other )
{
	return assignPtr( other );
}

template<class T>
inline void CCopyOnWritePtr<T>::Release()
{
	if( ptr != 0 ) {
		ptr->release();
	}
	ptr = 0;
}

template<class T>
inline const T& CCopyOnWritePtr<T>::operator*() const
{
	AssertFO( ptr != 0 );
	return *ptr;
}

template<class T>
inline const T* CCopyOnWritePtr<T>::operator->() const
{
	AssertFO( ptr != 0 );
	return ptr;
}

template<class T>
inline T* CCopyOnWritePtr<T>::CopyOnWrite()
{
	AssertFO( ptr != 0 );
	if( ptr->RefCount() != 1 ) {
		*this = ptr->Duplicate();
	}
	return ptr;
}

template<class T>
inline const CCopyOnWritePtr<T>& CCopyOnWritePtr<T>::assignPtr( T* newPtr )
{
	if( newPtr != 0 ) {
		newPtr->addRef();
	}
	T* oldPtr = ptr;
	ptr = newPtr;
	if( oldPtr != 0 ) {
		oldPtr->release();
	}
	return *this;
}

//------------------------------------------------------------------------------------------------

template<class T>
inline void ArrayMemMove( CPtr<T>* dest, CPtr<T>* source, int count )
{
	ArrayMemMoveBitwize( dest, source, count );
}

template<class T>
inline void ArrayMemMove( CCopyOnWritePtr<T>* dest, CCopyOnWritePtr<T>* source, int count )
{
	ArrayMemMoveBitwize( dest, source, count );
}

} // namespace FObj

