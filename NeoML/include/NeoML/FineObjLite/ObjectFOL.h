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

	template<class T>
	friend class CPtr;
	
	template<class T>
	friend class CCopyOnWritePtr;
};

//---------------------------------------------------------------------------------------------

// Smart pointer template for the interfaces that inherit from IObject
template<class T>
class CPtr {
public:
	CPtr();
	CPtr( T* );
	CPtr( const CPtr& );
	~CPtr();

	void Release();

	// Binds a weak pointer to the object
	// Returns true if the pointer has been successfully bound
	// and false otherwise (because weakPtr is 0 or the object is already being destroyed)
	bool PinWeakPtr( T* weakPtr );

	int HashKey() const;

	T& operator*() const;
	T* operator->() const;
	operator T*() const { return ptr; }
	T* Ptr() const { return ptr; }

	const CPtr<T>& operator =( T* );
	const CPtr<T>& operator =( const CPtr<T>& );

	void Swap( CPtr<T>& );

private:
	T* ptr;

	const CPtr<T>& assignPtr( T* );
	void replacePtr( T* newPtr );
};

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

inline IObject::IObject() : refCounter( 0 )
{
}
    
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

//---------------------------------------------------------------------------------------------

template<class T>
inline CPtr<T>::CPtr()
{
	ptr = 0;
}

template<class T>
inline CPtr<T>::CPtr( T* _ptr )
{
	ptr = _ptr;
	if( ptr != 0 ) {
		ptr->addRef();
	}
}

template<class T>
inline CPtr<T>::CPtr( const CPtr<T>& _ptr )
{
	ptr = _ptr.ptr;
	if( ptr != 0 ) {
		ptr->addRef();
	}
}

template<class T>
inline CPtr<T>::~CPtr()
{
	Release();
}

template<class T>
inline void CPtr<T>::Release()
{
	T* tempPtr = ptr;
	if( tempPtr != 0 ) {
		ptr = 0;
		tempPtr->release();
	}
}

template<class T>
inline T& CPtr<T>::operator*() const
{
	AssertFO( ptr != 0 );
	return *ptr;
}

template<class T>
inline T* CPtr<T>::operator->() const
{
	AssertFO( ptr != 0 );
	return ptr;
}

template<class T>
inline const CPtr<T>& CPtr<T>::operator =( T* newPtr )
{
	return assignPtr( newPtr );
}

template<class T>
inline const CPtr<T>& CPtr<T>::operator =( const CPtr<T>& newPtr )
{
	return assignPtr( newPtr.ptr );
}

template<class T>
inline void CPtr<T>::Swap( CPtr<T>& other )
{
	std::swap<T*>( ptr, other.ptr );
}

template<class T>
inline bool CPtr<T>::PinWeakPtr( T* weakPtr )
{
	if( weakPtr == 0 || !weakPtr->weakAddRef() ) {
		return false;
	}

	replacePtr( weakPtr );
	return true;
}

template<class T>
inline int CPtr<T>::HashKey() const
{
	return static_cast<int>(reinterpret_cast<UINT_PTR>(ptr));
}

template<class T>
inline const CPtr<T>& CPtr<T>::assignPtr( T* newPtr )
{
	if( newPtr != 0 ) {
		newPtr->addRef();
	}
	replacePtr( newPtr );
	return *this;
}

template<class T>
inline void CPtr<T>::replacePtr( T* newPtr )
{
	T* oldPtr = ptr;
	ptr = newPtr;
	if( oldPtr != 0 ) {
		oldPtr->release();
	}
}

//---------------------------------------------------------------------------------------------

template<class T>
class CCopyOnWritePtr {
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

