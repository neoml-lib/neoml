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

template<class T>
class CPtr;

template<class T>
struct IsMemmoveable< CPtr<T> > {
	static constexpr bool Value = true;
};

//---------------------------------------------------------------------------------------------------------------------

// The base class for all interfaces with reference counting
// They should virtually inherit from this class and use CPtr for all pointers
// Thread-safe
class NEOML_API IObject {
public:
	int RefCount() const;

	virtual void Serialize( CArchive& ) { AssertFO( false ); }

	void ReadFromArchive( CArchive& archive );
	void WriteToArchive( CArchive& archive ) const;

protected:
	IObject() : refCounter( 0 ) {}
	// copy restricted
	IObject( const IObject& ) = delete;
	virtual ~IObject();

	// copy restricted
	IObject& operator=( const IObject& ) = delete;
	
private:
	mutable std::atomic_int refCounter;

	void addRef() const;
	void release() const;
	void detach();
	bool weakAddRef() const;

	template<class T>
	friend class CPtr;
	
	template<class T>
	friend class CCopyOnWritePtr;
};

//---------------------------------------------------------------------------------------------------------------------

// Ananlog IObject
// Non-thread-safe
class CFastObject {
public:
	int RefCount() const;

protected:
	CFastObject() : refCounter( 0 ) {}
	// copy restricted
	CFastObject( const CFastObject& ) = delete;
	virtual ~CFastObject();

	// copy restricted
	CFastObject& operator=( const CFastObject& ) = delete;

private:
	mutable int refCounter;
	
	void addRef() const;
	void release() const;
	void detach();
	bool weakAddRef() const;

	template<class T>
	friend class CPtr;
	
	template<class T>
	friend class CCopyOnWritePtr;
};

//---------------------------------------------------------------------------------------------------------------------

// Smart pointer template for the interfaces that inherit from IObject
template<class T>
class CPtr final {
public:
	using TElement = T;

	CPtr() noexcept = default;
	
	CPtr( std::nullptr_t ) noexcept : CPtr() {}

	template <typename U, std::enable_if_t<std::is_convertible<U*, T*>::value, bool> = true>
	CPtr( U* ) noexcept;

	CPtr( const CPtr& ) noexcept;

	template <typename U, std::enable_if_t<std::is_convertible<U*, T*>::value, bool> = true>
	CPtr( const CPtr<U>& ) noexcept;

	CPtr( CPtr&& ) noexcept;

	template <typename U, std::enable_if_t<std::is_convertible<U*, T*>::value, bool> = true>
	CPtr( CPtr<U>&& ) noexcept;

	~CPtr();

	void Release();

	// Binds a weak pointer to the object
	// Returns true if the pointer has been successfully bound
	// and false otherwise (because weakPtr is 0 or the object is already being destroyed)
	bool PinWeakPtr( T* weakPtr );

	int HashKey() const { return static_cast<int>( reinterpret_cast<UINT_PTR>( ptr ) ); }

	auto operator*() const -> T&;
	auto operator->() const -> T*;
	auto Ptr() const -> T* { return ptr; }
	operator T*() const { return ptr; }

	auto operator =( const CPtr& ) noexcept -> CPtr&;
	auto operator =( CPtr&& ) noexcept -> CPtr&;

	template <typename U, std::enable_if_t<std::is_convertible<U*, T*>::value, bool> = true>
	CPtr& operator=( const CPtr<U>& );

	template <typename U, std::enable_if_t<std::is_convertible<U*, T*>::value, bool> = true>
	CPtr& operator=( CPtr<U>&& );

	void Swap( CPtr& ) noexcept;

private:
	T* ptr = nullptr;

	auto assignPtr( T* ) -> CPtr&;
	void replacePtr( T* );

	template<class U>
	friend class CPtr;
};

//---------------------------------------------------------------------------------------------------------------------

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

//---------------------------------------------------------------------------------------------------------------------

template <typename T>
bool operator==( const CPtr<T>& p, std::nullptr_t ) noexcept
{
	return p.Ptr() == nullptr;
}

template <typename T>
bool operator!=( const CPtr<T>& p, std::nullptr_t ) noexcept
{
	return !( p == nullptr );
}

//---------------------------------------------------------------------------------------------------------------------

template<class T>
class CCopyOnWritePtr {
public:
	using TElement = T;

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

//---------------------------------------------------------------------------------------------------------------------

// Implementation IObject

// NeoML.cpp
//inline IObject::~IObject()
//{
//	PresumeFO( refCounter == 0 );
//}

inline int IObject::RefCount() const
{
	return refCounter;
}

inline void IObject::addRef() const
{
	PresumeFO( refCounter >= 0 );
	PresumeFO( refCounter < INT_MAX );
	refCounter++;
}

inline void IObject::release() const
{
	PresumeFO( refCounter > 0 );
	if( refCounter.fetch_sub( 1 ) == 1 ) {
		delete this;
	}
}

inline void IObject::detach()
{
	refCounter.exchange( 0 );
}

// Increment the reference count given only a weak reference to the object.
// Returns false if the reference count is 0 (the object is already in the destructor).
inline bool IObject::weakAddRef() const
{
	while( 1 ) {
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

//---------------------------------------------------------------------------------------------------------------------

// Implementation CFastObject

inline CFastObject::~CFastObject()
{
	PresumeFO( refCounter == 0 );
}

inline void CFastObject::addRef() const
{
	PresumeFO( refCounter >= 0 );
	PresumeFO( refCounter < INT_MAX );
	refCounter++;
}

inline void CFastObject::release() const
{
	PresumeFO( refCounter > 0 );
	if( --refCounter == 0 ) {
		delete this;
	}
}

inline void CFastObject::detach()
{
	refCounter = 0;
}

inline bool CFastObject::weakAddRef() const
{
	PresumeFO( refCounter > 0 );
	addRef();
	return true;
}

inline int CFastObject::RefCount() const
{
	return refCounter;
}

//---------------------------------------------------------------------------------------------------------------------

// Implementation CPtr

template <typename T>
template <typename U, std::enable_if_t<std::is_convertible<U*, T*>::value, bool>>
CPtr<T>::CPtr( U* _ptr ) noexcept :
	ptr( _ptr )
{
	if( _ptr != nullptr ) {
		_ptr->addRef();
	}
}

template <typename T>
template <typename U, std::enable_if_t<std::is_convertible<U*, T*>::value, bool>>
CPtr<T>::CPtr( const CPtr<U>& other ) noexcept :
	CPtr( other.ptr )
{}

template<class T>
CPtr<T>::CPtr( const CPtr<T>& other ) noexcept :
	ptr( other.ptr )
{
	if( ptr != nullptr ) {
		ptr->addRef();
	}
}

template<class T>
CPtr<T>::CPtr( CPtr&& other ) noexcept
{
	Swap( other );
}

template <typename T>
template <typename U, std::enable_if_t<std::is_convertible<U*, T*>::value, bool>>
CPtr<T>::CPtr( CPtr<U>&& other ) noexcept
{
	ptr = other.ptr;
	other.ptr = nullptr;
}

template<class T>
CPtr<T>::~CPtr()
{
	Release();
}

template<class T>
void CPtr<T>::Release()
{
	T* tempPtr = ptr;
	if( tempPtr != nullptr ) {
		ptr = nullptr;
		tempPtr->release();
	}
}

template<class T>
auto CPtr<T>::operator*() const -> T&
{
	AssertFO( ptr != nullptr );
	return *ptr;
}

template<class T>
auto CPtr<T>::operator->() const -> T*
{
	AssertFO( ptr != nullptr );
	return ptr;
}

template<class T>
auto CPtr<T>::operator =( const CPtr& newPtr ) noexcept -> CPtr&
{
	return assignPtr( newPtr.ptr );
}

template<class T>
auto CPtr<T>::operator =( CPtr&& newPtr ) noexcept -> CPtr&
{
	Swap( newPtr );
	return *this;
}

template <class T>
template <typename U, std::enable_if_t<std::is_convertible<U*, T*>::value, bool>>
CPtr<T>& CPtr<T>::operator=( const CPtr<U>& newPtr )
{
	return assignPtr( newPtr.ptr );
}

template <class T>
template <typename U, std::enable_if_t<std::is_convertible<U*, T*>::value, bool>>
CPtr<T>& CPtr<T>::operator=( CPtr<U>&& newPtr )
{
	T* oldPtr = ptr;
	ptr = newPtr.ptr;
	newPtr.ptr = nullptr;
	if( oldPtr != nullptr ) {
		oldPtr->release();
	}
	return *this;
}

template<class T>
void CPtr<T>::Swap( CPtr& other ) noexcept
{
	swap( *this, other );
}

template<class T>
bool CPtr<T>::PinWeakPtr( T* weakPtr )
{
	if( weakPtr == nullptr || !weakPtr->weakAddRef() ) {
		return false;
	}

	replacePtr( weakPtr );
	return true;
}

template<class T>
auto CPtr<T>::assignPtr( T* newPtr ) -> CPtr&
{
	if( newPtr != nullptr ) {
		newPtr->addRef();
	}
	replacePtr( newPtr );
	return *this;
}

template<class T>
void CPtr<T>::replacePtr( T* newPtr )
{
	// Assigning a new value must be done before calling release on the old pointer
	// so that in case of an exception when destroying the old object, CPtr remains in the correct state
	T* oldPtr = ptr;
	ptr = newPtr;
	if( oldPtr != nullptr ) {
		oldPtr->release();
	}
}

//---------------------------------------------------------------------------------------------------------------------

// Implementation CCopyOnWritePtr

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
	// Assigning a new value must be done before calling release on the old pointer
	// so that in case of an exception when destroying the old object, CPtr remains in the correct state
	T* oldPtr = ptr;
	ptr = newPtr;
	if( oldPtr != 0 ) {
		oldPtr->release();
	}
	return *this;
}

//---------------------------------------------------------------------------------------------------------------------

// CPtr and CCopyOnWritePtr allow bitwise movement in memory.
// Therefore, you can write a corresponding specialization of ArrayMemMove

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

//---------------------------------------------------------------------------------------------------------------------

// Function for creating an IObject descendant object and wrapping it in CPtr
// Example: const auto ptr = MakeCPtr<CMyType>( param1, param2, param3 );
template<class T, class... Ts, typename = std::enable_if_t<std::is_base_of<IObject, T>::value>>
CPtr<T> MakeCPtr( Ts&&... params ) noexcept( std::is_nothrow_constructible<T, Ts...>::value )
{
	return { new T( std::forward<Ts>( params ) ... ) };
}

} // namespace FObj
