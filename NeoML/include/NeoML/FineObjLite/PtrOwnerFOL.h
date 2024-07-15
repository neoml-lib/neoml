/* Copyright © 2024 ABBYY

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

template<class T, class DELETER>
class CPtrOwner;

template<class T, class DELETER>
struct IsMemmoveable< CPtrOwner<T, DELETER> > {
	static constexpr bool Value = false;
};

namespace DetailsCPtrOwner {
	template<class TBase, class TDerived>
	struct IsSafeConvertiblePointerType {
		static constexpr bool Value =
			std::is_convertible<TDerived*, TBase*>::value
			&& !std::is_array<TDerived>::value
			&& !std::is_array<TBase>::value
			&& ( std::has_virtual_destructor<TBase>::value || std::is_trivially_destructible<TDerived>::value );
	};

	template<class TBase, class TDerived>
	static constexpr bool IsSafeConvertiblePointerTypeValue =
		IsSafeConvertiblePointerType<TBase, TDerived>::Value;

	template<class T>
	struct DefaultDeleter {
		constexpr DefaultDeleter() noexcept = default;

		// Allow conversion from derived to base
		template< class TOther,
			typename = std::enable_if_t<std::is_convertible<TOther*, T*>::value> >
		DefaultDeleter( const DefaultDeleter<TOther>& ) noexcept {}

		void operator() ( T* value ) { delete value; }
	};

	template<class T>
	struct DefaultDeleter<T[]> {
		void operator() ( T* value ) { delete [] value; }
	};
}

template< class TYPE, class DELETER = DetailsCPtrOwner::DefaultDeleter<TYPE> >
class CPtrOwner : private DELETER {
public:
	using TElement = std::remove_all_extents_t<TYPE>;
	using TDeleter = DELETER;

	explicit CPtrOwner( TElement* ptr = nullptr );
	CPtrOwner( CPtrOwner&& ) noexcept;

	template<class OTHER_TYPE, class OTHER_DELETER>
	CPtrOwner( CPtrOwner<OTHER_TYPE, OTHER_DELETER>&& ) noexcept;

	template<class OTHER_TYPE>
	CPtrOwner( CPtr<OTHER_TYPE>&& ) = delete;
	template<class OTHER_TYPE>
	CPtrOwner( const CPtrOwner<OTHER_TYPE>& ) = delete;

	~CPtrOwner();

	template<class OTHER_TYPE>
	auto operator =( const CPtrOwner<OTHER_TYPE>& ) -> CPtrOwner& = delete;

	template<class OTHER_TYPE>
	auto operator =( const CPtr<OTHER_TYPE>& ) -> CPtrOwner& = delete;

	template<class OTHER_TYPE>
	auto operator =( CPtr<OTHER_TYPE>&& ) -> CPtrOwner& = delete;

	auto operator =( TElement* other ) -> CPtrOwner&;
	auto operator =( CPtrOwner&& ) noexcept -> CPtrOwner&;

	// Allow CPtrOwner<Base> base = std::move( derived );
	template<class OTHER_TYPE, class OTHER_DELETER>
	auto operator =( CPtrOwner<OTHER_TYPE, OTHER_DELETER>&& ) noexcept -> CPtrOwner&;

	void Release();
	auto Detach() -> TElement*;

	bool IsNull() const;

	TElement* Ptr() { return ptr; }
	const TElement* Ptr() const { return ptr; }

	auto SafePtr() -> TElement*;
	auto SafePtr() const -> const TElement*;

	operator TElement* () { return ptr; }
	operator const TElement* () const { return ptr; }

	auto operator * () -> TElement&;
	auto operator * () const -> const TElement&;

	auto operator -> () -> TElement*;
	auto operator -> () const -> const TElement*;

private:
	TElement* ptr = nullptr;
};

template<class TYPE, class DELETER>
CPtrOwner<TYPE, DELETER>::CPtrOwner( TElement* ptr ) :
	ptr( ptr )
{
}

template<class TYPE, class DELETER>
CPtrOwner<TYPE, DELETER>::CPtrOwner( CPtrOwner&& other ) noexcept
{
	ptr = other.Detach();
}

template<class TYPE, class DELETER>
template<class OTHER_TYPE, class OTHER_DELETER>
CPtrOwner<TYPE, DELETER>::CPtrOwner( CPtrOwner<OTHER_TYPE, OTHER_DELETER>&& other ) noexcept
{
	static_assert( DetailsCPtrOwner::IsSafeConvertiblePointerTypeValue<TYPE, OTHER_TYPE>,
		"Pointer types are not safe to convert") ;

	ptr = other.Detach();
}

template<class TYPE, class DELETER>
CPtrOwner<TYPE, DELETER>::~CPtrOwner()
{
	Release();
}

template<class TYPE, class DELETER>
auto CPtrOwner<TYPE, DELETER>::operator =( TElement* other ) -> CPtrOwner&
{
	DELETER::operator() ( ptr );
	ptr = other;
	return *this;
}

template<class TYPE, class DELETER>
auto CPtrOwner<TYPE, DELETER>::operator =( CPtrOwner&& other ) noexcept -> CPtrOwner&
{
	if( this == &other ) {
		return *this;
	}
	Release();
	ptr = other.Detach();
	return *this;
}

template<class TYPE, class DELETER>
template<class OTHER_TYPE, class OTHER_DELETER>
auto CPtrOwner<TYPE, DELETER>::operator =( CPtrOwner<OTHER_TYPE, OTHER_DELETER>&& other ) noexcept -> CPtrOwner&
{
	static_assert( DetailsCPtrOwner::IsSafeConvertiblePointerTypeValue<TYPE, OTHER_TYPE>,
		"Pointer types are not safe to convert") ;

	Release();
	ptr = other.Detach();
	return *this;
}

template<class TYPE, class DELETER>
void CPtrOwner<TYPE, DELETER>::Release()
{
	DELETER::operator() ( Detach() );
}

template<class TYPE, class DELETER>
auto CPtrOwner<TYPE, DELETER>::Detach() -> TElement*
{
	TElement* result = ptr;
	ptr = nullptr;
	return result;
}

template<class TYPE, class DELETER>
bool CPtrOwner<TYPE, DELETER>::IsNull() const
{
	return ptr == nullptr;
}

template<class TYPE, class DELETER>
auto CPtrOwner<TYPE, DELETER>::SafePtr() -> TElement*
{
	assert( ptr != nullptr );
	return ptr;
}

template<class TYPE, class DELETER>
auto CPtrOwner<TYPE, DELETER>::SafePtr() const -> const TElement*
{
	assert( ptr != nullptr );
	return ptr;
}

template<class TYPE, class DELETER>
auto CPtrOwner<TYPE, DELETER>::operator * () -> TElement&
{
	assert( ptr != nullptr );
	return *ptr;
}

template<class TYPE, class DELETER>
auto CPtrOwner<TYPE, DELETER>::operator * () const -> const TElement&
{
	assert( ptr != nullptr );
	return *ptr;
}

template<class TYPE, class DELETER>
auto CPtrOwner<TYPE, DELETER>::operator -> () -> TElement*
{
	assert( ptr != nullptr );
	return ptr;
}

template<class TYPE, class DELETER>
auto CPtrOwner<TYPE, DELETER>::operator -> () const -> const TElement*
{
	assert( ptr != nullptr );
	return ptr;
}

template<class T, class... Ts>
CPtrOwner<T> MakeCPtrOwner( Ts&&... params )
{
	return CPtrOwner<T>( new T( std::forward<Ts>( params ) ... ) );
}

} // namespace FObj
