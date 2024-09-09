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

#include <type_traits>
#include <NeoML/NeoMLDefs.h>

namespace NeoML {

// Simple analog for std::function() that does not use std::allocator

namespace details {

template<typename T>
class ILambdaHolderBase;

// Base class for lambda holder. This interface hide actual lambda type.
template<typename Out, typename ...In>
class ILambdaHolderBase<Out( In... )> : public virtual IObject {
public:
	// Executes lambda.
	virtual Out Execute( In... arguments ) = 0;
};

//---------------------------------------------------------------------------------------------------------------------

template<typename T, typename U>
class CLambdaHolder;

// Lambda holder implementation.
template<typename F, typename Out, typename ...In>
class CLambdaHolder<F, Out( In... )> : public ILambdaHolderBase<Out( In... )> {
public:
	CLambdaHolder( F&& func ) : lambda( std::move( func ) ) {}
	CLambdaHolder( const F& func ) : lambda( func ) {}

	Out Execute( In... in ) override
		{ return lambda( in... ); }

private:
	F lambda;
};

} // namespace details

//---------------------------------------------------------------------------------------------------------------------

template<typename T>
class CLambda;

// Type that captures lambda.
template<typename Out, typename ...In>
class CLambda<Out( In... )> {
public:
	CLambda() = default;
	// Be copied and moved by default, because it stores the shared pointer

	// Convert from a function, except itself type
	// By coping
	template<class F,
		typename std::enable_if<!std::is_same<CLambda, typename std::decay<F>::type>::value, int>::type = 0>
	CLambda( const F& function ) :
		lambda( new details::CLambdaHolder<F, Out( In... )>( function ) ) {}
	// By moving
	template<class F,
		typename std::enable_if<!std::is_same<CLambda, typename std::decay<F>::type>::value, int>::type = 0>
	CLambda( F&& function ) :
		lambda( new details::CLambdaHolder<F, Out( In... )>( std::move( function ) ) ) {}

	Out operator()( In... in )
	{
		NeoAssert( !IsEmpty() );
		return lambda->Execute( in... );
	}

	// Is lambda undefined?
	bool IsEmpty() const { return lambda == nullptr; }

private:
	CPtr<details::ILambdaHolderBase<Out( In... )>> lambda;
};

} // namespace NeoML
