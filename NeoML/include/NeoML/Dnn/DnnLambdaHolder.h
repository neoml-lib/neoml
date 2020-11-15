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

//////////////////////////////////////////////////////////////////////////////////////////

// Simple analog for std::function() that does not use std::allocator

template<typename T>
class CLambdaHolderBase {};

// Base class for lambda holder. This interface hide actual lambda type.
template<typename Out, typename ...In>
class CLambdaHolderBase<Out( In... )> : public virtual IObject {
public:
	// Executes lambda.
	virtual void Execute( In... arguments ) = 0;
	// Copies lambda.
	virtual CPtr<CLambdaHolderBase<Out( In... )>> Copy() = 0;
};

template<typename T, typename U>
class CLambdaHolder {};

// Lambda holder implementation.
template<typename T, typename Out, typename ...In>
class CLambdaHolder<T, Out( In... )> : public CLambdaHolderBase<Out( In... )> {
public:
	CLambdaHolder( T _lambda ) : lambda( _lambda ) {}

	virtual void Execute( In... in )
		{ lambda( in... ); }

	virtual CPtr<CLambdaHolderBase<Out( In... )>> Copy()
		{ return new CLambdaHolder<T, Out( In... )>( lambda ); }

private:
	T lambda;
};

template<typename T>
class CLambda {};

// Type that captures lambda.
template<typename Out, typename ...In>
class CLambda<Out( In... )> {
public:
	CLambda() {}
	template<class T>
	CLambda( const T& t ) : lambda( new CLambdaHolder<T, Out( In... )>( t ) ) {}
	CLambda( const CLambda& other ) :
		lambda( other.lambda != 0 ? other.lambda->Copy() : nullptr ) {}

	Out operator()( In... in )
	{
		if( lambda != nullptr ) {
			return lambda->Execute( in... );
		}
	}

	// Is lambda undefined?
	bool IsEmpty() const { return lambda == nullptr; }

private:
	CPtr<CLambdaHolderBase<Out( In... )>> lambda;
};

//////////////////////////////////////////////////////////////////////////////////////////
} // namespace NeoML
