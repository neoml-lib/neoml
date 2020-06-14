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
#include <NeoML/TraditionalML/Function.h>
#include <NeoML/Random.h>
#include <float.h>
#include <limits.h>

namespace NeoML {

// Function parameter interface
class NEOML_API IParam : public IObject {
private:
	void dummy() { NeoAssert( false ); }
};

// Function parameter as a constant object
typedef CPtr<const IParam> CFunctionParam;

//////////////////////////////////////////////////////////////////////////////////
class NEOML_API CFunctionParamVectorBody : public IObject {
public:
	CArray<CFunctionParam> Values;
	CFunctionParamVectorBody( int size ) { Values.SetBufferSize( size ); Values.SetSize( size ); }

	CFunctionParamVectorBody* Duplicate() const
	{
		CFunctionParamVectorBody* ret = FINE_DEBUG_NEW CFunctionParamVectorBody( Values.Size() );
		Values.CopyTo( ret->Values );
		return ret;
	}
};

// Function parameter vector as a "constant" (copy-on-write) object
class NEOML_API CFunctionParamVector {
public:
	CFunctionParamVector() {}
	explicit CFunctionParamVector( int size ) : body( FINE_DEBUG_NEW CFunctionParamVectorBody( size ) ) {}
	CFunctionParamVector( int size, const CFunctionParam& param ) : body( FINE_DEBUG_NEW CFunctionParamVectorBody( size ) )
	{
		CArray<CFunctionParam>& bodyArr = CopyOnWrite();
		for( int i = 0; i < size; ++i ) {
			bodyArr[i] = param;
		}
	}

	bool IsNull() const { return body == 0; }
	int Size() const { return body->Values.Size(); }

	CFunctionParam operator [] ( int i ) const { return ( body->Values )[i]; }
	void SetAt( int i, const CFunctionParam& what ) { CopyOnWrite()[i] = what; }
	CArray<CFunctionParam>& CopyOnWrite() { return body.CopyOnWrite()->Values; }
	const CArray<CFunctionParam>&  GetArray() const { return body->Values; }

private:
	CCopyOnWritePtr<CFunctionParamVectorBody> body;
};

//////////////////////////////////////////////////////////////////////////////////
// Interface for working with the parameters in specific ways
class NEOML_API IParamTraits {
public:
	// Gets a random function parameter
	virtual CFunctionParam GenerateRandom( CRandom& random, const CFunctionParam& min, const CFunctionParam& max ) const;

	// Mutates a parameter (for the differential evolution algorithm)
	virtual CFunctionParam Mutate( CRandom& random, const CFunctionParam& base,
		const CFunctionParam& left, const CFunctionParam& right, double fluctuation,
		const CFunctionParam& min, const CFunctionParam& max ) const;

	// Gets the default minimum and maximum values for the parameter
	virtual CFunctionParam GetDefaultMin() const { NeoAssert(0); return 0; }
	virtual CFunctionParam GetDefaultMax() const { NeoAssert(0); return 0; }

	// Checks if the first parameter value is less than the second
	virtual bool Less( const CFunctionParam& left, const CFunctionParam& right ) const = 0;

	// Dumps the parameter value into a stream
	virtual void Dump( CTextStream& stream, const CFunctionParam& value ) const = 0;

	// Writing into CTextStream
	struct CDumper {
		CTextStream& stream;
		const IParamTraits& traits;

		CDumper( CTextStream& _stream, const IParamTraits& _traits ) : stream( _stream ), traits( _traits ) {}

		CTextStream& operator<<( const CFunctionParam& value )
		{
			traits.Dump( stream, value );
			return stream;
		}
	};
};

inline IParamTraits::CDumper operator<<( CTextStream& stream, const IParamTraits& traits )
{
	return IParamTraits::CDumper( stream, traits );
}

//////////////////////////////////////////////////////////////////////////////////
// The interface for evaluating the function on several parameter sets at the same time
// Implemented on the client side
class NEOML_API IFunctionEvaluation {
public:
	// The problem dimensions
	virtual int NumberOfDimensions() const = 0;

	// The parameter types
	virtual const IParamTraits& GetParamTraits( int index ) const = 0;
	// The return value type
	virtual const IParamTraits& GetResultTraits() const  = 0;

	// Returns the minimum and maximum values of the parameter vector
	virtual CFunctionParam GetMinConstraint( int index ) const = 0;
	virtual CFunctionParam GetMaxConstraint( int index ) const = 0;

	// You need to provide the implementation for at least one of the Evaluate functions,
	// because the default implementations call each other.
	// Evaluates the function on several parameter sets
	// The default implementation calls the Evaluate function on one parameter several times
	virtual void Evaluate( const CArray<CFunctionParamVector>& params, CArray<CFunctionParam>& results );
	// Evaluates the function on one parameter
	// The default implementation calls the Evaluate for several parameters
	virtual CFunctionParam Evaluate( const CFunctionParamVector& param );
};

//////////////////////////////////////////////////////////////////////////////////
// The implementation of a double parameter
class NEOML_API CDoubleTraits : public IParamTraits {
public:
	class CParam : public IParam {
	public:
		const double Value;
		CParam( double value = 0 ) : Value( value ) {}
	};

	static const CDoubleTraits& GetInstance();
	static CFunctionParam Box( double value ) { return FINE_DEBUG_NEW CParam( value ); }
	static double Unbox( const CFunctionParam& param )
	{
		NeoPresume( dynamic_cast<const CParam*>( param.Ptr() ) != 0 );
		return static_cast<const CParam*>( param.Ptr() )->Value;
	}

	virtual CFunctionParam GenerateRandom( CRandom& random, const CFunctionParam& min, const CFunctionParam& max ) const;
	virtual CFunctionParam Mutate( CRandom& random, const CFunctionParam& base,
		const CFunctionParam& left, const CFunctionParam& right, double fluctuation,
		const CFunctionParam& min, const CFunctionParam& max ) const;

	virtual CFunctionParam GetDefaultMin() const { return Box( -DBL_MAX ); }
	virtual CFunctionParam GetDefaultMax() const { return Box( DBL_MAX ); }

	virtual bool Less( const CFunctionParam& left, const CFunctionParam& right ) const;

	virtual void Dump( CTextStream& stream, const CFunctionParam& value ) const { stream << Unbox( value ); }
};


//////////////////////////////////////////////////////////////////////////////////
// The implementation of an int parameter
class NEOML_API CIntTraits : public IParamTraits {
public:
	class CParam : public IParam {
	public:
		const int Value;
		CParam( int value = 0 ) : Value( value ) {}
	};

	static const CIntTraits& GetInstance();
	static CFunctionParam Box( int value ) { return FINE_DEBUG_NEW CParam( value ); }
	static int Unbox( const CFunctionParam& param )
	{
		NeoPresume( dynamic_cast<const CParam*>( param.Ptr() ) != 0 );
		return static_cast<const CParam*>( param.Ptr() )->Value;
	}

	virtual CFunctionParam GenerateRandom( CRandom& random, const CFunctionParam& min, const CFunctionParam& max ) const;
	virtual CFunctionParam Mutate( CRandom& random, const CFunctionParam& base,
		const CFunctionParam& left, const CFunctionParam& right, double fluctuation,
		const CFunctionParam& min, const CFunctionParam& max ) const;

	virtual CFunctionParam GetDefaultMin() const { return Box( INT_MIN ); }
	virtual CFunctionParam GetDefaultMax() const { return Box( INT_MAX ); }

	virtual bool Less( const CFunctionParam& left, const CFunctionParam& right ) const;

	virtual void Dump( CTextStream& stream, const CFunctionParam& value ) const { stream << Unbox( value ); }
};

//////////////////////////////////////////////////////////////////////////////////
// The IFunctionEvaluation wrapper over IFunction
class NEOML_API CFunctionEvaluation : public IFunctionEvaluation {
public:
	explicit CFunctionEvaluation( CFunction& _func );

	// The problem dimensions
	virtual int NumberOfDimensions() const { return func.NumberOfDimensions(); }

	virtual CFunctionParam GetMinConstraint( int index ) const { return minConstraint[index]; }
	virtual CFunctionParam GetMaxConstraint( int index ) const { return maxConstraint[index]; }

	// The parameter types
	virtual const IParamTraits& GetParamTraits( int ) const { return CDoubleTraits::GetInstance(); }
	virtual const IParamTraits& GetResultTraits() const { return CDoubleTraits::GetInstance(); }

	// Evaluates the function on one parameter (the default implementation calls the function for several parameters)
	virtual CFunctionParam Evaluate( const CFunctionParamVector& param );

	// Sets the minimum and maximum for the i-th component of the parameter vector
	void SetMinConstraint( int i, double minBoundValue ) { minConstraint.SetAt( i, CDoubleTraits::Box( minBoundValue ) ); }
	void SetMaxConstraint( int i, double maxBoundValue ) { maxConstraint.SetAt( i, CDoubleTraits::Box( maxBoundValue ) ); }

private:
	CFunction& func;
	CFunctionParamVector minConstraint;
	CFunctionParamVector maxConstraint;
};

}
