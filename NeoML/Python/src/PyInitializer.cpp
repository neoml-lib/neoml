/* Copyright © 2017-2021 ABBYY Production LLC

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

#include <common.h>
#pragma hdrstop

#include "PyInitializer.h"

std::string CPyInitializer::GetClassName() const
{
	if( Initializer<CDnnXavierInitializer>() != 0 ) {
		return "Xavier";
	}
	if( Initializer<CDnnXavierUniformInitializer>() != 0 ) {
		return "XavierUniform";
	}
	if( Initializer<CDnnUniformInitializer>() != 0 ) {
		return "Uniform";
	}
	assert( false );
	return "";
}

//------------------------------------------------------------------------------------------------------------

class CPyXavierInitializer : public CPyInitializer {
public:
	explicit CPyXavierInitializer( CPyRandomOwner& _randomOwner, CDnnInitializer* _initializer ) :
		CPyInitializer( _randomOwner, _initializer )
	{
	}
};

//------------------------------------------------------------------------------------------------------------

class CPyXavierUniformInitializer : public CPyInitializer {
public:
	explicit CPyXavierUniformInitializer( CPyRandomOwner& _randomOwner, CDnnInitializer* _initializer ) :
		CPyInitializer( _randomOwner, _initializer )
	{
	}
};

//------------------------------------------------------------------------------------------------------------

class CPyUniformInitializer : public CPyInitializer {
public:
	explicit CPyUniformInitializer( CPyRandomOwner& _randomOwner, CDnnInitializer* _initializer ) :
		CPyInitializer( _randomOwner, _initializer )
	{
	}

	float GetLowerBound() const { return Initializer<CDnnUniformInitializer>()->GetLowerBound(); }
	void SetLowerBound(float _lowerBound) { Initializer<CDnnUniformInitializer>()->SetLowerBound( _lowerBound ); }

	float GetUpperBound() const { return Initializer<CDnnUniformInitializer>()->GetUpperBound(); }
	void SetUpperBound(float _upperBound) { Initializer<CDnnUniformInitializer>()->SetUpperBound( _upperBound ); }
};

//------------------------------------------------------------------------------------------------------------

void InitializeInitializer( py::module& m )
{
	py::class_<CPyInitializer>(m, "Initializer")
		.def( "get_random", &CPyInitializer::GetRandom, py::return_value_policy::reference )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyXavierInitializer, CPyInitializer>(m, "Xavier")
		.def( py::init([]( const CPyRandom& random ) {
			CPtr<CDnnXavierInitializer> initializer( new CDnnXavierInitializer( random.Random() ) );
			return new CPyXavierInitializer( random.RandomOwner(), initializer );
		}) )
		.def( py::init([]( const CPyInitializer& initializer )
		{
			return new CPyXavierInitializer( initializer.RandomOwner(), initializer.Initializer<CDnnXavierInitializer>() );
		}) )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyXavierUniformInitializer, CPyInitializer>(m, "XavierUniform")
		.def( py::init([]( const CPyRandom& random ) {
			CPtr<CDnnXavierUniformInitializer> initializer( new CDnnXavierUniformInitializer( random.Random() ) );
			return new CPyXavierUniformInitializer( random.RandomOwner(), initializer );
		}) )
		.def( py::init([]( const CPyInitializer& initializer )
		{
			return new CPyXavierUniformInitializer( initializer.RandomOwner(), initializer.Initializer<CDnnXavierUniformInitializer>() );
		}) )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyUniformInitializer, CPyInitializer>(m, "Uniform")
		.def( py::init([]( float lowerBound, float upperBound, const CPyRandom& random ) {
			CPtr<CDnnUniformInitializer> initializer( new CDnnUniformInitializer( random.Random(), lowerBound, upperBound ) );
			return new CPyUniformInitializer( random.RandomOwner(), initializer );
		}) )
		.def( py::init([]( const CPyInitializer& initializer )
		{
			return new CPyUniformInitializer( initializer.RandomOwner(), initializer.Initializer<CDnnUniformInitializer>() );
		}) )

		.def( "get_lower_bound", &CPyUniformInitializer::GetLowerBound, py::return_value_policy::reference )
		.def( "set_lower_bound", &CPyUniformInitializer::SetLowerBound, py::return_value_policy::reference )
		.def( "get_upper_bound", &CPyUniformInitializer::GetUpperBound, py::return_value_policy::reference )
		.def( "set_upper_bound", &CPyUniformInitializer::SetUpperBound, py::return_value_policy::reference )
	;
}
